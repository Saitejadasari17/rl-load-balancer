"""
Deep Q-Network Agent for Adaptive Load Balancing.

This is the "production-style" agent described in the project roadmap:
    PyTorch (autograd, GPU/CUDA/Apple-MPS/CPU support)
    Double DQN            -- decouples action selection from action evaluation
    Dueling architecture  -- separate state-value and action-advantage streams
    Prioritized Experience Replay (proportional, sum-tree backed)
    Soft ("Polyak") target updates instead of periodic hard copies
    AdamW optimizer + Huber (smooth L1) loss
    Gradient clipping
    Learning-rate scheduling
    Mixed-precision training on CUDA
    Checkpointing (resumable) + TensorBoard logging
    Early stopping
    Reproducibility via fixed seeds

It replaces the earlier NumPy-only manual-backprop implementation. The
public surface (`DQNAgent.act/predict/remember/train_step/save/load`,
`train_rl_agent`, `evaluate_rl_agent`, `load_rl_agent`) is kept compatible
with `src/baselines.py`, `src/evaluation.py` and `backend/main.py` so
nothing downstream needs to change.
"""

import os
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .environment import LoadBalancerEnv
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:  # tensorboard is an optional dependency
    _TENSORBOARD_AVAILABLE = False


# ──────────────────────────────────────────────
#  Reproducibility & device helpers
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """Fix every RNG we touch so training runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred=None):
    """Pick the best available compute device: CUDA > Apple MPS > CPU."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────
#  Dueling Q-Network
# ──────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Shared feature trunk feeding two heads:
      - V(s): a single scalar estimate of how good the state is
      - A(s, a): the relative advantage of each action in that state
    Combined as Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)), which is the
    standard identifiable Dueling-DQN aggregation (Wang et al., 2016).
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        half = max(hidden_size // 2, action_size, 8)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(inplace=True),
            nn.Linear(half, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(inplace=True),
            nn.Linear(half, action_size),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ──────────────────────────────────────────────
#  Prioritized Experience Replay (sum-tree backed)
# ──────────────────────────────────────────────

class SumTree:
    """
    Binary tree where each leaf holds a transition's priority and each
    internal node holds the sum of its children. This gives O(log n)
    weighted sampling and O(log n) priority updates, instead of the O(n)
    cost of re-weighting a flat array on every sample -- the difference
    that matters once the buffer holds tens of thousands of transitions.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        # Iterative walk up to the root; avoids per-level Python call overhead
        # (this runs once per sampled/updated transition, i.e. batch_size times
        # per training step).
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        tree_len = len(self.tree)
        while True:
            left = 2 * idx + 1
            if left >= tree_len:
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Proportional PER (Schaul et al., 2016): transitions with larger TD
    error are sampled more often, since they carry more learning signal.
    Importance-sampling weights correct the bias this introduces, with
    beta annealed from `beta_start` towards 1.0 over `beta_frames` samples.
    """

    def __init__(self, capacity=20000, alpha=0.6, beta_start=0.4, beta_frames=100_000, eps=1e-5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = eps
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        # New transitions get max priority so they're guaranteed to be sampled at least once.
        self.tree.add(self.max_priority ** self.alpha, experience)

    def _beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        beta = self._beta()
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        probs = np.asarray(priorities, dtype=np.float64) / max(self.tree.total(), 1e-8)
        weights = np.power(self.tree.n_entries * probs + 1e-8, -beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(float(td_error)) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ──────────────────────────────────────────────
#  DQN Agent
# ──────────────────────────────────────────────

# kwargs accepted by DQNAgent's constructor -- used to filter **kwargs coming
# from config files / API payloads without raising on unrelated keys.
_AGENT_KWARGS = {
    "hidden_size", "lr", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay_steps",
    "buffer_size", "batch_size", "tau", "per_alpha", "per_beta_start", "per_beta_frames",
    "grad_clip", "weight_decay", "device", "seed", "use_amp", "train_every", "warmup_steps",
    "sticky_min", "sticky_max",
}


class DQNAgent:
    """
    Dueling Double DQN with Prioritized Experience Replay.

    - Double DQN: the online network picks the next action, the target
      network evaluates it. This decoupling removes the systematic
      Q-value overestimation of vanilla DQN.
    - Soft target updates: after every training step the target network
      is nudged a little (`tau`) towards the online network instead of
      being hard-copied every N episodes, giving smoother, more stable
      targets.
    - AdamW + Huber loss + gradient clipping + LR scheduling: standard
      stabilizers for TD learning with function approximation.
    - Mixed precision (autocast + GradScaler) automatically activates on
      CUDA GPUs for faster training; it's a no-op on CPU/MPS.
    """

    def __init__(self, state_size=9, action_size=3, hidden_size=128,
                 lr=5e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay_steps=50_000, buffer_size=20000, batch_size=64,
                 tau=0.005, per_alpha=0.4, per_beta_start=0.4, per_beta_frames=100_000,
                 grad_clip=10.0, weight_decay=1e-5, device=None, seed=42, use_amp=None,
                 train_every=2, warmup_steps=2000, sticky_min=5, sticky_max=15):

        set_seed(seed)
        self.seed = seed

        # gymnasium's Discrete.n (and some numpy-derived shapes) come through as
        # numpy integer types; normalize to plain Python ints so checkpoints stay
        # loadable with torch's default `weights_only=True` (which only allowlists
        # plain Python/NumPy array types, not numpy scalar objects).
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.gamma = gamma

        # Epsilon is annealed linearly over a fixed number of *environment steps*
        # rather than per episode. Per-episode decay (the earlier implementation)
        # is tightly coupled to episode length and episode count: with short
        # training budgets (tens to low hundreds of episodes) it barely decays at
        # all and the agent never stops behaving mostly randomly. A step-based
        # schedule reaches epsilon_end predictably after epsilon_decay_steps
        # environment steps, independent of how episodes are chunked.
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.epsilon = epsilon_start
        self._explore_steps = 0

        self.batch_size = batch_size
        # Don't start bootstrapping Q-value targets off a handful of samples:
        # wait until the buffer holds at least `warmup_steps` (mostly-random,
        # since epsilon starts at 1.0) transitions. Training from a nearly
        # empty, low-diversity buffer is a common cause of DQN prematurely
        # locking onto a degenerate policy (e.g. always routing to a single
        # server in this environment) before it has seen enough of the state
        # space to know better -- this mirrors the "replay start size" warm-up
        # used in the original Nature DQN paper.
        self.warmup_steps = max(batch_size, int(warmup_steps))
        self.tau = tau
        self.grad_clip = grad_clip

        # Sticky exploration: when a random action is chosen, commit to it for
        # a burst of sticky_min..sticky_max consecutive steps rather than just
        # one (see act() for why this matters here specifically).
        self.sticky_min = max(1, int(sticky_min))
        self.sticky_max = max(self.sticky_min, int(sticky_max))
        self._sticky_remaining = 0
        self._sticky_action = 0


        self.device = device if isinstance(device, torch.device) else get_device(device)
        self.use_amp = (self.device.type == "cuda") if use_amp is None else use_amp

        self.q_network = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Gently decays the learning rate as training progresses (per optimizer step,
        # not per episode, so it scales naturally with however many steps a run takes).
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99995)
        self.min_lr = lr * 0.05

        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=per_alpha, beta_start=per_beta_start, beta_frames=per_beta_frames
        )

        # Gradient updates are the expensive part of each environment step (network
        # forward + backward + optimizer step dominates over everything else). Only
        # doing one every `train_every` environment steps -- rather than after every
        # single step -- cuts wall-clock training time roughly proportionally with
        # only a minor, well-established sample-efficiency cost (the same trick the
        # original Nature DQN paper uses via its update_frequency=4 setting).
        self.train_every = max(1, int(train_every))
        self._steps_since_train = 0

        self.training_losses = []
        self.episode_rewards = []
        self.train_steps_done = 0

    # ---- acting ----------------------------------------------------

    def act(self, state, training=True):
        """
        Choose an action with sticky epsilon-greedy exploration:
        - With probability epsilon: pick a random server and commit to it for
          several consecutive steps (a "sticky" burst), not just one.
        - Otherwise: pick the highest-Q server (exploit).

        Plain per-step (i.i.d.) epsilon-greedy exploration only ever deviates
        for isolated single steps. In this environment that's not enough to
        escape a self-reinforcing "always route to server k" policy: a lone
        exploratory step barely dents a saturated server's queue (which only
        decays a little per step), so the agent never actually experiences
        the multi-step payoff of sustained balanced routing -- it only ever
        sees single-step blips that look like noise. Committing to an
        exploratory action for a short burst lets that payoff actually show
        up in the states the agent visits (and therefore in what gets stored
        for training), which single-step exploration structurally cannot
        provide regardless of how large epsilon is.
        """
        if training:
            if self._sticky_remaining > 0:
                self._sticky_remaining -= 1
                return self._sticky_action
            if np.random.random() < self.epsilon:
                self._sticky_action = np.random.randint(0, self.action_size)
                self._sticky_remaining = np.random.randint(self.sticky_min - 1, self.sticky_max)
                return self._sticky_action

        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def reset_exploration_state(self):
        """Clear any in-progress sticky-exploration burst. Call at the start
        of each training episode so a commitment made near the end of one
        episode doesn't carry over into the next (which starts from a fresh,
        unrelated environment state)."""
        self._sticky_remaining = 0
        self._sticky_action = 0

    def predict(self, state, deterministic=True):
        """Predict an action (compatible interface for evaluation/baseline code)."""
        if deterministic:
            with torch.no_grad():
                state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
                q_values = self.q_network(state_t)
                action = int(torch.argmax(q_values, dim=1).item())
        else:
            action = self.act(state, training=True)
        return action, None

    # ---- learning ----------------------------------------------------

    def remember(self, state, action, reward, next_state, done):
        """Store one transition in the prioritized replay buffer and advance
        the step-based epsilon schedule (see the note in __init__)."""
        self.memory.push(state, action, reward, next_state, done)
        self._steps_since_train += 1

        self._explore_steps += 1
        frac = min(1.0, self._explore_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train_step(self):
        """
        Sample a prioritized mini-batch and update the online network via
        the Double-DQN Bellman target:
            y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

        Actually performs a gradient update only once every `train_every`
        calls to `remember()` (see the note on `self.train_every` in
        __init__); the other calls are cheap no-ops. This is what keeps
        training fast without shrinking the replay buffer or batch size.
        """
        if len(self.memory) < self.warmup_steps:
            return 0.0
        if self._steps_since_train < self.train_every:
            return 0.0
        self._steps_since_train = 0

        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(self.batch_size)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)
        weights_t = torch.as_tensor(is_weights, device=self.device).unsqueeze(1)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            q_values = self.q_network(states_t).gather(1, actions_t)

            with torch.no_grad():
                # Double DQN: online net selects the action, target net evaluates it.
                next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states_t).gather(1, next_actions)
                target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            # Huber/smooth-L1 loss is more robust to the occasional large TD error
            # than plain MSE, and is weighted per-sample by the PER importance weights.
            elementwise_loss = F.smooth_l1_loss(q_values, target_q, reduction="none")
            loss = (weights_t * elementwise_loss).mean()

        td_errors = (target_q - q_values).detach().squeeze(1).abs().cpu().numpy()

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            self.optimizer.step()

        if self.optimizer.param_groups[0]["lr"] > self.min_lr:
            self.scheduler.step()

        self.memory.update_priorities(idxs, td_errors)
        self._soft_update_target()

        self.train_steps_done += 1
        loss_val = float(loss.item())
        self.training_losses.append(loss_val)
        return loss_val

    def _soft_update_target(self):
        """Polyak-average the target network towards the online network."""
        with torch.no_grad():
            for target_param, online_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def update_epsilon(self):
        """Kept for backward compatibility with callers that decay epsilon once
        per episode. Epsilon is now annealed per environment step inside
        remember(), so this is a safe no-op -- it changes nothing and can be
        called (or not) without affecting the schedule."""
        pass

    def update_target_network(self):
        """Hard-copy Q-network weights to the target network. Soft updates
        happen automatically every train_step(); this is kept for callers
        that want an occasional hard sync as well (harmless either way)."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ---- persistence ----------------------------------------------------

    def save(self, path="models/dqn_agent"):
        """Save a full, resumable checkpoint (weights + optimizer + scheduler state)."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "explore_steps": self._explore_steps,
            "train_steps_done": self.train_steps_done,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "seed": self.seed,
        }
        torch.save(checkpoint, path + ".pt")

        # Small human-readable sidecar, handy for the API/frontend without loading torch.
        with open(path + "_config.json", "w") as f:
            json.dump({
                "state_size": int(self.state_size),
                "action_size": int(self.action_size),
                "epsilon": float(self.epsilon),
                "gamma": float(self.gamma),
                "train_steps_done": int(self.train_steps_done),
                "device": str(self.device),
                "architecture": "Dueling Double DQN + PER",
            }, f, indent=2)

    def load(self, path="models/dqn_agent"):
        """Load a checkpoint saved by `save()`, resuming optimizer/scheduler state too."""
        checkpoint_path = path if path.endswith(".pt") else path + ".pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (KeyError, ValueError):
            pass  # Optimizer/scheduler state is optional for pure inference use.

        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self._explore_steps = checkpoint.get("explore_steps", self.epsilon_decay_steps)
        self.train_steps_done = checkpoint.get("train_steps_done", 0)


# ──────────────────────────────────────────────
#  Training & Evaluation Functions
# ──────────────────────────────────────────────

def train_rl_agent(env, n_episodes=200, max_steps=200, model_path="models/dqn_agent",
                    progress_callback=None, log_dir="logs", seed=42,
                    early_stopping_patience=40, early_stopping_min_delta_ms=0.5,
                    early_stopping_min_episodes=30, eval_every=10, eval_episodes_for_selection=3,
                    **kwargs):
    """
    Train the DQN agent on the load balancing environment.

    Args:
        env: LoadBalancerEnv instance
        n_episodes: Maximum number of training episodes
        max_steps: Steps per episode
        model_path: Where to save the trained model (also where the
            best-so-far checkpoint is saved, as f"{model_path}_best")
        progress_callback: Optional function(episode, n_episodes, avg_reward)
        log_dir: TensorBoard log directory (skipped if tensorboard isn't installed)
        seed: Random seed for reproducibility
        early_stopping_patience: Stop if no new best checkpoint has been found
            for this many episodes
        early_stopping_min_delta_ms: Minimum avg-latency improvement (ms) for a
            periodic evaluation to count as a new "best" checkpoint
        early_stopping_min_episodes: Don't early-stop before this many episodes
        eval_every: Run a small clean (deterministic) evaluation every this
            many episodes to decide whether the current policy is a new best
        eval_episodes_for_selection: Episodes used for each of those periodic
            evaluations

    Returns:
        agent: Trained DQNAgent
        episode_rewards: List of total rewards per episode
        episode_latencies: List of average latency per episode
    """
    set_seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # If the caller didn't pin down an explicit exploration schedule, size it to
    # this run's budget: reach epsilon_end after ~60% of the total environment
    # steps, regardless of whether this is a 10-episode smoke test or a
    # 2,000-episode full run. A fixed constant here would leave short runs
    # exploring almost the whole time (as the old per-episode decay did) or
    # make long runs greedy far too early.
    if "epsilon_decay_steps" not in kwargs:
        kwargs["epsilon_decay_steps"] = max(2000, int(0.6 * n_episodes * max_steps))

    agent_kwargs = {k: v for k, v in kwargs.items() if k in _AGENT_KWARGS}
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed, **agent_kwargs)

    # Early stopping tracks periodic clean-evaluation results, but those are
    # inherently noisy and non-monotonic while epsilon is still annealing
    # (more random actions => more variance episode to episode). Starting the
    # "no improvement" patience clock before exploration has mostly finished
    # can trigger a stop while the agent is still exploring, well before it's
    # had a real chance to exploit what it's learned -- so the effective
    # minimum is whichever is larger: the caller's requested floor, or the
    # episode at which epsilon reaches epsilon_end (plus a short buffer for
    # the policy to stabilize once greedy).
    epsilon_anneal_episodes = int(np.ceil(agent.epsilon_decay_steps / max_steps))
    early_stopping_min_episodes = max(early_stopping_min_episodes, epsilon_anneal_episodes + 15)

    writer = None
    if _TENSORBOARD_AVAILABLE:
        run_dir = os.path.join(log_dir, f"dqn_{time.strftime('%Y%m%d_%H%M%S')}")
        try:
            writer = SummaryWriter(run_dir)
        except Exception:
            writer = None  # Logging is best-effort; never let it break training.

    episode_rewards = []
    episode_latencies = []
    # How "best" is decided: see the note at the periodic-evaluation call below.
    # Guarantee at least a handful of checkpoints even on short runs (e.g. an
    # 8-episode smoke test) rather than only ever checking once at the end.
    eval_every = max(1, min(eval_every, max(1, n_episodes // 4)))
    best_eval_latency = np.inf
    best_episode = None
    best_avg_latency = None
    episodes_without_improvement = 0

    print(f"Starting DQN training for up to {n_episodes} episodes...")
    print(f"Device: {agent.device} | Mixed precision: {agent.use_amp}")
    print(f"Network: {state_size} -> Dueling({agent_kwargs.get('hidden_size', 128)}) -> {action_size}")
    print(f"Replay buffer (PER): {agent.memory.capacity}, Batch: {agent.batch_size}")
    print()

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        agent.reset_exploration_state()
        total_reward = 0.0
        ep_latencies = []

        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            total_reward += reward
            ep_latencies.append(-reward)  # reward = -latency
            state = next_state

            if done:
                break

        agent.update_epsilon()

        episode_rewards.append(total_reward)
        avg_latency = float(np.mean(ep_latencies))
        episode_latencies.append(avg_latency)

        if writer is not None:
            writer.add_scalar("train/episode_reward", total_reward, episode)
            writer.add_scalar("train/avg_latency_ms", avg_latency, episode)
            writer.add_scalar("train/epsilon", agent.epsilon, episode)
            writer.add_scalar("train/learning_rate", agent.optimizer.param_groups[0]["lr"], episode)
            if agent.training_losses:
                writer.add_scalar("train/loss", agent.training_losses[-1], episode)

        # Checkpoint selection is based on a small, clean (deterministic, no
        # exploration) evaluation run periodically -- not on raw training-episode
        # latency, and not even on a smoothed window of it. Training-time numbers
        # still have exploration noise baked in (whatever epsilon currently is),
        # and a smoothed window can still be dragged down by one bad neighboring
        # episode while a genuinely-better checkpoint sits right next to it: a
        # single great episode next to a single terrible one can lose out to a
        # neighborhood of merely-good-but-consistent episodes, even though the
        # great episode's underlying policy was actually better. Evaluating
        # directly and deterministically, the same way the final reported
        # metrics are computed, is the only way "best" reliably means best.
        just_evaluated = (episode + 1) % eval_every == 0 or (episode + 1) == n_episodes
        improved = False
        if just_evaluated:
            selection_eval = evaluate_rl_agent(env, agent, n_episodes=eval_episodes_for_selection, max_steps=max_steps)
            eval_latency = selection_eval["avg_latency"]
            improved = eval_latency < best_eval_latency - early_stopping_min_delta_ms
            if improved:
                best_eval_latency = eval_latency
                best_episode = episode + 1
                best_avg_latency = eval_latency
                episodes_without_improvement = 0
                agent.save(model_path + "_best")
            else:
                episodes_without_improvement += eval_every

        print(f"Episode {episode + 1}/{n_episodes} | "
              f"Reward: {total_reward:.1f} | "
              f"Avg Latency: {avg_latency:.1f} ms | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"LR: {agent.optimizer.param_groups[0]['lr']:.2e}")
        if just_evaluated:
            print(f"    -> checkpoint eval ({eval_episodes_for_selection} clean episodes): "
                  f"{eval_latency:.1f} ms avg (best so far: {best_eval_latency:.1f} ms)"
                  + ("  <- new best" if improved else ""))

        if progress_callback:
            progress_callback(episode + 1, n_episodes, total_reward)

        # Don't let early stopping trigger before a first real "best" has even
        # been recorded (possible in principle if the window never fills, or
        # if reward is still declining when min_episodes is reached) -- there
        # would be nothing meaningful to have "stopped improving" from.
        if (best_episode is not None
                and (episode + 1) >= early_stopping_min_episodes
                and episodes_without_improvement >= early_stopping_patience):
            print(f"\nEarly stopping at episode {episode + 1}: "
                  f"no new best checkpoint found in the last {early_stopping_patience} episodes.")
            break

    # Restore the best checkpoint before the final save/return. Training is noisy
    # (especially with a nonzero epsilon_end that keeps some exploration on
    # forever), so the *last* episode's weights are not necessarily the best
    # ones seen -- without this, evaluate_rl_agent() would score a policy that
    # may have regressed after its peak, which is a mismatch people will
    # rightly notice against the per-episode training log.
    best_checkpoint_path = model_path + "_best"
    if os.path.exists(best_checkpoint_path + ".pt"):
        agent.load(best_checkpoint_path)
        print(f"\nRestored best checkpoint: episode {best_episode}, "
              f"clean-eval avg latency {best_avg_latency:.1f} ms")

    # Exposed for callers (e.g. the API layer) that want to report which
    # checkpoint/episode the returned agent actually corresponds to.
    agent.best_episode = best_episode
    agent.best_avg_latency = best_avg_latency
    agent.checkpoint_path = model_path
    agent.best_checkpoint_path = best_checkpoint_path

    agent.save(model_path)
    print(f"Model saved to {model_path}.pt")
    print(f"Final epsilon: {agent.epsilon:.4f} | Total training steps: {agent.train_steps_done}")

    if writer is not None:
        writer.close()

    return agent, episode_rewards, episode_latencies


def evaluate_rl_agent(env, agent, n_episodes=10, max_steps=200):
    """
    Evaluate the trained DQN agent.

    Returns dict with:
        avg_reward, avg_latency, avg_utilization, fairness_index,
        std_latency, std_utilization, per_server_utilization, p99_latency
    """
    total_rewards = []
    latencies = []
    utilizations = []
    fairness_scores = []
    all_per_server = []
    all_latencies_flat = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_latencies = []
        episode_utilizations = []
        episode_per_server = []

        for step in range(max_steps):
            # Use deterministic policy (no exploration)
            action, _ = agent.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            latency = -reward
            episode_latencies.append(latency)
            all_latencies_flat.append(latency)

            server_cpus = [next_state[i * 3] for i in range(env.n_servers)]
            episode_utilizations.append(np.mean(server_cpus))
            episode_per_server.append(server_cpus)

            state = next_state
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        latencies.append(np.mean(episode_latencies))
        utilizations.append(np.mean(episode_utilizations))

        # Jain's Fairness Index on final server utilizations
        final_utils = [state[i * 3] for i in range(env.n_servers)]
        if sum(final_utils) > 0:
            fairness = sum(final_utils) ** 2 / (len(final_utils) * sum(u ** 2 for u in final_utils))
        else:
            fairness = 0
        fairness_scores.append(fairness)

        per_server_avg = np.mean(episode_per_server, axis=0).tolist()
        all_per_server.append(per_server_avg)

    return {
        'avg_reward': float(np.mean(total_rewards)),
        'avg_latency': float(np.mean(latencies)),
        'avg_utilization': float(np.mean(utilizations)),
        'fairness_index': float(np.mean(fairness_scores)),
        'std_latency': float(np.std(latencies)),
        'std_utilization': float(np.std(utilizations)),
        'per_server_utilization': np.mean(all_per_server, axis=0).tolist(),
        'p99_latency': float(np.percentile(all_latencies_flat, 99)) if all_latencies_flat else 0.0,
    }


def load_rl_agent(model_path="models/dqn_agent", state_size=9, action_size=3, device=None, **kwargs):
    """Load a trained DQN agent from disk."""
    agent_kwargs = {k: v for k, v in kwargs.items() if k in _AGENT_KWARGS}
    agent = DQNAgent(state_size=state_size, action_size=action_size, device=device, **agent_kwargs)
    agent.load(model_path)
    print(f"DQN agent loaded from {model_path}.pt (device={agent.device})")
    return agent
