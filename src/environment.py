"""
Custom Gymnasium environment simulating a distributed server cluster for
adaptive load balancing.

Performance note: server state (CPU, queue length, latency) is kept as
plain NumPy arrays rather than a list of per-server dicts. A training run
calls `step()` hundreds of thousands to millions of times, and Python-level
dict/attribute access in that inner loop is the dominant cost at that scale.
Vectorizing the "decay other servers" update in particular turns an
O(n_servers) Python loop into a couple of NumPy array ops.
"""

import gymnasium as gym
import numpy as np


class LoadBalancerEnv(gym.Env):
    """
    The agent must decide which server to route each incoming request to,
    based on the current state of every server (CPU utilization, queue
    length, recent latency).

    Supports both synthetic random traffic and replayed real-world traces
    (e.g. the Azure Functions dataset via `src/azure_dataset.py`).

    Observation: for each server, [cpu (0-1), queue (normalized 0-1),
    latency (normalized)] concatenated -> shape (n_servers * 3,).
    Action: Discrete(n_servers) -- which server receives the request.
    Reward: -latency of the just-served request (lower latency = higher
    reward), matching the project's "minimize response time" objective.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_servers=3, max_queue=20, max_steps=200, trace_data=None):
        super().__init__()

        self.n_servers = n_servers
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.trace_data = np.asarray(trace_data, dtype=np.float32) if trace_data is not None else None
        self.trace_index = 0

        # Action space: choose which server (0 to n_servers-1)
        self.action_space = gym.spaces.Discrete(n_servers)

        # Observation space: [cpu, queue, latency] for each server, all normalized to 0-1
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_servers * 3,),
            dtype=np.float32
        )

        # Vectorized server state (replaces the old list-of-dicts representation)
        self.cpu = np.zeros(n_servers, dtype=np.float32)
        self.queue = np.zeros(n_servers, dtype=np.float32)
        self.latency = np.zeros(n_servers, dtype=np.float32)
        self.current_step = 0
        self.total_requests = 0

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state. Seeding also resets the
        environment's own RNG (`self.np_random`), which is what request
        loads and decay noise are drawn from -- this makes training runs
        reproducible when a seed is supplied."""
        super().reset(seed=seed)

        self.cpu.fill(0.0)
        self.queue.fill(0.0)
        self.latency.fill(0.0)

        self.current_step = 0
        self.total_requests = 0

        if self.trace_data is not None and len(self.trace_data) > 0:
            # Start each episode at a random offset into the trace, rather than
            # always index 0. With a fixed start, every episode -- and every
            # evaluation call -- replays exactly the same leading `max_steps`
            # rows of the trace; on a large real-world dataset (thousands of
            # points) that means the vast majority of it is never seen during
            # training or evaluation, no matter how many episodes are run.
            self.trace_index = int(self.np_random.integers(0, len(self.trace_data)))
        else:
            self.trace_index = 0

        return self._get_state(), {}

    def _get_state(self):
        """Get current state as a single flat numpy array."""
        state = np.empty(self.n_servers * 3, dtype=np.float32)
        state[0::3] = self.cpu
        state[1::3] = np.minimum(self.queue / self.max_queue, 1.0)
        state[2::3] = self.latency
        return state

    def step(self, action):
        """Execute one step in the environment."""
        if not (0 <= action < self.n_servers):
            raise ValueError(f"Invalid action {action}. Must be 0-{self.n_servers - 1}")

        # Get request load from traces or generate randomly
        if self.trace_data is not None:
            request_load = float(self.trace_data[self.trace_index % len(self.trace_data)])
            self.trace_index += 1
        else:
            request_load = float(self.np_random.uniform(0.05, 0.3))

        # Update chosen server
        self.queue[action] += 1
        # max_queue represents each server's actual queue capacity, not just a
        # normalization constant: without this cap, a bad policy that keeps
        # flooding one server can grow its queue unboundedly, and because the
        # *observed* state already saturates at 1.0 once queue >= max_queue
        # (see _get_state), the agent becomes unable to distinguish "queue at
        # capacity" from "queue enormously past capacity" while the reward
        # (which is driven by the uncapped raw queue) keeps getting more and
        # more negative. That combination -- saturated observations paired
        # with unbounded reward magnitude -- is what destabilizes training.
        self.queue[action] = min(self.queue[action], self.max_queue)
        self.cpu[action] = min(self.cpu[action] + request_load, 1.0)

        # Calculate response time based on server load
        base_latency = 50.0  # Base response time in ms
        cpu_penalty = self.cpu[action] * 100.0
        queue_penalty = self.queue[action] * 10.0
        latency = base_latency + cpu_penalty + queue_penalty

        # Update server latency (moving average)
        self.latency[action] = 0.8 * self.latency[action] + 0.2 * (latency / 200.0)

        # Natural decay for the other servers (simulating background processing),
        # vectorized across all non-chosen servers instead of a per-server Python loop.
        if self.n_servers > 1:
            others = np.ones(self.n_servers, dtype=bool)
            others[action] = False
            self.cpu[others] = np.maximum(self.cpu[others] - 0.02, 0.0)
            decay = self.np_random.integers(0, 2, size=int(others.sum()))
            self.queue[others] = np.maximum(self.queue[others] - decay, 0.0)

        # Reward: negative latency (lower latency = higher reward)
        reward = -latency

        self.current_step += 1
        self.total_requests += 1

        # Episode ends after max_steps
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_state(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Render the current state."""
        print(f"Step: {self.current_step}")
        for i in range(self.n_servers):
            print(f"Server {i}: CPU={self.cpu[i]:.2f}, Queue={int(self.queue[i])}, Latency={self.latency[i]:.2f}")
        print()

    def close(self):
        """Clean up resources."""
        pass
