import numpy as np
try:
    from .environment import LoadBalancerEnv
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv

class BaselineAgent:
    """Base class for baseline load balancing algorithms."""
    def __init__(self, n_servers):
        self.n_servers = n_servers
        self.reset()

    def reset(self):
        self.last_action = -1

    def act(self, state):
        """Choose action based on current state."""
        raise NotImplementedError

class RoundRobinAgent(BaselineAgent):
    """Round Robin: Cycle through servers sequentially."""
    def act(self, state):
        self.last_action = (self.last_action + 1) % self.n_servers
        return self.last_action

class LeastConnectionsAgent(BaselineAgent):
    """Least Connections: Route to server with fewest queued requests."""
    def act(self, state):
        # State format: [cpu1, queue1, lat1, cpu2, queue2, lat2, ...]
        queues = [state[i*3 + 1] for i in range(self.n_servers)]  # Extract queue values
        return np.argmin(queues)  # Choose server with smallest queue

class RandomAgent(BaselineAgent):
    """Random: Choose server randomly."""
    def act(self, state):
        return np.random.randint(0, self.n_servers)

class WeightedRoundRobinAgent(BaselineAgent):
    """Weighted Round Robin: Consider server capacities."""
    def __init__(self, n_servers, weights=None):
        super().__init__(n_servers)
        self.weights = weights or [1.0] * n_servers  # Default equal weights
        self.cumulative_weights = np.cumsum(self.weights)
        self.total_weight = self.cumulative_weights[-1]
        self.current_weight = 0

    def act(self, state):
        # Cycle current_weight through (0, total_weight] rather than
        # [0, total_weight) -- with a 0-based wrap the counter never actually
        # reached `total_weight`, so the threshold check below never matched
        # the last server's cumulative-weight bucket and it received zero
        # traffic (e.g. with 3 equally-weighted servers, server index 2 was
        # never selected).
        self.current_weight = (self.current_weight % self.total_weight) + 1
        for i, cw in enumerate(self.cumulative_weights):
            if self.current_weight <= cw:
                return i
        return self.n_servers - 1  # Fallback (floating point edge case)

def evaluate_agent(env, agent, n_episodes=10, max_steps=200):
    """Evaluate an agent over multiple episodes."""
    total_rewards = []
    latencies = []
    utilizations = []
    fairness_scores = []
    all_latencies_flat = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        agent.reset()
        episode_reward = 0
        episode_latencies = []
        episode_utilizations = []

        for step in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            # Convert reward back to latency (since reward = -latency)
            latency = -reward
            episode_latencies.append(latency)
            all_latencies_flat.append(latency)

            # Track server utilizations
            server_cpus = [next_obs[i*3] for i in range(env.n_servers)]
            episode_utilizations.append(np.mean(server_cpus))

            obs = next_obs

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        latencies.append(np.mean(episode_latencies))
        utilizations.append(np.mean(episode_utilizations))

        # Jain's Fairness Index on final server utilizations, computed the same
        # way as evaluate_rl_agent() so baselines and the RL agent are directly
        # comparable on fairness rather than baselines always reading as "N/A".
        final_utils = [obs[i * 3] for i in range(env.n_servers)]
        if sum(final_utils) > 0:
            fairness = sum(final_utils) ** 2 / (len(final_utils) * sum(u ** 2 for u in final_utils))
        else:
            fairness = 0
        fairness_scores.append(fairness)

    return {
        'avg_reward': np.mean(total_rewards),
        'avg_latency': np.mean(latencies),
        'avg_utilization': np.mean(utilizations),
        'std_latency': np.std(latencies),
        'std_utilization': np.std(utilizations),
        'fairness_index': float(np.mean(fairness_scores)),
        'p99_latency': float(np.percentile(all_latencies_flat, 99)) if all_latencies_flat else 0.0,
    }

def run_baseline_comparison(env, n_episodes=10):
    """Compare all baseline algorithms."""
    agents = {
        'Round Robin': RoundRobinAgent(env.n_servers),
        'Least Connections': LeastConnectionsAgent(env.n_servers),
        'Random': RandomAgent(env.n_servers),
        'Weighted RR': WeightedRoundRobinAgent(env.n_servers)
    }

    results = {}
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_agent(env, agent, n_episodes)
        print(f"  Avg Latency: {results[name]['avg_latency']:.2f} ms")
        print(f"  Avg Utilization: {results[name]['avg_utilization']:.3f}")
        print(f"  Fairness Index: {results[name]['fairness_index']:.3f}")

    return results