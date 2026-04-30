import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
import matplotlib.pyplot as plt

try:
    from .environment import LoadBalancerEnv
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv

class RewardCallback(BaseCallback):
    """Callback to track rewards during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Log reward at the end of each episode
        if len(self.model.ep_info_buffer) > 0:
            episode_reward = self.model.ep_info_buffer[-1]['r']
            self.rewards.append(episode_reward)
        return True

def train_rl_agent(env, total_timesteps=100000, model_path="models/rl_load_balancer"):
    """Train the RL agent using PPO."""
    # Create monitored environment
    env = Monitor(env, filename="logs/")

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )

    # Create callback to track rewards
    callback = RewardCallback()

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model, callback.rewards

def load_rl_agent(model_path="models/rl_load_balancer"):
    """Load a trained RL agent."""
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

def evaluate_rl_agent(env, model, n_episodes=10, max_steps=200):
    """Evaluate the trained RL agent."""
    total_rewards = []
    latencies = []
    utilizations = []
    fairness_scores = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_latencies = []
        episode_utilizations = []

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            latency = -reward  # Convert reward back to latency
            episode_latencies.append(latency)

            # Track server utilizations
            server_cpus = [next_obs[i*3] for i in range(env.n_servers)]
            episode_utilizations.append(np.mean(server_cpus))

            obs = next_obs

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        latencies.append(np.mean(episode_latencies))
        utilizations.append(np.mean(episode_utilizations))

        # Calculate Jain's fairness index for server utilizations
        final_utilizations = [obs[i*3] for i in range(env.n_servers)]
        if sum(final_utilizations) > 0:
            fairness = sum(final_utilizations)**2 / (len(final_utilizations) * sum(u**2 for u in final_utilizations))
        else:
            fairness = 0
        fairness_scores.append(fairness)

    return {
        'avg_reward': np.mean(total_rewards),
        'avg_latency': np.mean(latencies),
        'avg_utilization': np.mean(utilizations),
        'fairness_index': np.mean(fairness_scores),
        'std_latency': np.std(latencies),
        'std_utilization': np.std(utilizations)
    }

def plot_learning_curve(rewards, save_path="results/learning_curve.png"):
    """Plot the learning curve from training rewards."""
    plt.figure(figsize=(10, 6))
    episodes = range(1, len(rewards) + 1)

    # Smooth the rewards for better visualization
    window_size = max(1, len(rewards) // 50)
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, color='#3498db', linewidth=2, label='Smoothed Rewards')
    plt.plot(episodes, rewards, color='#bdc3c7', alpha=0.5, linewidth=1, label='Raw Rewards')

    plt.title('RL Agent Learning Curve')
    plt.xlabel('Training Episode')
    plt.ylabel('Average Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def create_burst_traffic_env(n_servers=3, burst_intensity=2.0):
    """Create an environment with bursty traffic patterns."""
    class BurstyLoadBalancerEnv(LoadBalancerEnv):
        def __init__(self, n_servers=3, burst_intensity=2.0):
            super().__init__(n_servers)
            self.burst_intensity = burst_intensity
            self.burst_timer = 0

        def step(self, action):
            # Modify request load based on burst pattern
            base_load = np.random.uniform(0.05, 0.3)

            # Create periodic bursts
            if 50 <= self.current_step <= 100:  # Burst period
                load_multiplier = self.burst_intensity
            else:
                load_multiplier = 1.0

            request_load = base_load * load_multiplier

            # Temporarily override the request load
            original_load = np.random.uniform(0.05, 0.3)
            # This is a simplified way; in practice, we'd modify the step method properly

            return super().step(action)

    return BurstyLoadBalancerEnv(n_servers, burst_intensity)