import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stable_baselines3 import PPO

try:
    from .environment import LoadBalancerEnv
    from .baselines import run_baseline_comparison
    from .agent import evaluate_rl_agent, load_rl_agent
except ImportError:
    # For direct execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv
    from baselines import run_baseline_comparison
    from agent import evaluate_rl_agent, load_rl_agent

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def run_full_evaluation(model_path="models/rl_load_balancer", n_episodes=20):
    """Run complete evaluation comparing RL agent with baselines."""
    print("Starting full evaluation...")

    # Create environment
    env = LoadBalancerEnv(n_servers=3, max_steps=200)

    # Evaluate baselines
    print("Evaluating baseline algorithms...")
    baseline_results = run_baseline_comparison(env, n_episodes)

    # Evaluate RL agent
    print("Evaluating RL agent...")
    try:
        model = load_rl_agent(model_path)
        rl_results = evaluate_rl_agent(env, model, n_episodes)
        baseline_results['RL Agent'] = rl_results
    except FileNotFoundError:
        print(f"Warning: RL model not found at {model_path}. Skipping RL evaluation.")
        rl_results = None

    return baseline_results

def plot_latency_comparison(results, save_path="results/latency_comparison.png"):
    """Plot average latency comparison."""
    algorithms = list(results.keys())
    latencies = [results[alg]['avg_latency'] for alg in algorithms]
    errors = [results[alg]['std_latency'] for alg in algorithms]

    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, latencies, yerr=errors, capsize=5, color=colors[:len(algorithms)])
    plt.bar_label(bars, fmt='%.1f ms')

    plt.title('Average Response Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Average Latency (ms)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_utilization_fairness(results, save_path="results/utilization_fairness.png"):
    """Plot server utilization and fairness comparison."""
    algorithms = list(results.keys())
    utilizations = [results[alg]['avg_utilization'] for alg in algorithms]
    fairness_scores = [results[alg].get('fairness_index', 0) for alg in algorithms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Utilization plot
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']
    bars1 = ax1.bar(algorithms, utilizations, color=colors[:len(algorithms)])
    ax1.bar_label(bars1, fmt='%.3f')
    ax1.set_title('Average Server Utilization', fontweight='bold')
    ax1.set_ylabel('Average CPU Utilization')
    ax1.set_xlabel('Algorithm')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Fairness plot
    bars2 = ax2.bar(algorithms, fairness_scores, color=colors[:len(algorithms)])
    ax2.bar_label(bars2, fmt='%.3f')
    ax2.set_title('Server Utilization Fairness (Jain\'s Index)', fontweight='bold')
    ax2.set_ylabel('Fairness Index (0-1)')
    ax2.set_xlabel('Algorithm')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_bursty_traffic_comparison(model_path="models/rl_load_balancer", save_path="results/bursty_traffic.png"):
    """Plot performance under bursty traffic."""
    from .agent import create_burst_traffic_env

    env = create_burst_traffic_env(n_servers=3, burst_intensity=2.0)

    # Simulate bursty traffic over time
    time_steps = 200
    time_range = range(time_steps)

    # Round Robin simulation
    rr_latencies = []
    obs, _ = env.reset()
    for t in time_range:
        action = t % env.n_servers  # Round Robin
        obs, reward, terminated, truncated, _ = env.step(action)
        rr_latencies.append(-reward)  # Convert reward to latency
        if terminated or truncated:
            break

    # RL Agent simulation
    try:
        model = load_rl_agent(model_path)
        rl_latencies = []
        obs, _ = env.reset()
        for t in time_range:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rl_latencies.append(-reward)
            if terminated or truncated:
                break
    except FileNotFoundError:
        print("RL model not found, using baseline for RL")
        rl_latencies = rr_latencies  # Placeholder

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_range[:len(rr_latencies)], rr_latencies, label='Round Robin', color='#e74c3c', linewidth=2)
    plt.plot(time_range[:len(rl_latencies)], rl_latencies, label='RL Agent', color='#27ae60', linewidth=2)

    # Highlight burst period
    plt.axvspan(50, 100, alpha=0.1, color='red', label='Traffic Burst Period')

    plt.title('Latency Under Bursty Traffic Patterns', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Response Time (ms)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def generate_summary_report(results, save_path="results/summary_report.txt"):
    """Generate a text summary of results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("ADAPTIVE LOAD BALANCING EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("ALGORITHMS COMPARED:\n")
        for alg in results.keys():
            f.write(f"- {alg}\n")
        f.write("\n")

        f.write("PERFORMANCE METRICS:\n\n")

        # Create table
        header = "| Algorithm | Avg Latency (ms) | Std Dev | Avg Utilization | Fairness Index |\n"
        separator = "|-----------|------------------|---------|-----------------|---------------|\n"
        f.write(header)
        f.write(separator)

        for alg, metrics in results.items():
            latency = metrics['avg_latency']
            std_lat = metrics['std_latency']
            util = metrics['avg_utilization']
            fairness = metrics.get('fairness_index', 'N/A')
            f.write(f"| {alg:<10} | {latency:>15.2f} | {std_lat:>7.2f} | {util:>15.3f} | {fairness:>13} |\n")

        f.write("\n")

        # Find best performers
        best_latency = min(results.items(), key=lambda x: x[1]['avg_latency'])
        best_fairness = max(results.items(), key=lambda x: x[1].get('fairness_index', 0))

        f.write("KEY FINDINGS:\n")
        f.write(f"- Best latency: {best_latency[0]} ({best_latency[1]['avg_latency']:.2f} ms)\n")
        f.write(f"- Best fairness: {best_fairness[0]} (Jain's Index: {best_fairness[1].get('fairness_index', 'N/A')})\n")

        if 'RL Agent' in results:
            rl_latency = results['RL Agent']['avg_latency']
            rr_latency = results.get('Round Robin', {}).get('avg_latency', rl_latency)
            if rr_latency > rl_latency:
                improvement = ((rr_latency - rl_latency) / rr_latency) * 100
                f.write(f"- RL improvement over Round Robin: {improvement:.1f}%\n")

    print(f"Summary report saved to {save_path}")

def run_all_evaluations():
    """Run all evaluations and generate plots."""
    results = run_full_evaluation()

    # Generate plots
    plot_latency_comparison(results)
    plot_utilization_fairness(results)
    plot_bursty_traffic_comparison()

    # Generate report
    generate_summary_report(results)

    print("All evaluations completed!")
    return results