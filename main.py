#!/usr/bin/env python3
"""
Adaptive Load Balancing Using Reinforcement Learning
Main script to run training, evaluation, and analysis.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment import LoadBalancerEnv
from agent import train_rl_agent, evaluate_rl_agent, plot_learning_curve
from baselines import run_baseline_comparison
from evaluation import run_all_evaluations

def main():
    parser = argparse.ArgumentParser(description='Adaptive Load Balancing with RL')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare', 'all'],
                       default='all', help='Mode to run')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Number of training timesteps')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--servers', type=int, default=3,
                       help='Number of servers in simulation')

    args = parser.parse_args()

    print("Adaptive Load Balancing Using Reinforcement Learning")
    print("=" * 55)

    # Create environment
    env = LoadBalancerEnv(n_servers=args.servers)

    if args.mode in ['train', 'all']:
        print(f"\n1. Training RL Agent ({args.timesteps} timesteps)...")
        model, rewards = train_rl_agent(env, total_timesteps=args.timesteps)

        print(f"\n2. Plotting learning curve...")
        plot_learning_curve(rewards)

    if args.mode in ['evaluate', 'all']:
        print(f"\n3. Evaluating trained agent ({args.episodes} episodes)...")
        from stable_baselines3 import PPO
        try:
            model = PPO.load("models/rl_load_balancer")
            results = evaluate_rl_agent(env, model, args.episodes)
            print("RL Agent Results:")
            print(".2f")
            print(".3f")
            print(".3f")
        except FileNotFoundError:
            print("No trained model found. Run with --mode train first.")

    if args.mode in ['compare', 'all']:
        print(f"\n4. Comparing all algorithms ({args.episodes} episodes)...")
        results = run_all_evaluations()
        print("\nComparison Results:")
        for alg, metrics in results.items():
            print(f"{alg}: Latency = {metrics['avg_latency']:.2f} ms")

    if args.mode == 'all':
        print("\n5. All tasks completed!")
        print("Check the 'results/' directory for plots and reports.")

if __name__ == "__main__":
    main()