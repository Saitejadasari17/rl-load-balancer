#!/usr/bin/env python3
"""
Adaptive Load Balancing Using Reinforcement Learning
FastAPI Backend + RL Training System
"""

import argparse
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment import LoadBalancerEnv
from agent import train_rl_agent, evaluate_rl_agent, plot_learning_curve
from baselines import run_baseline_comparison
from evaluation import run_all_evaluations

# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="RL Load Balancer API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API ROUTES
# =========================

@app.get("/")
def home():
    return {
        "message": "Adaptive RL Load Balancer Backend Running",
        "status": "success"
    }


@app.get("/train")
def train_model():
    try:
        env = LoadBalancerEnv(n_servers=3)

        model, rewards = train_rl_agent(
            env,
            total_timesteps=1000
        )

        return {
            "status": "Training completed",
            "episodes": len(rewards),
            "final_reward": rewards[-1] if rewards else 0
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/evaluate")
def evaluate_model():
    try:
        from stable_baselines3 import PPO

        env = LoadBalancerEnv(n_servers=3)

        try:
            model = PPO.load("models/rl_load_balancer")
        except:
            return {
                "status": "error",
                "message": "Trained model not found"
            }

        results = evaluate_rl_agent(env, model, 5)

        return {
            "status": "success",
            "results": str(results)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/compare")
def compare_algorithms():
    try:
        results = run_all_evaluations()

        return {
            "status": "success",
            "comparison_results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# =========================
# CLI MODE (LOCAL EXECUTION)
# =========================

def cli_main():
    parser = argparse.ArgumentParser(
        description='Adaptive Load Balancing with RL'
    )

    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'compare', 'all'],
        default='all',
        help='Mode to run'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Number of training timesteps'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )

    parser.add_argument(
        '--servers',
        type=int,
        default=3,
        help='Number of servers in simulation'
    )

    args = parser.parse_args()

    print("Adaptive Load Balancing Using Reinforcement Learning")
    print("=" * 55)

    # Create environment
    env = LoadBalancerEnv(n_servers=args.servers)

    # TRAIN
    if args.mode in ['train', 'all']:
        print(f"\n1. Training RL Agent ({args.timesteps} timesteps)...")

        model, rewards = train_rl_agent(
            env,
            total_timesteps=args.timesteps
        )

        print("\n2. Plotting learning curve...")
        plot_learning_curve(rewards)

    # EVALUATE
    if args.mode in ['evaluate', 'all']:
        print(f"\n3. Evaluating trained agent ({args.episodes} episodes)...")

        from stable_baselines3 import PPO

        try:
            model = PPO.load("models/rl_load_balancer")

            results = evaluate_rl_agent(
                env,
                model,
                args.episodes
            )

            print("RL Agent Results:")
            print(results)

        except FileNotFoundError:
            print("No trained model found. Run with --mode train first.")

    # COMPARE
    if args.mode in ['compare', 'all']:
        print(f"\n4. Comparing all algorithms ({args.episodes} episodes)...")

        results = run_all_evaluations()

        print("\nComparison Results:")

        for alg, metrics in results.items():
            print(
                f"{alg}: Latency = "
                f"{metrics['avg_latency']:.2f} ms"
            )

    # COMPLETE
    if args.mode == 'all':
        print("\n5. All tasks completed!")
        print("Check the 'results/' directory for plots and reports.")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    cli_main()