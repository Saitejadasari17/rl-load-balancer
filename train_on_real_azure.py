#!/usr/bin/env python3
"""
Train RL Agent on Real Microsoft Azure Functions Dataset
Trains on actual Azure function invocation traces with memory and duration data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment import LoadBalancerEnv
from agent import train_rl_agent, evaluate_rl_agent
from baselines import (
    RoundRobinAgent, LeastConnectionsAgent, RandomAgent, 
    WeightedRoundRobinAgent, evaluate_agent
)


def load_azure_invocations(nrows=None):
    """Load invocations per function data from extracted Azure dataset"""
    extracted_dir = Path('data/extracted')
    
    if not extracted_dir.exists():
        print("❌ data/extracted/ not found. Run setup_tar_xz_dataset.py first!")
        return None
    
    csv_files = sorted(extracted_dir.glob('invocations_per_function_md*.csv'))
    
    if not csv_files:
        print("❌ No invocations CSV files found")
        return None
    
    print(f"\n📊 Loading {len(csv_files)} invocation CSV files...")
    
    dfs = []
    total_rows = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(csv_file, nrows=nrows)
            dfs.append(df)
            total_rows += len(df)
            print(f"  ✓ {csv_file.name}: {len(df):,} rows")
            
            if nrows and total_rows >= nrows:
                break
        except Exception as e:
            print(f"  ⚠ Skipped {csv_file.name}: {e}")
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total loaded: {len(combined_df):,} rows")
    
    return combined_df


def extract_load_pattern(df):
    """Extract normalized load pattern from invocation data"""
    
    # Try different column names
    possible_cols = [
        'count', 'invocations', 'num_invocations', 
        'function_invocations', 'total_invocations'
    ]
    
    count_col = None
    for col in possible_cols:
        if col in df.columns:
            count_col = col
            break
    
    if count_col is None:
        print(f"⚠ Available columns: {list(df.columns)[:5]}")
        # Use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            count_col = numeric_cols[0]
            print(f"Using column: {count_col}")
        else:
            return None
    
    # Get invocation counts
    counts = df[count_col].dropna().values.astype(float)
    
    if len(counts) == 0:
        return None
    
    # Normalize to load range [0.05, 0.3]
    min_load, max_load = 0.05, 0.3
    counts_normalized = min_load + (counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_load - min_load)
    
    print(f"\nLoad Pattern Statistics:")
    print(f"  Raw invocations: min={counts.min():.0f}, max={counts.max():.0f}, mean={counts.mean():.0f}")
    print(f"  Normalized load: min={counts_normalized.min():.3f}, max={counts_normalized.max():.3f}, mean={counts_normalized.mean():.3f}")
    
    return counts_normalized


def main():
    print("\n" + "="*60)
    print("ADAPTIVE LOAD BALANCING - REAL AZURE DATA TRAINING")
    print("="*60)
    
    # Load Azure dataset
    df = load_azure_invocations(nrows=10000)
    
    if df is None:
        print("\n❌ Failed to load dataset")
        return
    
    print(f"\nDataset Info:")
    print(f"  Columns: {list(df.columns)[:8]}")
    print(f"  Shape: {df.shape}")
    
    # Extract load pattern
    trace_data = extract_load_pattern(df)
    
    if trace_data is None:
        print("\n⚠ Using synthetic data as fallback")
        trace_data = None
    
    # Number of servers
    num_servers = 3
    num_requests = len(trace_data) if trace_data is not None else 1000
    
    print(f"\n🎯 Simulation Setup:")
    print(f"  Servers: {num_servers}")
    print(f"  Requests: {num_requests}")
    print(f"  Data source: {'Azure real data' if trace_data is not None else 'Synthetic'}")
    
    # Create environment with real data
    env = LoadBalancerEnv(n_servers=num_servers, trace_data=trace_data)
    
    # Evaluate baselines
    print(f"\n{'='*60}")
    print("BASELINE ALGORITHMS")
    print(f"{'='*60}")
    
    baselines = {
        'Round Robin': RoundRobinAgent(num_servers),
        'Least Connections': LeastConnectionsAgent(num_servers),
        'Random': RandomAgent(num_servers),
        'Weighted RR': WeightedRoundRobinAgent(num_servers),
    }
    
    baseline_results = {}
    
    for name, agent in baselines.items():
        result = evaluate_agent(env, agent, n_episodes=5, max_steps=500)
        avg_latency = result['avg_latency']
        avg_util = result['avg_utilization'] * 100  # Convert to percentage
        baseline_results[name] = avg_latency
        print(f"\n{name}:")
        print(f"  Avg Latency:     {avg_latency:.2f} ms")
        print(f"  Avg Utilization: {avg_util:.1f}%")
        print(f"  Std Latency:     {result['std_latency']:.2f} ms")
    
    best_baseline = min(baseline_results.values())
    
    # Train RL Agent
    print(f"\n{'='*60}")
    print("TRAINING RL AGENT (PPO)")
    print(f"{'='*60}")
    
    timesteps = 50000
    print(f"\nTraining for {timesteps:,} timesteps on REAL Azure data...")
    
    model, training_rewards = train_rl_agent(
        env=env,
        total_timesteps=timesteps,
        model_path='models/rl_agent_azure'
    )
    
    # Evaluate RL Agent
    print(f"\n{'='*60}")
    print("EVALUATING RL AGENT")
    print(f"{'='*60}")
    
    rl_results = evaluate_rl_agent(env, model, n_episodes=10, max_steps=500)
    rl_latency = rl_results['avg_latency']
    rl_util = rl_results['avg_utilization'] * 100
    
    print(f"\nRL Agent Results:")
    print(f"  Avg Latency:     {rl_latency:.2f} ms")
    print(f"  Avg Utilization: {rl_util:.1f}%")
    print(f"  Fairness Index:  {rl_results['fairness_index']:.3f}")
    
    # Calculate improvement
    improvement = ((best_baseline - rl_latency) / best_baseline) * 100
    
    print(f"\n{'='*60}")
    print("RESEARCH CLAIM - CAPSTONE VALIDATION")
    print(f"{'='*60}")
    print(f"\n📈 IMPROVEMENT METRICS:")
    print(f"  Best Baseline:   {best_baseline:.2f} ms")
    print(f"  RL Agent:        {rl_latency:.2f} ms")
    print(f"  Improvement:     +{improvement:.1f}% ✓")
    
    if improvement > 10:
        print(f"\n✅ SIGNIFICANT improvement achieved!")
        print(f"   Trained on REAL Microsoft Azure Functions dataset (2019)")
        print(f"   {len(trace_data):,} real invocation traces used")
        print(f"   {num_servers} heterogeneous servers simulated")
    elif improvement > 0:
        print(f"\n✅ Positive improvement achieved!")
    else:
        print(f"\n⚠ Using real data without improvement - may need tuning")
    
    print(f"\n📁 Models saved to: models/rl_agent_azure.zip")
    print(f"📊 Results ready for capstone presentation!")
    
    return {
        'rl_latency': rl_latency,
        'best_baseline': best_baseline,
        'improvement': improvement,
        'trace_count': len(trace_data) if trace_data is not None else 0
    }


if __name__ == '__main__':
    results = main()
