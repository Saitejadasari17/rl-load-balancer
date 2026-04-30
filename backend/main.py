"""
FastAPI Backend for Adaptive Load Balancing RL System
Provides REST API endpoints for training and evaluation
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from src.environment import LoadBalancerEnv
from src.agent import train_rl_agent, evaluate_rl_agent
from src.baselines import (
    RoundRobinAgent, LeastConnectionsAgent, RandomAgent,
    WeightedRoundRobinAgent, evaluate_agent
)

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Load Balancing - RL System",
    description="Train and evaluate RL agents for load balancing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = {
    "status": "idle",
    "progress": 0,
    "current_episode": 0,
    "total_episodes": 0,
    "message": ""
}

# Request/Response models
class TrainingConfig(BaseModel):
    timesteps: int = 50000
    episodes: int = 10
    servers: int = 3
    learning_rate: float = 3e-4

class EvaluationResult(BaseModel):
    baseline_name: str
    avg_latency: float
    avg_utilization: float
    std_latency: float

class TrainingResult(BaseModel):
    status: str
    baselines: dict
    rl_agent: dict
    improvement: float
    best_baseline_ms: float
    rl_agent_ms: float

# Utility functions
def update_training_state(status: str, progress: int, message: str, episode: int = 0, total: int = 0):
    """Update global training state"""
    training_state["status"] = status
    training_state["progress"] = progress
    training_state["message"] = message
    training_state["current_episode"] = episode
    training_state["total_episodes"] = total

def extract_load_from_csv(df):
    """Extract load pattern from CSV data"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    
    col = numeric_cols[0]
    counts = df[col].dropna().values.astype(float)
    
    if len(counts) == 0:
        return None
    
    # Normalize to [0.05, 0.3]
    min_load, max_load = 0.05, 0.3
    normalized = min_load + (counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_load - min_load)
    
    return normalized

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Adaptive Load Balancing RL System",
        "version": "1.0.0"
    }

@app.get("/training-status")
async def get_training_status():
    """Get current training status"""
    return training_state

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and validate dataset"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and validate CSV
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        # Extract load pattern
        load_pattern = extract_load_from_csv(df)
        
        if load_pattern is None:
            raise HTTPException(status_code=400, detail="No numeric columns found in CSV")
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns)[:5],
            "load_pattern": {
                "min": float(load_pattern.min()),
                "max": float(load_pattern.max()),
                "mean": float(load_pattern.mean())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start RL agent training"""
    try:
        update_training_state("starting", 0, "Initializing environment...", 0, 1)
        
        # Create environment
        env = LoadBalancerEnv(n_servers=config.servers)
        
        # Evaluate baselines
        update_training_state("baselines", 10, "Evaluating baseline algorithms...", 1, 4)
        
        baselines = {
            'Round Robin': RoundRobinAgent(config.servers),
            'Least Connections': LeastConnectionsAgent(config.servers),
            'Random': RandomAgent(config.servers),
            'Weighted RR': WeightedRoundRobinAgent(config.servers),
        }
        
        baseline_results = {}
        for name, agent in baselines.items():
            result = evaluate_agent(env, agent, n_episodes=3, max_steps=500)
            baseline_results[name] = {
                'avg_latency': float(result['avg_latency']),
                'avg_utilization': float(result['avg_utilization']),
                'std_latency': float(result['std_latency'])
            }
        
        best_baseline = min(r['avg_latency'] for r in baseline_results.values())
        
        # Train RL agent
        update_training_state("training", 30, f"Training RL agent for {config.timesteps:,} timesteps...", 5, 6)
        
        model, training_rewards = train_rl_agent(
            env=env,
            total_timesteps=config.timesteps,
            model_path='models/rl_agent_demo'
        )
        
        # Evaluate RL agent
        update_training_state("evaluating", 80, "Evaluating trained agent...", 7, 8)
        
        rl_results = evaluate_rl_agent(env, model, n_episodes=config.episodes, max_steps=500)
        rl_latency = rl_results['avg_latency']
        
        # Calculate improvement
        improvement = ((best_baseline - rl_latency) / best_baseline) * 100
        
        # Save results
        results = {
            "status": "completed",
            "baselines": baseline_results,
            "rl_agent": {
                "avg_latency": float(rl_latency),
                "avg_utilization": float(rl_results['avg_utilization']),
                "fairness_index": float(rl_results['fairness_index'])
            },
            "best_baseline_ms": float(best_baseline),
            "rl_agent_ms": float(rl_latency),
            "improvement": float(improvement)
        }
        
        update_training_state("completed", 100, "Training complete!", 8, 8)
        
        # Save results to file
        os.makedirs('results', exist_ok=True)
        with open('results/api_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    except Exception as e:
        update_training_state("failed", 0, f"Error: {str(e)}", 0, 0)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    """Get last training results"""
    try:
        with open('results/api_training_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No training results found")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and capabilities"""
    return {
        "max_servers": 10,
        "max_episodes": 100,
        "max_timesteps": 500000,
        "supported_algorithms": [
            "Round Robin",
            "Least Connections",
            "Random",
            "Weighted Round Robin",
            "PPO (RL)"
        ],
        "rl_config": {
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
