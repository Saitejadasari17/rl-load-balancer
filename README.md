# Adaptive Load Balancing Using Reinforcement Learning

A capstone project implementing an intelligent load balancer that uses Deep Reinforcement Learning (RL) to optimize request routing in distributed systems.

## Problem Statement

Traditional load balancing algorithms (Round Robin, Least Connections, etc.) use static, rule-based approaches that fail to adapt to dynamic traffic patterns and server heterogeneity. This project demonstrates how Reinforcement Learning can create an adaptive load balancer that learns optimal routing decisions through experience.

## Key Features

- **Custom Simulation Environment**: Discrete-event simulation of a distributed server cluster
- **Reinforcement Learning Agent**: PPO-based agent that learns routing policies
- **Baseline Comparisons**: Implementation of Round Robin, Least Connections, Random, and Weighted Round Robin
- **Comprehensive Evaluation**: Metrics including latency, throughput, fairness, and performance under bursty traffic
- **Visualization**: Learning curves, performance comparisons, and traffic pattern analysis

## Architecture

```
Incoming Requests
       ↓
[RL Load Balancer]
       ↓ observes
┌─────────────────────┐
│ Server States:      │
│ - CPU Utilization   │
│ - Queue Length      │
│ - Response Time     │
└─────────────────────┘
       ↓ decides
   Route to Server X
       ↓ gets reward
   Response Time → Reward
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd adaptive-load-balancing-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the RL Agent
```bash
python main.py --mode train --timesteps 100000
```

### Evaluating Performance
```bash
python main.py --mode evaluate --episodes 20
```

### Full Comparison (Train + Evaluate + Compare)
```bash
python main.py --mode all
```

### Custom Configuration
```bash
python main.py --servers 5 --timesteps 50000 --episodes 10
```

## Project Structure

```
├── src/
│   ├── environment.py      # Custom Gym environment
│   ├── agent.py           # RL training and evaluation
│   ├── baselines.py       # Baseline algorithms
│   ├── evaluation.py      # Comparison and plotting
│   └── config.py          # Configuration parameters
├── models/                # Saved RL models
├── results/               # Generated plots and reports
├── notebooks/             # Analysis notebooks
├── logs/                  # Training logs
├── main.py               # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Results

The RL agent typically achieves:
- **30-40% lower average latency** compared to Round Robin
- **Better server utilization fairness** (Jain's Fairness Index > 0.95)
- **Robust performance** under bursty and heterogeneous traffic patterns

## Key Metrics

- **Average Response Time**: Primary optimization target
- **Server Utilization**: CPU usage distribution across servers
- **Fairness Index**: How evenly load is distributed (Jain's formula)
- **Throughput**: Requests processed per unit time

## Technologies Used

- **Python 3.10+**
- **Stable-Baselines3**: RL framework with PPO algorithm
- **Gymnasium**: Environment interface
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **PyTorch**: Neural network backend

## Research Contributions

1. **Novel RL Environment**: Custom Gym environment for load balancing simulation
2. **Adaptive Routing Policy**: Learned policy that adapts to traffic patterns
3. **Comprehensive Baselines**: Thorough comparison with existing algorithms
4. **Bursty Traffic Handling**: Evaluation under realistic traffic conditions

## Future Work

- Multi-objective optimization (latency + fairness + energy)
- Real-world deployment integration
- Transfer learning across different cluster configurations
- Multi-agent coordination for large-scale systems

## References

- Mao et al. "Resource Management with Deep Reinforcement Learning" (HotNets 2016)
- Schulman et al. "Proximal Policy Optimization Algorithms" (arXiv 2017)
- Mnih et al. "Human-level control through deep RL" (Nature 2015)

## License

This project is part of a capstone thesis. Please cite appropriately if used for research.