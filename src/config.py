# Configuration parameters for the Adaptive Load Balancing project

# Environment parameters
N_SERVERS = 3
MAX_QUEUE = 20
MAX_STEPS_PER_EPISODE = 200

# DQN Hyperparameters
HIDDEN_SIZE = 128          # Neurons per hidden layer
REPLAY_BUFFER_SIZE = 5000  # Experience replay buffer capacity
BATCH_SIZE = 32            # Mini-batch size for training
GAMMA = 0.99               # Discount factor
EPSILON_START = 1.0        # Initial exploration rate
EPSILON_END = 0.05         # Minimum exploration rate
EPSILON_DECAY = 0.995      # Epsilon decay per episode
TARGET_UPDATE_FREQ = 10    # Episodes between target network updates
LEARNING_RATE = 0.001      # Neural network learning rate
N_EPISODES = 20            # Training episodes

# Evaluation parameters
N_EVALUATION_EPISODES = 10

# Azure Dataset
AZURE_DATA_DIR = "data/extracted"
AZURE_DEFAULT_DAY = 1      # Load only Day 1 by default

# File paths
MODEL_PATH = "models/rl_load_balancer"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# Plot settings
PLOT_DPI = 150
FIGURE_SIZE = (10, 6)

# Random seeds for reproducibility
ENV_SEED = 42
MODEL_SEED = 42