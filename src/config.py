# Configuration parameters for the Adaptive Load Balancing project

# Environment parameters
N_SERVERS = 3
MAX_QUEUE = 20
MAX_STEPS_PER_EPISODE = 200

# Training parameters
TOTAL_TIMESTEPS = 100000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10

# Evaluation parameters
N_EVALUATION_EPISODES = 20

# File paths
MODEL_PATH = "models/rl_load_balancer"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
TENSORBOARD_DIR = "tensorboard"

# Plot settings
PLOT_DPI = 150
FIGURE_SIZE = (10, 6)

# Random seeds for reproducibility
ENV_SEED = 42
MODEL_SEED = 42