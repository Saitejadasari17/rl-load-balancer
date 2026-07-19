# Configuration parameters for the Adaptive Load Balancing project

# Environment parameters
N_SERVERS = 3
MAX_QUEUE = 20
MAX_STEPS_PER_EPISODE = 200

# ── DQN Hyperparameters (Dueling Double DQN + Prioritized Experience Replay) ──
HIDDEN_SIZE = 128              # Neurons per hidden layer in the shared trunk
REPLAY_BUFFER_SIZE = 20000     # Experience replay buffer capacity
BATCH_SIZE = 64                # Mini-batch size for training
GAMMA = 0.99                   # Discount factor
LEARNING_RATE = 5e-4           # AdamW learning rate
WEIGHT_DECAY = 1e-5            # AdamW weight decay
GRAD_CLIP = 10.0               # Max gradient norm (gradient clipping)
TAU = 0.005                    # Soft target-network update coefficient (Polyak averaging)
TRAIN_EVERY = 2                # Gradient update every N environment steps (speed/throughput knob)
WARMUP_STEPS = 2000             # Transitions collected before any gradient update starts

# Exploration (epsilon-greedy), annealed linearly over environment steps
EPSILON_START = 1.0
EPSILON_END = 0.1              # Kept relatively high: a small but permanent exploration
                                # rate prevents the agent from locking onto a degenerate
                                # "always flood one server" policy and never correcting it
EPSILON_DECAY_STEPS = 50000    # train_rl_agent() auto-scales this to ~60% of the
                                # run's total steps unless explicitly overridden

# Prioritized Experience Replay
PER_ALPHA = 0.4                # How much prioritization is used (0 = uniform, 1 = full)
PER_BETA_START = 0.4           # Initial importance-sampling correction exponent
PER_BETA_FRAMES = 100000       # Steps over which beta is annealed to 1.0

# Training budget / early stopping
N_EPISODES = 200               # Default training episodes for a full run
EARLY_STOPPING_PATIENCE = 40      # Episodes without a new best checkpoint before stopping
EARLY_STOPPING_MIN_DELTA_MS = 0.5 # Minimum avg-latency improvement (ms) to count as a new best
EARLY_STOPPING_MIN_EPISODES = 30
EVAL_EVERY = 10                   # How often (episodes) to run a clean eval for checkpoint selection
EVAL_EPISODES_FOR_SELECTION = 3   # Episodes per periodic selection eval

# Reproducibility & hardware
SEED = 42
DEVICE = None                  # None = auto-detect (CUDA > Apple MPS > CPU)
USE_AMP = None                 # None = auto (mixed precision only on CUDA)

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
