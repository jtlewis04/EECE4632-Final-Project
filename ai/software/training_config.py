# training_config.py — Shared hyperparameters for training and evaluation

# Network
HIDDEN_DIM  = 64       # Q-Network size
NUM_ACTIONS = 3        # LEFT, STAY, RIGHT

# Agent
GAMMA       = 0.995
EPS1        = 0.95     # greedy probability
EPS2        = 0.9      # update probability
UPDATE_STEP = 15       # target sync interval
RESET_AFTER = 250      # reset after N episodes of no improvement
REG         = 0.1      # L2 reg δ
LAM         = 0.9999  # forgetting factor

# Training loop
EPISODES    = 10000
RENDER_EVERY = 0       # 0 = headless
FAST_FPS    = 0        # 0 = uncapped when not rendering
MAX_STEPS   = 15000    # cap per episode
