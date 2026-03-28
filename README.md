# EECE4632-Final-Project
James Lewis and Akshaj Sirineni

## Project Overview
This project implements an OS-ELM (Online Sequential Extreme Learning Machine) based reinforcement learning agent to play a custom Breakout game, with the primary goal of comparing training and inference performance between an FPGA implementation on an AUP-ZU3 board (Zynq Ultrascale+ XCZU3EG MPSoC) and a pure software implementation running on a Windows PC with an AMD Ryzen 7 7800X3D. OS-ELM is chosen over a conventional backpropagation-trained DQN because its weight updates are analytically computed through recursive least squares with batch size 1, reducing the sequential training step to matrix multiplies, additions, and a single scalar division with no need for SVD, QR decomposition, or iterative gradient optimization. This makes the core compute loop directly mappable to fixed-point arithmetic in programmable logic. The FPGA implementation accelerates the two most frequently called operations: the forward pass prediction (computing Q-values for action selection) and the RLS rank-1 sequential weight update, while the one-time initial batch training, environment interaction, and control logic remain on the CPU. The agent encodes game state as a 6-feature vector and uses spectral normalization of the input weights and L2 regularization of the output weights to stabilize Q-learning, following the OS-ELM-L2-Lipschitz approach proposed by Watanabe et al.

## Game Description

![Breakout Game](media/breakout_game.png)

- Paddle moves left/right to bounce a ball into a grid of colored bricks
- Bricks slowly invade downward over time — if they reach the paddle, it's game over
- Combo system rewards consecutive brick hits without touching the paddle
- Ball speed increases with combo via `set_combo_speed(combo)`
- Single-life design: losing the ball ends the game immediately
- Press `0` to restart after a game over or win

## Files
### AI
- `ai/software/training_config.py`
  - Shared hyperparameters (network size, learning rates, episode limits) imported by both training and evaluation
  - **Network**
    - `HIDDEN_DIM = 64` — number of hidden neurons in the OS-ELM Q-network
    - `NUM_ACTIONS = 3` — left, stay, right
  - **Agent**
    - `GAMMA = 0.995` — discount factor for future rewards
    - `EPS1 = 0.95` — probability of picking the greedy action (5% random exploration)
    - `EPS2 = 0.9` — probability of performing a weight update each step
    - `UPDATE_STEP = 15` — episodes between target network syncs
    - `RESET_AFTER = 250` — episodes of no improvement before resetting weights
    - `REG = 0.1` — L2 regularization coefficient δ
    - `LAM = 0.9999` — forgetting factor for RLS (1.0 = no forgetting)
  - **Training loop**
    - `EPISODES = 10000` — total training episodes
    - `RENDER_EVERY = 0` — render every N episodes (0 = headless) (view the game state using pygame window)
    - `FAST_FPS = 0` — FPS cap when not rendering (0 = uncapped)
    - `MAX_STEPS = 15000` — maximum steps per episode before forced termination
- `ai/software/training_loop.py`
  - Main training script; runs the agent through episodes, performs OS-ELM updates, logs progress, and saves weights
- `ai/software/evaluate.py`
  - Loads a saved `.npz` weight file and renders the agent playing the game greedily (no exploration)
- `ai/software/os_elm_dqn.py`
  - Implements `OSELM_QNetwork` (forward pass, batch init, RLS rank-1 update, save/load) and `DQNAgent` (action selection, experience buffering, target network sync, weight reset)
- `ai/software/encoder.py`
  - Encodes game state into a 6-feature vector for the network: ball position, direction, combo, paddle position, and predicted ball landing x

### Game
- `game/main.py`
  - Entry point for manual play; handles pygame loop, keyboard input, and rendering
- `game/main_jupyter.py`
  - Jupyter-compatible version of the game loop for running on the AUP-ZU3 board with physical buttons
- `game/settings.py`
  - All game constants: screen dimensions, colors, paddle/ball/brick sizing and starting positions
- `game/ball.py`
  - Ball physics: movement, wall/ceiling bouncing, paddle bouncing, combo-driven speed scaling
- `game/paddle.py`
  - Paddle rendering and left/right movement
- `game/bricks.py`
  - Brick grid: rendering, ball collision detection, downward invasion logic
- `game/scoreboard.py`
  - Tracks and displays score, high score, and combo counter


## How to Run

### Install Dependencies
- Run `pip install -r requirements.txt`

### Play Manually
- Run `python game/main.py`
- Use left/right arrow keys to move the paddle
- Press '0' to restart on loss, exit to end the game

### Play on AUP-ZU3 Board
- Build `pygame` from source on AUP-ZU3 board
- Load all files in `game` onto the AUP-ZU3 board
- Run `main_jupyter.py` in Jupyter notebook
- Button0 to move left, Button 1 to move right, Button 2 to restart, Button 3 to exit

### Play using Agent Weights
- Place a saved `.npz` weight file in the `weights/` folder
- Run with the filename (not full path) as the argument:
```
python ai/software/evaluate.py "demo0-122score.npz"
```
- The agent will play `NUM_EPISODES` episodes (default: 2) using the same randomness factor (ESP1) as the training_loop. I.E. picks a random move `(1 - ESP1)*100%` of the time.
- Episode results (score, reward, steps, bricks left) are printed after each episode

### Train Agent (Software)
- Adjust hyperparameters in `ai/software/training_config.py` if desired
- Run from the project root:
```
python ai/software/training_loop.py
```
- Progress is printed every 25 episodes:
  - `Ep` — episode number
  - `R` — total reward for the most recent episode
  - `AvgRwd25` — mean reward over the last 25 episodes
  - `BestRwd25` — highest single-episode reward in the last 25 episodes
  - `ScAvg25` — mean game score over the last 25 episodes
  - `Resets` — total number of weight resets so far
  - `AvgSteps25` — mean steps per episode over the last 25 episodes
  - `BestSteps25` — highest step count in the last 25 episodes
  - `Init` — `Y` if the network has completed initial batch training, `N` if still collecting samples
- Weights are automatically saved to `weights/` when average score exceeds the threshold set in `training_loop.py`
- After `RESET_AFTER` episodes, will choose whether to reset based on the average score and if the change in reward is positive across the episodes (still improving)

### Train Agent (Hardware)
- Load files onto AUP-ZU3
- Run `training_loop_jupyter.py` in a Jupyter Notebook (TODO)