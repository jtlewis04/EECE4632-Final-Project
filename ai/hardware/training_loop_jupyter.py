# training_loop_jupyter.py — OS-ELM-L2-Lipschitz training on AUP-ZU3 with FPGA
#
# Paper: Watanabe et al. (arXiv:2005.04646)
#
# CPU (PS) handles:
#   - Environment interaction (Steps 2b / 4b)
#   - Initial training via init_batch (Step 2c, Equation 7)
#   - Epsilon-greedy action selection (coin flip + argmax of FPGA Q-values)
#   - Reward computation, episode bookkeeping, weight reset logic
#
# FPGA (PL) handles:
#   - Predict Q-values using θ₁ for action selection (Step 4a)
#   - Predict Q-values using θ₂ for target computation (part of Step 4c)
#   - Sequential RLS rank-1 weight update on θ₁ (Step 4c, Equation 5)
#   - Target network sync: copy θ₁ β -> θ₂ β (Lines 24-25)
#
# DMA protocol (single AXI DMA, MM2S + S2MM, HP0 FPD):
#   All values are 32-bit.  Opcodes and action indices are raw int32.
#   All other numeric values are Q20 fixed-point (ap_fixed<32,12>).
#
#   OP 0  PREDICT_Q:     send [0] + state[6]                      -> recv q[3]
#   OP 1  PREDICT_TGT:   send [1] + state[6]                      -> recv q[3]
#   OP 2  TRAIN_SEQ:     send [2] + state[6] + action + target    -> recv ack[1]
#   OP 3  LOAD_WEIGHTS:  send [3] + W_in[S*H] + b[H] + β[H*A] + P[H*H]
#                                                                  -> recv ack[1]
#   OP 4  READ_WEIGHTS:  send [4]             -> recv β[H*A] + P[H*H]
#   OP 5  SYNC_TARGET:   send [5]             -> recv ack[1]

# headless pygame (BEFORE any pygame import)
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# path setup
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "game"))
sys.path.insert(0, str(_root / "ai" / "software"))

# imports
import pygame as pg
import numpy as np
import time
from collections import deque
from IPython.display import display
from PIL import Image
import io
import ipywidgets as widgets

from pynq import Overlay, allocate

from ball import Ball
from bricks import Bricks
from paddle import Paddle
from settings import *
from scoreboard import ScoreBoard
from encoder import encode_game as extract_state, STATE_DIM
from training_config import (
    HIDDEN_DIM, NUM_ACTIONS, GAMMA, EPS1, EPS2,
    UPDATE_STEP, RESET_AFTER, REG, LAM,
    EPISODES, RENDER_EVERY, MAX_STEPS,
)

#  Q20 fixed-point helpers
FRAC_BITS = 20
SCALE = 1 << FRAC_BITS          # 1048576


def to_q20(val):
    """float (scalar or array) → Q20 int32."""
    return np.round(np.asarray(val, dtype=np.float64) * SCALE).astype(np.int32)


def from_q20(val):
    """Q20 int32 → float64."""
    return np.asarray(val, dtype=np.float64) / SCALE


#  DMA opcodes
OP_PREDICT_Q    = 0
OP_PREDICT_TGT  = 1
OP_TRAIN_SEQ    = 2
OP_LOAD_WEIGHTS = 3
OP_READ_WEIGHTS = 4
OP_SYNC_TARGET  = 5


#  FPGAAgent — DMA wrapper for the OS-ELM HLS core
class FPGAAgent:
    """Communicates with the OS-ELM FPGA core via AXI DMA."""

    def __init__(self, overlay_path, dma_name="axi_dma_0"):
        self.ol  = Overlay(overlay_path)
        self.dma = getattr(self.ol, dma_name)

        # Pre-allocate contiguous DMA buffers (avoids per-call allocation)
        # -- predict --
        self._pred_in  = allocate(shape=(7,), dtype=np.int32)
        self._pred_out = allocate(shape=(3,), dtype=np.int32)
        # -- train_seq --
        self._train_in  = allocate(shape=(9,), dtype=np.int32)
        self._train_ack = allocate(shape=(1,), dtype=np.int32)
        # -- load_weights --
        _W = STATE_DIM * HIDDEN_DIM
        _b = HIDDEN_DIM
        _beta = HIDDEN_DIM * NUM_ACTIONS
        _P = HIDDEN_DIM * HIDDEN_DIM
        self._load_in  = allocate(shape=(1 + _W + _b + _beta + _P,), dtype=np.int32)
        self._load_ack = allocate(shape=(1,), dtype=np.int32)
        # -- read_weights --
        self._read_in  = allocate(shape=(1,), dtype=np.int32)
        self._read_out = allocate(shape=(_beta + _P,), dtype=np.int32)
        # -- sync_target --
        self._sync_in  = allocate(shape=(1,), dtype=np.int32)
        self._sync_ack = allocate(shape=(1,), dtype=np.int32)

    # low-level DMA transfer
    def _xfer(self, send_buf, recv_buf):
        """Start recv first (so S2MM is ready), then send, then wait."""
        self.dma.recvchannel.transfer(recv_buf)
        self.dma.sendchannel.transfer(send_buf)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

    # high-level operations
    def predict(self, state, use_target=False):
        """OP 0/1 — returns Q-values (3,) as float64."""
        self._pred_in[0] = OP_PREDICT_TGT if use_target else OP_PREDICT_Q
        self._pred_in[1:7] = to_q20(state)
        self._xfer(self._pred_in, self._pred_out)
        return from_q20(np.array(self._pred_out))

    def train_seq(self, state, action, target):
        """OP 2 — RLS rank-1 update on θ₁ inside FPGA."""
        self._train_in[0] = OP_TRAIN_SEQ
        self._train_in[1:7] = to_q20(state)
        self._train_in[7]   = int(action)          # raw int
        self._train_in[8]   = int(to_q20(target))   # Q20 scalar
        self._xfer(self._train_in, self._train_ack)

    def load_weights(self, W_in, b, beta, P):
        """OP 3 — upload W_in, b, β, P into FPGA BRAM."""
        buf = self._load_in
        idx = 0
        buf[idx] = OP_LOAD_WEIGHTS; idx += 1

        w = to_q20(W_in.flatten(order='C'))
        buf[idx:idx + len(w)] = w; idx += len(w)

        bq = to_q20(b.flatten())
        buf[idx:idx + len(bq)] = bq; idx += len(bq)

        betaq = to_q20(beta.flatten(order='C'))
        buf[idx:idx + len(betaq)] = betaq; idx += len(betaq)

        pq = to_q20(P.flatten(order='C'))
        buf[idx:idx + len(pq)] = pq; idx += len(pq)

        self._xfer(buf, self._load_ack)

    def read_weights(self):
        """OP 4 — read β (H×A) and P (H×H) back from FPGA."""
        self._read_in[0] = OP_READ_WEIGHTS
        self._xfer(self._read_in, self._read_out)
        data = np.array(self._read_out)   # copy from CMA
        bs = HIDDEN_DIM * NUM_ACTIONS
        beta = from_q20(data[:bs]).reshape(HIDDEN_DIM, NUM_ACTIONS)
        P    = from_q20(data[bs:]).reshape(HIDDEN_DIM, HIDDEN_DIM)
        return beta, P

    def sync_target(self):
        """OP 5 — copy θ₁ β → θ₂ β inside FPGA."""
        self._sync_in[0] = OP_SYNC_TARGET
        self._xfer(self._sync_in, self._sync_ack)


#  SoftwareOSELM — CPU-side init_batch (Paper Step 2c)
class SoftwareOSELM:
    """Generates random α (with spectral norm) and runs init_batch on CPU."""

    def __init__(self, hidden_dim=HIDDEN_DIM, reg=REG, seed=42):
        self.hidden_dim = hidden_dim
        self.reg = reg
        self.seed = seed
        self._init_params()

    def _init_params(self):
        rng = np.random.RandomState(self.seed)
        # Algorithm 1 lines 1-3
        self.W_in = rng.rand(STATE_DIM, self.hidden_dim).astype(np.float64)
        self.b    = rng.rand(self.hidden_dim).astype(np.float64)
        _, S, _ = np.linalg.svd(self.W_in, full_matrices=False)
        self.W_in /= S[0]
        self.beta = np.zeros((self.hidden_dim, NUM_ACTIONS), dtype=np.float64)
        self.P = None
        self.is_initialized = False

    def _hidden(self, X):
        return np.maximum(0.0, X @ self.W_in + self.b)

    def init_batch(self, states, actions, targets):
        """Equation 7: P₀ = (H^T H + δI)^{-1}, β₀ = P₀ H^T Y."""
        H = self._hidden(states)
        HtH = H.T @ H + self.reg * np.eye(self.hidden_dim, dtype=np.float64)
        self.P = np.linalg.inv(HtH)
        Y = H @ self.beta                # (N, A) — all zeros before init
        for i in range(len(states)):
            Y[i, actions[i]] = targets[i]
        self.beta = self.P @ (H.T @ Y)
        self.is_initialized = True

    def reset(self):
        """New random seed → new α, clear β/P."""
        self.seed += 1
        self._init_params()


#  Configuration
OVERLAY_PATH = "os_elm.bit"           # ← change to your bitstream path
WEIGHTS_DIR  = _root / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

JUPYTER_RENDER_EVERY = 0              # 0 = headless, N = render every N eps
JUPYTER_FRAME_SKIP   = 3              # show every Nth frame during rendered eps


#  Pygame + Jupyter display
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Breakout — FPGA Training")
clock = pg.time.Clock()

img_widget = widgets.Image(format='jpeg', width=WIDTH, height=HEIGHT)
display(img_widget)


def render_frame():
    """Capture pygame surface → Jupyter Image widget."""
    frame   = np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))
    pil_img = Image.fromarray(frame)
    buf     = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=70)
    img_widget.value = buf.getvalue()


#  Initialize agents
sw   = SoftwareOSELM(HIDDEN_DIM, REG)
fpga = FPGAAgent(OVERLAY_PATH)


#  Training state
init_buf         = []                  # buffer D (pre-init samples)
is_initialized   = False
global_step      = 0
ep_since_reset   = 0
recent_scores    = deque(maxlen=RESET_AFTER)
recent_rewards   = deque(maxlen=25)
resets           = 0

ep_rewards       = []
ep_scores        = []
ep_steps_history = []

_start = time.perf_counter()


#  Training loop
for ep in range(1, EPISODES + 1):

    # reset game objects
    pad    = Paddle(paddle_x, paddle_y)
    ball   = Ball(ball_x, ball_y, screen)
    bricks = Bricks(bricks_per_row, bricks_per_col, screen)
    score  = ScoreBoard(10, "white", screen)
    score.set_high_score()
    ball.y_speed = -abs(ball.y_speed)

    state        = extract_state(ball, pad, bricks, score)
    total_reward = 0.0
    done         = False
    step         = 0
    rendering    = JUPYTER_RENDER_EVERY > 0 and ep % JUPYTER_RENDER_EVERY == 0

    while not done and step < MAX_STEPS:
        global_step += 1

        # Action selection (Algorithm 1 lines 10-13)
        if is_initialized:
            # Step 4a: FPGA predict with θ₁
            if np.random.rand() < EPS1:
                q = fpga.predict(state, use_target=False)
                if np.all(np.isfinite(q)):
                    action = int(np.argmax(q))
                else:
                    action = np.random.randint(NUM_ACTIONS)
            else:
                action = np.random.randint(NUM_ACTIONS)
        else:
            # Step 2a: random action while collecting buffer
            action = np.random.randint(NUM_ACTIONS)

        if action == 0:
            pad.move_left()
        elif action != 1:
            pad.move_right()

        # Environment step (Algorithm 1 line 14)
        invaded = bricks.invade_update(pad.rect)

        ball.move()
        ball.check_for_contact_on_x()
        ball.check_for_contact_on_y()

        ball_rect = pg.Rect(
            int(ball.x - ball.radius), int(ball.y - ball.radius),
            ball.radius * 2, ball.radius * 2,
        )
        paddle_hit = False
        if ball_rect.colliderect(pad.rect) and ball.y_speed > 0:
            ball.bounce_from_paddle(pad.rect)
            ball.y = pad.rect.top - ball.radius - 1
            score.reset_combo()
            ball.set_combo_speed(score.combo)
            paddle_hit = True

        brick_hit = bricks.hit_by_ball(ball.x, ball.y, ball.radius)
        if brick_hit:
            ball.bounce_y()
            score.brick_hit()
            ball.set_combo_speed(score.combo)

        ball_lost = ball.y + ball.radius >= HEIGHT

        # Reward (same as software training_loop)
        reward = 0.0
        if brick_hit:
            reward += 2.0
        if paddle_hit:
            reward += 10.0
        if ball.y_speed > 0:
            pad_center = (pad.rect.x + pad.rect.width / 2) / WIDTH
            reward += (1.0 - abs(pad_center - state[5])) * 0.1
        if ball_lost:
            reward -= 10.0
            done = True
        if invaded:
            reward -= 10.0
            done = True

        next_state = extract_state(ball, pad, bricks, score)

        # Store / Update
        if not is_initialized:
            # Step 2b: accumulate buffer D
            init_buf.append((state, action, reward, next_state, done))

            if len(init_buf) >= HIDDEN_DIM:
                # Step 2c: initial training on CPU (Equation 7)
                buf = init_buf[:HIDDEN_DIM]
                ss, aa, rr, sn, dd = zip(*buf)
                S_arr = np.array(ss, dtype=np.float64)
                A_arr = np.array(aa, dtype=np.int32)
                R_arr = np.clip(np.array(rr, dtype=np.float64), -10.0, 20.0)
                sw.init_batch(S_arr, A_arr, R_arr)
                is_initialized = True
                init_buf.clear()

                # Step 3: upload α, b, β, P to FPGA BRAM
                fpga.load_weights(sw.W_in, sw.b, sw.beta, sw.P)
                fpga.sync_target()            # θ₂ ← θ₁
                print(f"  Init done at ep {ep}, global_step {global_step}. "
                      f"Weights uploaded to FPGA.")
        else:
            # Step 4c: sequential training (ε₂ gating on CPU)
            if np.random.rand() < EPS2:
                # Predict with θ₂ (target net) for TD target
                q_next = fpga.predict(next_state, use_target=True)
                if np.all(np.isfinite(q_next)):
                    target = reward + (1.0 - float(done)) * GAMMA * float(np.max(q_next))
                    target = float(np.clip(target, -10.0, 20.0))
                    # RLS update on θ₁
                    fpga.train_seq(state, action, target)

        state = next_state
        total_reward += reward
        step += 1

        # Render (Jupyter)
        if rendering:
            screen.fill(BG_COLOR)
            score.show_scores()
            pad.appear(screen)
            bricks.show_bricks()
            pg.draw.circle(screen, ball_color,
                           (int(ball.x), int(ball.y)), ball.radius)
            pg.display.flip()
            if step % JUPYTER_FRAME_SKIP == 0:
                render_frame()


    #  Episodes
    ep_rewards.append(total_reward)
    ep_scores.append(score.score)
    ep_steps_history.append(step)

    ep_since_reset += 1
    recent_scores.append(score.score)
    recent_rewards.append(total_reward)

    # Target sync (Algorithm 1 lines 24-25)
    if is_initialized and ep % UPDATE_STEP == 0:
        fpga.sync_target()

    # Weight reset check (same logic as software DQNAgent)
    if ep_since_reset > 0 and ep_since_reset % RESET_AFTER == 0:
        min_avg = 1.5 + (ep_since_reset // RESET_AFTER) * 0.25
        half = len(recent_scores) // 2
        first_half_avg  = np.mean(list(recent_scores)[:half]) if half > 0 else 0
        second_half_avg = np.mean(list(recent_scores)[half:]) if half > 0 else 0

        if second_half_avg <= first_half_avg + 0.1 and second_half_avg < min_avg:
            print(f"  second half avg:{second_half_avg:.1f}")
            sw.reset()
            init_buf.clear()
            is_initialized = False
            global_step = 0
            ep_since_reset = 0
            recent_scores.clear()
            recent_rewards.clear()
            resets += 1
            print(f"  *** RESET #{resets} at ep {ep}")

    # Logging (every 25 episodes)
    if ep % 25 == 0:
        avg25 = (np.mean(ep_rewards[-25:])
                 if len(ep_rewards) >= 25 else np.mean(ep_rewards))
        sc_avg = (np.mean(ep_scores[-25:])
                  if len(ep_scores) >= 25 else np.mean(ep_scores))
        steps_win = (ep_steps_history[-25:]
                     if len(ep_steps_history) >= 25 else ep_steps_history)
        avg_steps25  = np.mean(steps_win)
        best_rwd25   = max(ep_rewards[-25:]
                           if len(ep_rewards) >= 25 else ep_rewards)
        best_steps25 = max(steps_win)

        print(
            f"Ep {ep:5d} | R={total_reward:7.1f} | "
            f"AvgRwd25={avg25:6.1f} | BestRwd25={best_rwd25:6.1f} | "
            f"ScAvg25={sc_avg:5.1f} | Resets={resets} | "
            f"AvgSteps25={avg_steps25:7.1f} | BestSteps25={best_steps25:7.1f} | "
            f"Init={'Y' if is_initialized else 'N'}"
        )

        # Save weights (read β,P back from FPGA)
        if sc_avg > 25 and is_initialized:
            beta, P = fpga.read_weights()
            save_path = str(WEIGHTS_DIR / f"fpga_oselm-{sc_avg:.1f}score.npz")
            np.savez(save_path, W_in=sw.W_in, b=sw.b, beta=beta, P=P)
            print(f"  -> weights saved to {save_path}")


#  Quit
pg.quit()
elapsed = time.perf_counter() - _start
print(f"\nTotal resets: {resets}")
print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
