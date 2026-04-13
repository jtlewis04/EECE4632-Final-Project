# board_evaluate.py — Watch a saved OS-ELM agent play Breakout on AUP-ZU3
#
# Runs in a Jupyter notebook. Renders via ipywidgets.Image.
# Uses software inference (numpy), not the FPGA overlay, so no
# bitstream is needed — just a saved .npz weight file.
#
# Usage (in a notebook cell):
#   %run board_evaluate.py oselm_ep-27.3score.npz

# Headless pygame (BEFORE any pygame import)
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Path setup
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "game"))
sys.path.insert(0, str(_root / "ai" / "software"))

import pygame as pg
import numpy as np
from IPython.display import display
from PIL import Image
import io
import ipywidgets as widgets

from ball import Ball
from bricks import Bricks
from paddle import Paddle
from settings import *
from scoreboard import ScoreBoard
from encoder import encode_game as extract_state, STATE_DIM
from os_elm_dqn import OSELM_QNetwork
from training_config import HIDDEN_DIM, NUM_ACTIONS, MAX_STEPS, REG, EPS1

# Configuration
WEIGHTS_DIR  = _root / "weights"
WEIGHTS_PATH = str(WEIGHTS_DIR / sys.argv[1]) if len(sys.argv) > 1 else str(WEIGHTS_DIR / "oselm_ep5000.npz")
NUM_EPISODES = 3
FRAME_SKIP   = 2   # render every Nth frame (raise to speed up)

# Load network
q_net = OSELM_QNetwork(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS, REG)
q_net.load(WEIGHTS_PATH)
print(f"Loaded weights from {WEIGHTS_PATH}")

# Pygame init
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Breakout — Agent Evaluation")
clock = pg.time.Clock()

# Jupyter display
img_widget = widgets.Image(format='jpeg', width=WIDTH, height=HEIGHT)
display(img_widget)


def render_frame():
    frame   = np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))
    pil_img = Image.fromarray(frame)
    buf     = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=70)
    img_widget.value = buf.getvalue()


for ep in range(NUM_EPISODES):
    pad    = Paddle(paddle_x, paddle_y)
    ball   = Ball(ball_x, ball_y, screen)
    bricks = Bricks(bricks_per_row, bricks_per_col, screen)
    score  = ScoreBoard(10, "white", screen)
    score.set_high_score()

    ball.y_speed = -abs(ball.y_speed)

    state = extract_state(ball, pad, bricks, score)
    total_reward = 0.0
    done  = False
    step  = 0

    while not done and step < MAX_STEPS:
        for event in pg.event.get():
            pass

        # Action (eps1 exploration, same as training)
        if np.random.rand() < EPS1:
            q_vals = q_net.predict_single(state)
            action = int(np.argmax(q_vals))
        else:
            action = np.random.randint(NUM_ACTIONS)

        if action == 0:
            pad.move_left()
        elif action != 1:
            pad.move_right()

        # Physics (identical order to training)
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

        # Reward (same as training, for display only)
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
        state = next_state
        total_reward += reward
        step += 1

        # Render to Jupyter widget
        screen.fill(BG_COLOR)
        score.show_scores()
        pad.appear(screen)
        bricks.show_bricks()
        pg.draw.circle(screen, ball_color, (int(ball.x), int(ball.y)), ball.radius)
        pg.display.flip()

        if step % FRAME_SKIP == 0:
            render_frame()
        clock.tick(60)

    # Final frame for this episode
    render_frame()

    print(f"Episode {ep+1}: Score={score.score}  Reward={total_reward:.1f}  "
          f"Steps={step}  BricksLeft={bricks.bricks_left()}")

pg.quit()
print("Evaluation complete.")
