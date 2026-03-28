# training-loop.py — OS-ELM-L2-Lipschitz training for Breakout
#
# Matches Watanabe et al. (arXiv:2005.04646) Algorithm 1:
#   - No experience replay after initial training
#   - Batch size 1 sequential updates
#   - Random update gating (ε₂)
#   - Fixed exploration rate (ε₁)
#   - Target network synced every UPDATE_STEP episodes
#   - Weight (evaluated to be) reset after RESET_AFTER failed episodes
#   - Q-value clipping [-10, +10]

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "game"))

import pygame as pg
import numpy as np

from ball import Ball
from bricks import Bricks
from paddle import Paddle
from settings import *
from scoreboard import ScoreBoard
from os_elm_dqn import DQNAgent
from encoder import encode_game as extract_state, STATE_DIM
from training_config import (HIDDEN_DIM, GAMMA, EPS1, EPS2, UPDATE_STEP,
                    RESET_AFTER, REG, LAM, EPISODES, RENDER_EVERY,
                    FAST_FPS, MAX_STEPS)


# ── Pygame init ─────────────────────────────────────────────
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Breakout — OS-ELM-L2-Lipschitz Training")
clock = pg.time.Clock()

agent = DQNAgent(
    hidden_dim=HIDDEN_DIM,
    gamma=GAMMA,
    eps1=EPS1,
    eps2=EPS2,
    update_step=UPDATE_STEP,
    reset_after=RESET_AFTER,
    reg=REG,
    lam=LAM,
)

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

ep_rewards = []
ep_scores = []
ep_steps_history = []
resets = 0


_start_time = time.perf_counter()

for ep in range(1, EPISODES + 1):
    # ── reset game objects ──────────────────────────────────
    pad   = Paddle(paddle_x, paddle_y)
    ball  = Ball(ball_x, ball_y, screen)
    bricks = Bricks(bricks_per_row, bricks_per_col, screen)
    score = ScoreBoard(10, "white", screen)
    score.set_high_score()

    ball.y_speed = -abs(ball.y_speed)

    state = extract_state(ball, pad, bricks, score)
    total_reward = 0.0
    done = False
    step = 0
    rendering = RENDER_EVERY > 0 and ep % RENDER_EVERY == 0

    while not done and step < MAX_STEPS:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                raise SystemExit

        # ── Determine (Algorithm 1 lines 10-13) ────────────
        action = agent.select_action(state)
        
        if action == 0:
            pad.move_left()
        elif action != 1:
            pad.move_right()

        # ── Observe (Algorithm 1 line 14) ───────────────────
        invaded = bricks.invade_update(pad.rect)

        ball.move()
        ball.check_for_contact_on_x()
        ball.check_for_contact_on_y()

        ball_rect = pg.Rect(
            int(ball.x - ball.radius), int(ball.y - ball.radius),
            ball.radius * 2, ball.radius * 2
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

        # ── Rewards ───
        reward = 0.000
        if brick_hit:
            reward += 2.0            # objective
        if paddle_hit:
            reward += 10.0           # survival signal
        if ball.y_speed > 0:
            pad_center = (pad.rect.x + pad.rect.width / 2) / WIDTH
            #reward += (1.0 - abs(pad_center - ball.x / WIDTH))
            reward += (1.0 - abs(pad_center - state[5])) * 0.1
        if ball_lost:
            reward -= 10.0
            done = True
        if invaded:
            reward -= 10.0
            done = True

        # ── next state ──────────────────────────────────────
        next_state = extract_state(ball, pad, bricks, score)

        # ── Store (Algorithm 1 line 17) ─────────────────────
        agent.store(state, action, reward, next_state, done)

        # ── Update (Algorithm 1 lines 18-23) ────────────────
        # Try initial training if enough samples collected
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        step += 1

        # ── render ──────────────────────────────────────────
        if rendering:
            screen.fill(BG_COLOR)
            score.show_scores()
            pad.appear(screen)
            bricks.show_bricks()
            pg.draw.circle(screen, ball_color,
                           (int(ball.x), int(ball.y)), ball.radius)
            pg.display.flip()
            clock.tick(60)
        else:
            clock.tick(FAST_FPS)

    # ── episode bookkeeping ─────────────────────────────────
    ep_rewards.append(total_reward)
    ep_scores.append(score.score)
    ep_steps_history.append(step)

    

    # End-of-episode: target sync + possible weight reset
    did_reset = agent.on_episode_end(ep, score.score, total_reward)
    if did_reset:
        resets += 1
        print(f"  *** RESET #{resets} at ep {ep} "
              f"(best score in window: {max(ep_scores[-RESET_AFTER:]) if len(ep_scores) >= RESET_AFTER else 0})")

    # ── logging ─────────────────────────────────────────────
    if ep % 25 == 0:
        avg25 = np.mean(ep_rewards[-25:]) if len(ep_rewards) >= 25 else np.mean(ep_rewards)

        sc_avg = np.mean(ep_scores[-25:]) if len(ep_scores) >= 25 else np.mean(ep_scores)
        ep_steps = ep_steps_history[-25:] if len(ep_steps_history) >= 25 else ep_steps_history
        avg_steps25 = np.mean(ep_steps)
        best_rwd25 = max(ep_rewards[-25:] if len(ep_rewards) >= 25 else ep_rewards)
        best_steps25 = max(ep_steps)
        print(f"Ep {ep:5d} | R={total_reward:7.1f} | "
              f"AvgRwd25={avg25:6.1f} | BestRwd25={best_rwd25:6.1f} | "
              f"ScAvg25={sc_avg:5.1f} | Resets={resets} | "
              f"AvgSteps25={avg_steps25:7.1f} | BestSteps25={best_steps25:7.1f} | "
              f"Init={'Y' if agent.q_net.is_initialized else 'N'}")
        
        if sc_avg > 25: #Arbitrary score to start saving weights
            agent.q_net.save(str(WEIGHTS_DIR / f"oselm_ep-{sc_avg}score.npz"))
            print(f"  -> weights saved (avg score: {sc_avg})")
    
    
    
        

pg.quit()
print(f"Total resets: {resets}")
elapsed = time.perf_counter() - _start_time
print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}m)")