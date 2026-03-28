# ── headless pygame setup (must be BEFORE any pygame import) ──────────────────
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# ── standard imports ───────────────────────────────────────────────────────────
import pygame as pg
import numpy as np
from IPython.display import display, clear_output
from PIL import Image
import io
import ipywidgets as widgets

from ball import Ball
from bricks import Bricks
from paddle import Paddle
from settings import *
from scoreboard import ScoreBoard
from encoder import encode_game

# ── try to import PYNQ buttons (won't crash if not available) ─────────────────
try:
    from pynq.overlays.base import BaseOverlay
    base = BaseOverlay("base.bit")
    btn0 = base.buttons[0]   # move left
    btn1 = base.buttons[1]   # move right
    USE_PYNQ_BUTTONS = True
    print("PYNQ buttons ready.")
except Exception:
    USE_PYNQ_BUTTONS = False
    print("No PYNQ buttons found — AI/manual mode only.")

# ── pygame init ────────────────────────────────────────────────────────────────
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Breakout Game")
clock = pg.time.Clock()

# ── game objects ───────────────────────────────────────────────────────────────
pad    = Paddle(paddle_x, paddle_y)
ball   = Ball(ball_x, ball_y, screen)
bricks = Bricks(bricks_per_row, bricks_per_col, screen)
score  = ScoreBoard(10, "white", screen)
score.set_high_score()

gameover = False
running  = True


RENDER_EVERY = 3   # show every Nth frame (raise to speed up, lower for smoothness)
frame_count  = 0

# ── widget buttons for manual control ─────────────────────────────────────────
btn_left  = widgets.Button(description="◀ Left",  button_style="info")
btn_right = widgets.Button(description="Right ▶", button_style="info")
btn_restart = widgets.Button(description="Restart (0)", button_style="warning")
move_left_flag  = [False]
move_right_flag = [False]

def on_left(b):  move_left_flag[0]  = True
def on_right(b): move_right_flag[0] = True
def on_restart(b):
    global gameover
    if gameover:
        score.score = 0
        gameover = False
        score.reset_combo()
        ball.set_combo_speed(0)
        bricks.reset_all()
        ball.x = ball_x
        ball.y = ball_y
        ball.y_speed = -abs(ball.y_speed)

btn_left.on_click(on_left)
btn_right.on_click(on_right)
btn_restart.on_click(on_restart)
img_widget = widgets.Image(format='jpeg', width=WIDTH, height=HEIGHT)
display(widgets.HBox([btn_left, btn_right, btn_restart]))
display(img_widget)

# ── main loop ─────────────────────────────────────────────────────────────────
while running:
    screen.fill(BG_COLOR)
    score.show_scores()
    pad.appear(screen)

    if bricks.bricks_left() == 0:
        score.success()

    elif not gameover:
        bricks.invade_update(pad.rect)
        ball.move()
        ball.check_for_contact_on_x()
        ball.check_for_contact_on_y()

        ball_rect = pg.Rect(
            int(ball.x - ball.radius), int(ball.y - ball.radius),
            ball.radius * 2, ball.radius * 2
        )
        if ball_rect.colliderect(pad.rect) and ball.y_speed > 0:
            ball.bounce_from_paddle(pad.rect)
            ball.y = pad.rect.top - ball.radius - 1
            score.reset_combo()
            ball.set_combo_speed(score.combo)

        if bricks.hit_by_ball(ball.x, ball.y, ball.radius):
            ball.bounce_y()
            score.brick_hit()
            ball.set_combo_speed(score.combo)

        if ball.y + ball.radius >= HEIGHT:
            bricks.reset_all()
            gameover = True

        # ── paddle control: PYNQ buttons → widget buttons → AI ───────────────
        if USE_PYNQ_BUTTONS:
            if btn0.read():
                pad.move_right()
            elif btn1.read():
                pad.move_left()
        else:
            if move_right_flag[0]:
                pad.move_right()
                move_right_flag[0] = False
            if move_left_flag[0]:
                pad.move_left()
                move_left_flag[0] = False

    else:
        score.game_over()

    bricks.show_bricks()

    # ── handle pygame events (quit / restart) ──────────────────────────────────
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if gameover and event.type == pg.KEYDOWN and event.key == pg.K_0:
            score.score = 0
            gameover    = False
            score.reset_combo()
            ball.set_combo_speed(0)
            bricks.reset_all()
            ball.x      = ball_x
            ball.y      = ball_y
            ball.y_speed = -abs(ball.y_speed)

    # ── render frame into notebook ─────────────────────────────────────────────
    frame_count += 1
    if frame_count % RENDER_EVERY == 0:
        frame = np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))
        pil_img = Image.fromarray(frame)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=70)
        img_widget.value = buf.getvalue()

    clock.tick(60)

pg.quit()
print("Game ended. Final score:", score.score)
