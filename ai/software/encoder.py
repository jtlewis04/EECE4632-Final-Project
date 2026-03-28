import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "game"))

import numpy as np
from settings import WIDTH, HEIGHT, paddle_width, paddle_y, ball_radius

STATE_DIM = 6


def _predict_landing_x(ball_x, ball_y, ball_x_speed, ball_y_speed):
    """Predict where ball crosses paddle y-line, accounting for wall bounces."""
    if ball_y_speed <= 0:
        return ball_x / WIDTH
    steps = (paddle_y - ball_y) / ball_y_speed
    if steps <= 0:
        return ball_x / WIDTH
    x_min = ball_radius
    x_max = WIDTH - ball_radius
    W = x_max - x_min
    if W <= 0:
        return 0.5
    unfolded = (ball_x - x_min) + ball_x_speed * steps
    folded = unfolded % (2 * W)
    if folded < 0:
        folded += 2 * W
    landing_x = x_min + (folded if folded <= W else 2 * W - folded)
    return np.clip(landing_x / WIDTH, 0.0, 1.0)


def encode_game(ball, paddle, bricks, score):
    """
    6-feature vector for FPGA-portable OS-ELM agent.

    [0]  ball_x / WIDTH
    [1]  ball_y / HEIGHT
    [2]  ball y-direction sign (+1 descending, -1 ascending)
    [3]  combo / combo_max
    [4]  paddle_x normalized (0..1)
    [5]  predicted landing_x normalized (0..1)
    """
    max_px = WIDTH - paddle.width

    return np.array([
        ball.x / WIDTH,
        ball.y / HEIGHT,
        1.0 if ball.y_speed >= 0 else -1.0,
        score.combo / max(score.combo_max, 1),
        paddle.rect.x / max_px if max_px > 0 else 0.5,
        _predict_landing_x(ball.x, ball.y, ball.x_speed, ball.y_speed),
    ], dtype=np.float64)