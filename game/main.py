from ball import Ball
import pygame as pg
from bricks import Bricks
from paddle import Paddle
from settings import *
from scoreboard import ScoreBoard

pg.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Breakout Game")
clock = pg.time.Clock()

pad = Paddle(paddle_x, paddle_y)
ball = Ball(ball_x, ball_y, screen)
bricks = Bricks(bricks_per_row, bricks_per_col, screen)

score = ScoreBoard(10, "white", screen)
score.set_high_score()

gameover = False

running = True
while running:
    screen.fill(BG_COLOR)
    score.show_scores()
    pad.appear(screen)

    if bricks.bricks_left() == 0:
        score.success()

    elif(not gameover):
        bricks.invade_update(pad.rect)

        ball.move()

        ball.check_for_contact_on_x()
        ball.check_for_contact_on_y()

        # robust paddle hit (ball rect vs paddle rect)
        ball_rect = pg.Rect(int(ball.x - ball.radius), int(ball.y - ball.radius),
                            ball.radius * 2, ball.radius * 2)

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
    else:
        score.game_over()
            

    bricks.show_bricks()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        if gameover and event.type == pg.KEYDOWN and event.key == pg.K_0:
            score.score = 0
            gameover = False
            score.reset_combo()
            ball.set_combo_speed(0)

            bricks.reset_all()

            ball.x = ball_x
            ball.y = ball_y
            ball.y_speed = -abs(ball.y_speed)

    keys = pg.key.get_pressed()
    if keys[pg.K_RIGHT] and not gameover:
        pad.move_right()
    if keys[pg.K_LEFT] and not gameover:
        pad.move_left()

    pg.display.flip()
    clock.tick(60)

pg.quit()