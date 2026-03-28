# paddle.py
import pygame as pg
from settings import paddle_height, paddle_width

class Paddle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = paddle_width
        self.height = paddle_height
        self.rect = pg.Rect(self.x, self.y, self.width, self.height)

        # ruler segments
        self.segments = 5
        self.segment_colors = [
            pg.Color("red"),
            pg.Color("orange"),
            pg.Color("yellow"),
            pg.Color("lightgreen"),
            pg.Color("cyan"),
        ]

        # outline
        self.outline_color = pg.Color("white")

    def appear(self, screen):
        seg_w = self.width / self.segments

        # colored ruler body
        for i in range(self.segments):
            x0 = self.rect.x + int(i * seg_w)
            w = int(seg_w) if i < self.segments - 1 else (self.rect.right - x0)
            seg_rect = pg.Rect(x0, self.rect.y, w, self.height)
            pg.draw.rect(screen, self.segment_colors[i], seg_rect)

            # small tick mark at segment boundary (except first)
            if i > 0:
                tick_x = x0
                pg.draw.line(screen, self.outline_color,
                             (tick_x, self.rect.y),
                             (tick_x, self.rect.y + self.height), 2)

        # border
        pg.draw.rect(screen, self.outline_color, self.rect, 2)

    def move_right(self):
        if self.rect.x + self.width <= 550:
            self.rect.x += 5

    def move_left(self):
        if self.rect.x >= 0:
            self.rect.x -= 5

    def hit_segment(self, ball_x):
        # returns 0..segments-1 based on where the ball hit
        rel = (ball_x - self.rect.x) / self.rect.width
        if rel < 0:
            rel = 0
        if rel > 0.999999:
            rel = 0.999999
        return int(rel * self.segments)