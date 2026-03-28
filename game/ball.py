# ball.py
import pygame as pg
import math
from settings import *

class Ball:
    def __init__(self, x, y, screen):
        self.screen = screen
        self.x = x
        self.y = y
        self.radius = ball_radius

        self._min_speed = ((ball_x_speed**2 + ball_y_speed**2) ** 0.5) * 1.6
        self.base_speed = self._min_speed
        self.x_speed = ball_x_speed * 1.6
        self.y_speed = ball_y_speed * 1.6
        self._normalize_speed()

    def set_combo_speed(self, combo):
        self.base_speed = self._min_speed * (1 + 0.05 * combo)
        self._normalize_speed()

    def _normalize_speed(self):
        mag = (self.x_speed**2 + self.y_speed**2) ** 0.5
        if mag == 0:
            return
        s = self.base_speed / mag
        self.x_speed *= s
        self.y_speed *= s

    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

        if self.x - self.radius < 0:
            self.x = self.radius
        if self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius

        pg.draw.circle(self.screen, ball_color, (int(self.x), int(self.y)), self.radius)

    def bounce_y(self):
        self.y_speed *= -1
        self._normalize_speed()
        return "wall"

    def bounce_x(self):
        self.x_speed *= -1
        self._normalize_speed()
        return "wall"

    def bounce_from_paddle(self, paddle_rect):
        paddle_center = paddle_rect.x + paddle_rect.width / 2
        distance = (self.x - paddle_center) / (paddle_rect.width / 2)

        if distance < -1:
            distance = -1
        if distance > 1:
            distance = 1

        max_angle = 1.0
        angle = distance * max_angle

        self.x_speed = self.base_speed * math.sin(angle)
        self.y_speed = -abs(self.base_speed * math.cos(angle))
        self._normalize_speed()

    def check_for_contact_on_x(self):
        if self.x - self.radius <= 0:
            self.x = self.radius
            return self.bounce_x()

        if self.x + self.radius >= WIDTH:
            self.x = WIDTH - self.radius
            return self.bounce_x()

        return None

    def check_for_contact_on_y(self):
        if self.y - self.radius <= 0:
            self.y = self.radius
            return self.bounce_y()

        return None