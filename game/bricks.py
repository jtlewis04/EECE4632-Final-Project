# bricks.py  (pure grid stepping, no pixel movement state)
import random
import pygame as pg
from settings import paddle_y

class Bricks:
    def __init__(self, bricks_per_row, bricks_per_col, screen):
        self.screen = screen
        self.cols = bricks_per_row
        self.start_rows = (int) (bricks_per_col / 3)
        self.random_colors = ['blue', 'yellow', 'red', 'green', 'orange', 'white']

        # render mapping only
        self.origin_x = 10
        self.origin_y = 100

        # size bricks to fill playable width and height with gaps
        playable_w = screen.get_width() - 2 * self.origin_x
        playable_h = paddle_y - self.origin_y
        self.gap_x = playable_w // bricks_per_row // 5
        self.gap_y = playable_h // bricks_per_col // 5
        self.brick_w = (playable_w - (bricks_per_row - 1) * self.gap_x) // bricks_per_row
        self.brick_h = (playable_h - (bricks_per_col - 1) * self.gap_y) // bricks_per_col

        self.grid = []
        self.color_grid = []
        self.set_values()

        self.start_delay_ms = 2000
        self.spawn_time = pg.time.get_ticks()

        self.step_interval_ms = 10000
        self.last_step_time = pg.time.get_ticks()

        self.max_rows = 200

    def set_values(self):
        self.grid = [[1 for _ in range(self.cols)] for _ in range(self.start_rows)]
        self.color_grid = [[random.choice(self.random_colors) for _ in range(self.cols)]
                           for _ in range(self.start_rows)]

    def reset_invade(self):
        now = pg.time.get_ticks()
        self.spawn_time = now
        self.last_step_time = now

    def reset_all(self):
        self.set_values()
        self.reset_invade()

    def bricks_left(self) -> int:
        return sum(sum(row) for row in self.grid)

    def _add_new_top_row(self):
        self.grid.insert(0, [1 for _ in range(self.cols)])
        self.color_grid.insert(0, [random.choice(self.random_colors) for _ in range(self.cols)])

        if len(self.grid) > self.max_rows:
            self.grid.pop()
            self.color_grid.pop()

    def _step_down(self):
        self._add_new_top_row()

    def _touch_row_index(self, paddle_rect) -> int:
        # row r touches when (origin_y + r*stride_y + brick_h) >= paddle_top
        stride_y = self.brick_h + self.gap_y
        numer = (paddle_rect.top - self.origin_y - self.brick_h)
        if numer <= 0:
            return 0
        return (numer + stride_y - 1) // stride_y  # ceil(numer/stride_y)

    def invade_update(self, paddle_rect) -> bool:
        now = pg.time.get_ticks()

        # during delay, keep timers synced so we dont "catch up" in one frame
        if now - self.spawn_time < self.start_delay_ms:
            self.last_step_time = now
            return False

        # step at most 1 row per frame to avoid jumps
        if now - self.last_step_time >= self.step_interval_ms:
            self.last_step_time = now
            self._step_down()

        # game over only if stack has actually reached the paddle row
        touch_r = self._touch_row_index(paddle_rect)
        if touch_r < 0:
            touch_r = 0

        if touch_r >= len(self.grid):
            return False

        for r in range(touch_r, len(self.grid)):
            if any(self.grid[r]):
                return True

        return False

    def hit_by_ball(self, ball_x, ball_y, ball_radius) -> bool:
        stride_x = self.brick_w + self.gap_x
        stride_y = self.brick_h + self.gap_y

        c = int((ball_x - self.origin_x) // stride_x)
        r = int((ball_y - self.origin_y) // stride_y)

        if r < 0 or c < 0:
            return False
        if r >= len(self.grid) or c >= self.cols:
            return False

        if self.grid[r][c] == 1:
            self.grid[r][c] = 0
            return True

        return False

    def show_bricks(self):
        stride_x = self.brick_w + self.gap_x
        stride_y = self.brick_h + self.gap_y
        screen_h = self.screen.get_height()

        for r in range(len(self.grid)):
            y = self.origin_y + r * stride_y
            if y > screen_h:
                return
            for c in range(self.cols):
                if self.grid[r][c] == 1:
                    x = self.origin_x + c * stride_x
                    rect = pg.Rect(x, y, self.brick_w, self.brick_h)
                    pg.draw.rect(self.screen, self.color_grid[r][c], rect)
