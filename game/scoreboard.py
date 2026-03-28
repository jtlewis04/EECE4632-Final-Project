# scores.py
import pygame as pg

class ScoreBoard:
    def __init__(self, x, color, screen):
        self.screen = screen
        self.color = color
        self.x = x
        self.score = 0
        self.high_score = 0
        self.font = pg.font.SysFont("calibri", 20)

        # combo system
        self.combo = 0
        self.base_points = 10
        self.combo_max = 50

    def show_scores(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.color)
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, self.color)
        combo_text = self.font.render(f"Combo: {self.combo}", True, self.color)

        self.screen.blit(score_text, (self.x, 10))
        self.screen.blit(high_score_text, (self.x, 26))
        self.screen.blit(combo_text, (self.x, 42))

    def brick_hit(self):
        self.combo = min(self.combo + 1, self.combo_max)
        points = self.combo
        self.score += points

    def reset_combo(self):
        self.combo = 0

    def game_over(self):
        font = pg.font.SysFont("calibri", 30)
        text = font.render("Game Over! Click '0' to restart.", True, "red")
        self.screen.blit(text, (50, 300))
        self.record_high_score()

    def success(self):
        font = pg.font.SysFont("calibri", 30)
        text = font.render("You won! Click '0' to restart.", True, "green")
        self.screen.blit(text, (50, 300))
        self.record_high_score()

    def set_high_score(self):
        try:
            with open("records.txt", "r") as file:
                score = file.readlines()[0]
        except FileNotFoundError:
            with open("records.txt", "w") as file:
                file.write("0")
            score = 0
        self.high_score = int(score)

    def record_high_score(self):
        if self.score > self.high_score:
            with open("records.txt", "w") as file:
                file.write(str(self.score))