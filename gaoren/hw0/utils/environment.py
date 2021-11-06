# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import pygame
from pygame.draw import rect
from numpy.random import RandomState
import numpy as np


class PointType:  # 枚举
    START = -2
    END = -1
    BG = 0
    WALL = 1
    VISITED = 2


class Environment:
    FPS = 10
    SQUARE_SIZE = 30  # 每个小正方形的大小. 20 * 20
    SCREEN_SIZE = 600, 300  # W, H
    #
    START_COLOR = "#379BBB"
    END_COLOR = "#D24B4E"
    BG_COLOR = "#A4F53C"
    WALL_COLOR = "#323929"  # barrier
    VISITED_COLOR = "#F2DB75"  # barrier
    COLOR_MAP = {
        PointType.START: START_COLOR, PointType.END: END_COLOR,
        PointType.BG: BG_COLOR, PointType.WALL: WALL_COLOR,
        PointType.VISITED: VISITED_COLOR
    }

    BLACK_COLOR = (0, 0, 0)
    RED_COLOR = (255, 0, 0)

    def __init__(self, wall_rate: float = 0.2):
        square_size = self.SQUARE_SIZE
        screen_size = self.SCREEN_SIZE
        #
        pygame.init()
        screen = pygame.display.set_mode(screen_size)  # W, H
        pygame.display.set_caption("Environment")
        fresh_clock = pygame.time.Clock()
        #
        self.screen = screen
        self.fresh_clock = fresh_clock
        self.env_matrix = np.zeros((screen_size[1] // square_size, screen_size[0] // square_size),
                                   dtype=np.int32)
        self.path = None
        #
        self.wall_rate = wall_rate

    def init_env(self, *, env_matrix=None, random_state: int = None) -> None:
        if env_matrix is not None:
            self.env_matrix = env_matrix
        else:
            random_state = random_state if isinstance(random_state, RandomState) \
                else RandomState(random_state)
            self._init_env_matrix_random(random_state)

    def _init_env_matrix_random(self, random_state=None):
        wall_rate = self.wall_rate
        env_matrix = np.ravel(self.env_matrix)  # view
        #
        idxs = random_state.permutation(env_matrix.size)
        start_idxs, end_idxs = idxs[:2]
        wall_idxs = idxs[2:int(env_matrix.size * wall_rate) + 2]
        env_matrix[start_idxs] = -2
        env_matrix[end_idxs] = -1
        env_matrix[wall_idxs] = 1

    def _draw_rect(self, i: int, j: int, square_type: int):
        screen = self.screen
        square_size = self.SQUARE_SIZE
        # 填充
        rect(screen, self.COLOR_MAP[square_type],
             (square_size * j + 1, square_size * i + 1, square_size - 2, square_size - 2))  # LTWH

    def step(self) -> None:
        screen_size = self.SCREEN_SIZE
        square_size = self.SQUARE_SIZE
        fps = self.FPS
        fresh_clock = self.fresh_clock
        env_matrix = self.env_matrix
        path = self.path
        #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        # draw
        for i in range(screen_size[1] // square_size):
            for j in range(screen_size[0] // square_size):
                self._draw_rect(i, j, env_matrix[i, j])
        if path:
            self._draw_path(path)
        #
        pygame.display.update()
        fresh_clock.tick(fps)

    def _draw_path(self, path):
        screen = self.screen
        red_color = self.RED_COLOR
        square_size = self.SQUARE_SIZE
        #
        path = [(square_size * pos[0] + square_size // 2,
                 square_size * pos[1] + square_size // 2)
                for pos in path]
        pygame.draw.lines(screen, red_color, False, path, 2)
