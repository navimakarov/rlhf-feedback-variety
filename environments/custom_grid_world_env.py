import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MazeGameEnv(gym.Env):
    def __init__(self, maze=(
            ['S', '.', '.', '.'],
            ['.', '.', '.', '.'],
            ['.', '.', '.', '.'],
            ['.', '.', '.', 'G']
    )):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = np.argwhere(self.maze == 'S')[0]
        self.goal_pos = np.argwhere(self.maze == 'G')[0]
        self.current_pos = self.start_pos.copy()
        self.num_rows, self.num_cols = self.maze.shape

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_rows * self.num_cols)

        pygame.init()
        self.cell_size = 120
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

        self.steps = 0
        self.goal_achieved = False

    def _pos_to_index(self, pos):
        # Converts 2D position to 1D index
        return pos[0] * self.num_cols + pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        self.steps = 0
        self.goal_achieved = False
        return self._pos_to_index(self.current_pos), {}

    def step(self, action):
        self.steps += 1

        new_pos = self.current_pos.copy()
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        distance_to_goal = abs(self.goal_pos[0] - self.current_pos[0]) + abs(self.goal_pos[1] - self.current_pos[1])
        reward = -distance_to_goal

        done = distance_to_goal == 0

        return self._pos_to_index(self.current_pos), reward, done, False, {}

    def _is_valid_position(self, pos):
        row, col = pos
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False
        if self.maze[row, col] == '#':
            return False
        return True

    def render(self):
        self.screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                color = (255, 255, 255)  # Default background color

                if self.maze[row, col] == 'G':
                    color = (0, 255, 0)  # Red for goal

                pygame.draw.rect(self.screen, color, (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(self.current_pos, [row, col]):
                    high_res_scale = 4
                    high_res_size = self.cell_size * high_res_scale
                    high_res_circle = pygame.Surface((high_res_size, high_res_size), pygame.SRCALPHA)
                    pygame.draw.circle(high_res_circle, (0, 0, 255), (high_res_size // 2, high_res_size // 2),
                                       int(high_res_size * 0.25))
                    # Scale down and blit to the main screen
                    scaled_circle = pygame.transform.smoothscale(high_res_circle, (self.cell_size, self.cell_size))
                    self.screen.blit(scaled_circle, (cell_left, cell_top))
        # Draw grid lines
        for row in range(self.num_rows + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (0, row * self.cell_size),
                             (self.num_cols * self.cell_size, row * self.cell_size))
        for col in range(self.num_cols + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (col * self.cell_size, 0),
                             (col * self.cell_size, self.num_rows * self.cell_size))

        pygame.display.update()

