import pygame
import numpy as np
import random
from matplotlib import pyplot as plt
import imageio
import os

# Конфигурация
GRID_SIZE = 3
CELL_SIZE_WINDOW = 100
HEADER_HEIGHT_WINDOW = 50
FPS = 30
EPISODES = 100
ALPHA = 0.3
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
MAX_STEPS = 50
MOVE_THRESHOLD = 30
OUTPUT_DIR = "numpuz_results"

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

class NumpuzEnv:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(size, size)
        self.goal = self.grid.copy()
        self.empty_pos = (size - 1, size - 1)
        self.actions = ['up', 'down', 'left', 'right']
        self.goal_positions = {}
        for i in range(size):
            for j in range(size):
                self.goal_positions[self.grid[i, j]] = (i, j)
        self.goal_positions[9] = None
        self.current_target = 1
        self.correct_tiles = set()
        self.reset()

    def reset(self):
        tiles = list(range(10))
        random.shuffle(tiles)
        self.grid = np.array(tiles[:9]).reshape(self.size, self.size)
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:
                    self.empty_pos = (i, j)
        self.current_target = 1
        self.correct_tiles = set()
        return self.get_state()

    def get_state(self):
        target_pos = None
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == self.current_target:
                    target_pos = (i, j)
                    break
            if target_pos:
                break
        if not target_pos:
            target_pos = self.empty_pos
        return (self.empty_pos, target_pos, self.current_target)

    def is_solved(self):
        for i in range(self.size):
            for j in range(self.size):
                val = self.grid[i, j]
                if val != 9 and val != self.goal[i, j]:
                    return False
        return True

    def get_valid_actions(self):
        x, y = self.empty_pos
        valid = []
        if x > 0:
            valid.append('down')
        if x < self.size - 1:
            valid.append('up')
        if y > 0:
            valid.append('right')
        if y < self.size - 1:
            valid.append('left')
        return valid

    def manhattan_distance(self, num, pos):
        if num == 9:
            return 0
        gi, gj = self.goal_positions[num]
        return abs(pos[0] - gi) + abs(pos[1] - gj)

    def check_correct_tiles(self):
        disturbed = False
        for num in self.correct_tiles:
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i, j] == num:
                        if (i, j) != self.goal_positions[num]:
                            disturbed = True
                        break
        return disturbed

    def step(self, action, move_count):
        x, y = self.empty_pos
        new_empty = None
        tile_pos = None
        if action == 'up':
            new_empty = (x + 1, y)
            tile_pos = (x + 1, y)
        elif action == 'down':
            new_empty = (x - 1, y)
            tile_pos = (x - 1, y)
        elif action == 'left':
            new_empty = (x, y + 1)
            tile_pos = (x, y + 1)
        elif action == 'right':
            new_empty = (x, y - 1)
            tile_pos = (x, y - 1)

        reward = -1
        done = False

        prev_distance = self.manhattan_distance(self.current_target, self.get_current_target_pos())
        if move_count > MOVE_THRESHOLD:
            reward -= 10

        if new_empty and 0 <= new_empty[0] < self.size and 0 <= new_empty[1] < self.size:
            moving_tile = self.grid[tile_pos[0], tile_pos[1]]
            if moving_tile in self.correct_tiles and moving_tile != 0:
                reward -= 30
            else:
                self.grid[x, y], self.grid[tile_pos[0], tile_pos[1]] = self.grid[tile_pos[0], tile_pos[1]], self.grid[x, y]
                self.empty_pos = new_empty
                new_distance = self.manhattan_distance(self.current_target, self.get_current_target_pos())
                if new_distance < prev_distance:
                    reward += 10
                if self.is_target_in_place():
                    reward += 100
                    self.correct_tiles.add(self.current_target)
                    self.current_target += 1
                    if self.current_target > self.size * self.size - 1:
                        if self.is_solved():
                            reward += 200
                            done = True
                if self.check_correct_tiles():
                    reward -= 30
                if self.is_solved():
                    reward += 200
                    done = True

        return self.get_state(), reward, done

    def get_current_target_pos(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == self.current_target:
                    return (i, j)
        return self.empty_pos

    def is_target_in_place(self):
        target_pos = self.get_current_target_pos()
        return target_pos == self.goal_positions[self.current_target]

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        self.Q = {}
        self.epsilon = EPSILON_START

    def get_q_values(self, state):
        if state not in self.Q:
            self.Q[state] = np.ones(len(self.actions)) * 0.1
        return self.Q[state]

    def select_action(self, state):
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        q_values = self.get_q_values(state)
        valid_indices = [self.actions.index(a) for a in valid_actions]
        return self.actions[max(valid_indices, key=lambda i: q_values[i])]

    def update(self, state, action, reward, next_state, done):
        if action is None:
            return
        a_idx = self.actions.index(action)
        current_q = self.get_q_values(state)[a_idx]
        next_q = max(self.get_q_values(next_state)) if not done else 0
        target = reward + GAMMA * next_q
        self.Q[state][a_idx] += ALPHA * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

def draw_grid(screen, env, episode, operation_count, step, cell_size, header_height):
    screen.fill(WHITE)
    font_header = pygame.font.SysFont('arial', int(header_height * 0.5), bold=True)
    text = f"Эпизод: {episode}  Ходы: {operation_count}  Шаг: {step}  Цель: {env.current_target}"
    text_surface = font_header.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(env.size * cell_size // 2, header_height // 2))
    screen.blit(text_surface, text_rect)

    number_colors = [
        (255, 204, 204), (255, 229, 204), (255, 255, 204), (204, 255, 204),
        (204, 255, 255), (204, 229, 255), (204, 204, 255), (229, 204, 255),
        (255, 204, 229)
    ]

    for i in range(env.size):
        for j in range(env.size):
            rect = pygame.Rect(j * cell_size, i * cell_size + header_height, cell_size, cell_size)
            value = env.grid[i, j]

            # Цвет фона плитки
            if value == 0:
                tile_color = GRAY
            else:
                tile_color = number_colors[value - 1]

            # Подсветка правильных позиций
            is_correct = (value in env.correct_tiles and value != 0)
            is_target = (value == env.current_target and value != 0)

            if is_correct:
                tile_color = (144, 238, 144)  # светло-зелёный
            elif is_target:
                tile_color = (135, 206, 250)  # голубой

            # Отрисовка плитки
            pygame.draw.rect(screen, tile_color, rect, border_radius=10)

            # Тень
            shadow = pygame.Rect(rect.x + 3, rect.y + 3, rect.width, rect.height)
            pygame.draw.rect(screen, (180, 180, 180), shadow, border_radius=10)

            # Текст плитки
            if value != 0:
                font_tile = pygame.font.SysFont('arial', int(cell_size * 0.5), bold=True)
                text = font_tile.render(str(value), True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

            # Обводка плитки
            pygame.draw.rect(screen, BLACK, rect, width=2, border_radius=10)

    pygame.display.flip()
    return pygame.surfarray.array3d(screen)

def main(grid_size=GRID_SIZE):
    # Создаем папку для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = NumpuzEnv(size=grid_size)
    agent = QLearningAgent(env)
    rewards_history = []
    solved_episodes = []
    gif_frames = []
    video_frames = []

    pygame.init()
    info = pygame.display.Info()
    screen_width = info.current_w
    screen_height = info.current_h
    cell_size = CELL_SIZE_WINDOW
    header_height = HEADER_HEIGHT_WINDOW
    screen = pygame.display.set_mode((env.size*cell_size, env.size*cell_size + header_height), pygame.RESIZABLE)
    pygame.display.set_caption(f"Numpuz Q-Learning {grid_size}x{grid_size}")
    clock = pygame.time.Clock()
    is_fullscreen = False

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        operation_count = 0
        episode_frames = []  # Для видео последнего эпизода

        while not done and step < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if event.key == pygame.K_f:
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            cell_size = min(screen_width // env.size, screen_height // (env.size + 1))
                            header_height = cell_size
                            screen = pygame.display.set_mode(
                                (env.size*cell_size, env.size*cell_size + header_height),
                                pygame.FULLSCREEN
                            )
                        else:
                            cell_size = CELL_SIZE_WINDOW
                            header_height = HEADER_HEIGHT_WINDOW
                            screen = pygame.display.set_mode(
                                (env.size*cell_size, env.size*cell_size + header_height),
                                pygame.RESIZABLE
                            )

            action = agent.select_action(state)
            if action:
                operation_count += 1
                next_state, reward, done = env.step(action, operation_count)
                agent.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

            frame = draw_grid(screen, env, episode + 1, operation_count, step, cell_size, header_height)
            if episode < 30:
                gif_frames.append(np.transpose(frame, (1, 0, 2)))
            if episode == EPISODES - 1:
                episode_frames.append(np.transpose(frame, (1, 0, 2)))
            clock.tick(FPS)
            step += 1

        if episode == EPISODES - 1:
            video_frames = episode_frames
        solved_episodes.append(1 if done else 0)
        rewards_history.append(total_reward)
        agent.decay_epsilon()
        print(f"Эпизод {episode+1} завершён с общей наградой {total_reward}, Epsilon={agent.epsilon:.3f}")

    # Сохранение GIF
    gif_path = os.path.join(OUTPUT_DIR, f'numpuz_training_{grid_size}x{grid_size}.gif')
    imageio.mimsave(gif_path, gif_frames, duration=0.5)

    # Сохранение видео последнего эпизода
    video_path = os.path.join(OUTPUT_DIR, f'numpuz_training_video_{grid_size}x{grid_size}.mp4')
    writer = imageio.get_writer(video_path, fps=2, macro_block_size=1)
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()

    # Создание графика
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Общая награда', color='tab:blue')
    ax1.plot(rewards_history, color='tab:blue', label='Награда')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Скользящее среднее наград', color='tab:red')
    window = 10
    moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax2.plot(range(window-1, EPISODES), moving_avg, color='tab:red', label='Среднее (окно=10)')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f'Обучение агента (Размер: {grid_size}x{grid_size})')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    graph_path = os.path.join(OUTPUT_DIR, f'numpuz_training_results_{grid_size}x{grid_size}.png')
    plt.savefig(graph_path)
    plt.close()

    pygame.quit()

if __name__ == "__main__":
    main(grid_size=3)