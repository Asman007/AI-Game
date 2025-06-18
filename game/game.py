
import pygame
import numpy as np
import random
from matplotlib import pyplot as plt
import imageio
import os

# Конфигурация
MAP = [
    "W....W",
    "W.T..W",
    "..O.T.",
    "W..O.W",
    "W....W"
]

EPISODES = 20
CELL_SIZE = 80
FPS = 10
ALPHA = 0.3
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.985
MAX_STEPS = 100
HEADER_HEIGHT = 50  # Пространство для текста сверху

# Цвета
GRAY = (192, 192, 192)
WHITE = (255, 255, 255)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

class CleanupEnv:
    def __init__(self):
        self.rows = len(MAP)
        self.cols = len(MAP[0])
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.actions = ['up', 'down', 'left', 'right']
        self._parse_map()
        
    def _parse_map(self):
        self.trash = []
        self.obstacles = []
        self.walls = []
        for i, row in enumerate(MAP):
            for j, ch in enumerate(row):
                if ch == 'T':
                    self.trash.append((i, j))
                elif ch == 'O':
                    self.obstacles.append((i, j))
                elif ch == 'W':
                    self.walls.append((i, j))

    def reset(self):
        self.agent_pos = random.choice([s for s in self.states if s not in self.obstacles and s not in self.walls])
        self.collected = set()
        self.game_over = False
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        new_pos = {
            'up': (x-1, y),
            'down': (x+1, y),
            'left': (x, y-1),
            'right': (x, y+1)
        }[action]

        reward = -1
        done = False
        
        if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
            return self.agent_pos, reward, done
        
        if new_pos in self.obstacles or new_pos in self.walls:
            reward = -20
            return self.agent_pos, reward, done
        
        self.agent_pos = new_pos
        
        if new_pos in self.trash and new_pos not in self.collected:
            reward = 20
            self.collected.add(new_pos)
            if len(self.collected) == len(self.trash):
                reward = 50
                done = True
        
        return self.agent_pos, reward, done

class QLearningAgent:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.Q = np.ones((len(states), len(actions))) * 0.1
        self.epsilon = EPSILON_START
        
    def select_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        return np.argmax(self.Q[state_idx])

    def update(self, s_idx, a_idx, reward, next_s_idx, done):
        target = reward + GAMMA * np.max(self.Q[next_s_idx]) * (not done)
        self.Q[s_idx, a_idx] += ALPHA * (target - self.Q[s_idx, a_idx])
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

def draw_grid(screen, env, agent_pos, episode, operation_count, step):
    screen.fill(WHITE)
    # Отрисовка текста в заголовке
    font = pygame.font.SysFont('arial', 24)
    text = f"Эпизод: {episode}  Операция: {operation_count}  Шаг: {step}"
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(env.cols*CELL_SIZE//2, HEADER_HEIGHT//2))
    screen.blit(text_surface, text_rect)
    
    # Отрисовка сетки
    for i in range(env.rows):
        for j in range(env.cols):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE + HEADER_HEIGHT, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (i, j) in env.walls:
                color = BROWN
            elif (i, j) in env.obstacles:
                color = PURPLE
            elif (i, j) in env.trash:
                color = GREEN if (i, j) in env.collected else GRAY
            pygame.draw.rect(screen, color, rect)
            if (i, j) == agent_pos:
                pygame.draw.circle(screen, YELLOW, rect.center, CELL_SIZE//3)
            pygame.draw.rect(screen, BLACK, rect, 1)
    pygame.display.flip()
    return pygame.surfarray.array3d(screen)

def main():
    env = CleanupEnv()
    agent = QLearningAgent(env.states, env.actions)
    rewards_history = []
    frames = []

    pygame.init()
    screen = pygame.display.set_mode((env.cols*CELL_SIZE, env.rows*CELL_SIZE + HEADER_HEIGHT))
    pygame.display.set_caption("Cleanup Q-Learning")
    clock = pygame.time.Clock()

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        operation_count = 0  # Счетчик операций начинается с 0
        
        while not done and step < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            s_idx = env.states.index(state)
            operation_count += 1  # Увеличиваем перед выбором действия
            a_idx = agent.select_action(s_idx)
            action = env.actions[a_idx]
            
            next_state, reward, done = env.step(action)
            next_s_idx = env.states.index(next_state)
            agent.update(s_idx, a_idx, reward, next_s_idx, done)
            
            print(f"Эпизод {episode+1}, Шаг {step}, Операция {operation_count}: Состояние={state}, Действие={action}, Награда={reward}")
            
            frame = draw_grid(screen, env, state, episode+1, operation_count, step)
            if episode < 30:
                frames.append(np.transpose(frame, (1, 0, 2)))
            clock.tick(FPS)
            
            total_reward += reward
            state = next_state
            step += 1
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        print(f"Эпизод {episode+1} завершён с общей наградой {total_reward}, Epsilon={agent.epsilon:.3f}\n")

    # Сохранение GIF
    imageio.mimsave('training_process.gif', frames, duration=0.5)
    
    pygame.quit()
    
    # Визуализация обучения
    plt.plot(rewards_history)
    plt.title("Обучение агента (Макс. шагов: 100)")
    plt.xlabel("Эпизод")
    plt.ylabel("Общая награда")
    plt.savefig('training_results.png')

if __name__ == "__main__":
    main()
