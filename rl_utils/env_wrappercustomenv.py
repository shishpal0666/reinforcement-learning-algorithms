
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class CustomEnv(gym.Env):
    def _init_(self):
        super(CustomEnv, self)._init_()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1000, 1000]), dtype=np.float32)

        # Initial position of the agent (start state)
        self.agent_pos = np.array([100, 100])
        self.goal_pos = np.array([700, 950])  # Position of the goal state

        # Obstacles as a list of rectangles (x, y, width, height)
        self.obstacles = [
            (100, 200, 200, 50),   # Obstacle 1
            (400, 300, 150, 50),   # Obstacle 2
            (600, 100, 200, 50),   # Obstacle 3
            (300, 500, 200, 100),  # Obstacle 4
            (800, 400, 150, 200),  # Obstacle 5
            (200, 700, 250, 100),  # Obstacle 6
            (700, 800, 150, 100),  # Obstacle 7
            (500, 600, 100, 150),  # Obstacle 8
            (900, 200, 50, 300),   # Obstacle 9
            (100, 900, 200, 50),   # Obstacle 10
            (500, 750, 300, 50),   # Obstacle 11
            (750, 200, 50, 300),   # Obstacle 12
            (250, 250, 50, 200),   # Obstacle 13
            (350, 100, 50, 200),   # Obstacle 14
            (150, 600, 100, 50),   # Obstacle 15
            (600, 500, 50, 100),   # Obstacle 16
            (400, 900, 150, 50),   # Obstacle 17
            (850, 750, 100, 150),  # Obstacle 18
        ]

    def reset(self):
        self.agent_pos = np.array([100, 100])  # Reset to start state
        return self.agent_pos

    def step(self, action):
        if action == 0:   # Up
            self.agent_pos[1] += 10
        elif action == 1: # Down
            self.agent_pos[1] -= 10
        elif action == 2: # Left
            self.agent_pos[0] -= 10
        elif action == 3: # Right
            self.agent_pos[0] += 10

        # Clip to stay within bounds
        self.agent_pos = np.clip(self.agent_pos, [0, 0], [1000, 1000])

        # Check if agent has reached the goal
        if np.linalg.norm(self.agent_pos - self.goal_pos) < 10:
            reward = 1  # Positive reward for reaching the goal
            done = True
        else:
            reward = -0.01  # Small penalty to encourage faster completion
            done = False

        # Check if agent hits an obstacle
        for obstacle in self.obstacles:
            if self._check_collision(obstacle):
                reward = -1  # Negative reward for hitting an obstacle
                done = True
                break

        return self.agent_pos, reward, done, {}

    def _check_collision(self, obstacle):
        ox, oy, w, h = obstacle
        return (ox <= self.agent_pos[0] <= ox + w) and (oy <= self.agent_pos[1] <= oy + h)

    def render(self, mode='human'):
        plt.figure(figsize=(10, 10))
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)

        # Draw the boundary
        plt.plot([0, 1000, 1000, 0, 0], [0, 0, 1000, 1000, 0], color='blue')

        # Draw the start state
        plt.scatter(self.agent_pos[0], self.agent_pos[1], color='green', s=100, label='Start')

        # Draw the goal state
        plt.scatter(self.goal_pos[0], self.goal_pos[1], color='red', s=100, label='Goal')

        # Draw obstacles
        for obstacle in self.obstacles:
            ox, oy, w, h = obstacle
            plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color='black'))

        plt.legend()
        plt.show()

# Create environment
env = CustomEnv()

# Test the environment
obs = env.reset()
done = False
# while not done:
action = env.action_space.sample()  # Random action
obs, reward, done, info = env.step(action)
env.render()
