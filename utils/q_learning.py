import random

import gymnasium as gym
import numpy as np

from data_models.Trajectory import Trajectory


class QLearning:
    def __init__(self, env: gym.Env, alpha=0.1, gamma=0.6, epsilon=0.3):
        self.env = env
        # Create a Q-table with dimensions based on the number of states and actions
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        self.epsilon = epsilon

    def train(self, num_episodes):
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state, :])

                # Execute the chosen action in the environment
                new_state, reward, done, _, _ = self.env.step(action)

                # Update the Q-table using the learning rule
                self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])

                # Move to the new state
                state = new_state

    def predict(self, state):
        return np.argmax(self.Q[state])

    def clear(self):
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def rollout(self, env, eps=0.0) -> Trajectory:
        trajectory: Trajectory = Trajectory()

        state, _ = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(self.Q[state, :])

            # Execute the chosen action in the environment
            new_state, reward, done, _, _ = env.step(action)

            trajectory.add(state, action, reward)

            # Move to the new state
            state = new_state

        return trajectory

    def evaluate(self, env, n_steps):
        rewards = []
        for _ in range(n_steps):
            rewards.append(self.rollout(env).total_reward)

        return np.mean(rewards), np.std(rewards)

