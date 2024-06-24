import abc
import gymnasium as gym
from typing import cast
from imitation.util import util
from stable_baselines3.common import preprocessing
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from data_models.Trajectory import Trajectory


class BaseRewardNet(nn.Module, abc.ABC):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        combined_size = (preprocessing.get_flattened_obs_dim(self.observation_space) +
                         preprocessing.get_flattened_obs_dim(self.action_space))

        self.fc1 = nn.Linear(combined_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward.squeeze()

    def state_action_tensor(self, state, action) -> torch.Tensor:
        state_th = self.state_to_tensor(state)
        action_th = self.action_to_tensor(action)

        return torch.cat([state_th.flatten(), action_th.flatten()])

    def state_to_tensor(self, state) -> torch.Tensor:
        state_th = util.safe_to_tensor(np.array(state, dtype=self.observation_space.dtype))
        state_th = cast(
            torch.Tensor,
            preprocessing.preprocess_obs(
                state_th,
                self.observation_space,
                False,
            ),
        )

        return state_th

    def action_to_tensor(self, action) -> torch.Tensor:
        action_th = util.safe_to_tensor(np.array(action, dtype=self.action_space.dtype))
        action_th = cast(
            torch.Tensor,
            preprocessing.preprocess_obs(
                action_th,
                self.action_space,
                False,
            ),
        )

        return action_th

    def trajectory_to_tensor(self, trajectory: Trajectory):
        states_actions = []
        for state, action in trajectory.trajectory:
            states_actions.append(self.state_action_tensor(state, action))

        tensor_trajectory = torch.stack(states_actions)
        return tensor_trajectory

    @abc.abstractmethod
    def train_model(self, learning_rate, epochs, loss_threshold=None):
        pass
