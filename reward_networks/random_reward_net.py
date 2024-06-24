from reward_networks.base_reward_net import BaseRewardNet
import gymnasium as gym
import torch


class RandomRewardNet(BaseRewardNet):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__(observation_space, action_space)
        self._initialize_random_weights()

    def _initialize_random_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def train_model(self, learning_rate, epochs, loss_threshold=None):
        pass
