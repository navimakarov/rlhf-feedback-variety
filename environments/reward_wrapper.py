from typing import Tuple

import types
import gymnasium
import numpy as np
import torch
from gymnasium.core import ActType, ObsType
from typing import Callable

from reward_networks.base_reward_net import BaseRewardNet


class RewardEnvWrapper(gymnasium.Env):
    def __init__(self, env: gymnasium.Env, reward_net: BaseRewardNet | None,
                 reward_function, episode_length: int | None, seed):
        self.env = env
        self.reward_net = reward_net
        self.reward_function = reward_function
        self.state = None

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.episode_length = episode_length

        self.step_count = 0
        self.seed = seed

        self.absorb = False

    def reset(self, seed=None, options=None, **kwargs):
        self.step_count = 0
        self.absorb = False
        if isinstance(self.seed, types.GeneratorType):
            self.state, info = self.env.reset(seed=next(self.seed), **kwargs)
        else:
            self.state, info = self.env.reset(seed=self.seed, **kwargs)
        return self.state, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def step(self, action: ActType, **kwargs) -> Tuple[ObsType, float, bool, bool, dict]:
        if self.state is None:
            self.reset()

        if self.absorb:
            self.step_count += 1
            obs = self.state
            if self.episode_length is not None:
                done = self.step_count == self.episode_length
            else:
                done = False
            reward = 0
            return obs, reward, done, False, {}

        prev_state = self.state

        obs, reward, done, truncated, info = self.env.step(action, **kwargs)

        if done or truncated:
            self.absorb = True

        if self.reward_net is not None:
            state_action_tensor = self.reward_net.state_action_tensor(prev_state, action)
            reward = self.reward_net.forward(state_action_tensor).detach().numpy()
        elif self.reward_function is not None:
            reward = self.reward_function(prev_state, action, obs, reward)

        self.state = obs

        self.step_count += 1

        if self.episode_length is not None:
            done = self.step_count == self.episode_length

        return obs, reward, done, False, info


def get_wrapper_environment(env: gymnasium.Env, reward_net: BaseRewardNet | None = None, reward_function=None,
                            episode_length: int | None = None, seed=None):
    return RewardEnvWrapper(env, reward_net, reward_function, episode_length, seed)
