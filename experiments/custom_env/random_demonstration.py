from tqdm import tqdm

from data_gathering.trajectory_feedback_utils import generate_preferences
from environments.custom_grid_world_env import MazeGameEnv
from environments.reward_wrapper import get_wrapper_environment
from reward_networks.random_reward_net import RandomRewardNet
from utils.q_learning import QLearning
from utils.seed import set_seeds
import pandas as pd

rewards_combined = []

for seed in range(5):
    base_env = get_wrapper_environment(MazeGameEnv(), episode_length=10)
    set_seeds(seed)
    base_env.action_space.seed(seed)

    reward_network = RandomRewardNet(base_env.observation_space, base_env.action_space)

    env = get_wrapper_environment(base_env, reward_network)

    model = QLearning(env)

    iterations = 200

    curr_seed_rewards = []
    for i in tqdm(range(iterations)):
        curr_seed_rewards.append(model.rollout(base_env, eps=0.0).total_reward)
        model.train(25)

    rewards_combined.append(curr_seed_rewards)


rewards = []
for i in range(len(rewards_combined[0])):
    reward = 0
    for reward_arr in rewards_combined:
        reward += reward_arr[i]

    reward /= len(rewards_combined)
    rewards.append(reward)

df = pd.DataFrame(rewards)
df.to_csv("results/random.csv", index=False)

