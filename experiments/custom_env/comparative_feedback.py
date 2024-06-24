from tqdm import tqdm

from data_gathering.trajectory_feedback_utils import generate_preferences
from environments.custom_grid_world_env import MazeGameEnv
from environments.reward_wrapper import get_wrapper_environment
from reward_networks.preference_reward_net import RewardNetPreference
from utils.q_learning import QLearning
from utils.seed import set_seeds
import pandas as pd

rewards_combined = []

for seed in range(5):
    base_env = get_wrapper_environment(MazeGameEnv(), episode_length=10)
    set_seeds(seed)
    base_env.action_space.seed(seed)

    reward_network = RewardNetPreference(base_env.observation_space, base_env.action_space, margin=10)

    env = get_wrapper_environment(base_env, reward_network)

    model = QLearning(env)

    comparison = 200

    curr_seed_rewards = []
    for i in tqdm(range(comparison)):
        curr_seed_rewards.append(model.rollout(base_env, eps=0.0).total_reward)
        model.train(25)
        traj = [model.rollout(base_env, eps=0.0), model.rollout(base_env, eps=0.1), model.rollout(base_env, eps=0.2),
                model.rollout(base_env, eps=0.3)]
        preference = generate_preferences(traj)
        if preference:
            reward_network.add_preference(*preference[0])

        reward_network.train_model(learning_rate=0.005, epochs=30)

    rewards_combined.append(curr_seed_rewards)


rewards = []
for i in range(len(rewards_combined[0])):
    reward = 0
    for reward_arr in rewards_combined:
        reward += reward_arr[i]

    reward /= len(rewards_combined)
    rewards.append(reward)

df = pd.DataFrame(rewards)
df.to_csv("results/preference_margin_fixed.csv", index=False)

