import torch
import numpy as np
import random
from data_models.Trajectory import Trajectory


def get_scalar_values(trajectories: list[Trajectory]) -> list[float]:
    return [float(traj.total_reward) for traj in trajectories]

def generate_preferences(trajectories: list[Trajectory],
                         tie_breaker: str | None = None,
                         threshold: int = 0,
                         cnt: int | None = None) -> list[(Trajectory, Trajectory, int, int)]:
    preferences = []
    for i in range(len(trajectories) - 1):
        for j in range(i + 1, len(trajectories)):
            u1, u2 = get_preference_values(trajectories[i], trajectories[j], tie_breaker, threshold)
            if u1 != u2:
                preferences.append((trajectories[i], trajectories[j], u1, u2))

            if cnt is not None and len(preferences) >= cnt:
                return preferences

    return preferences

def sample_preference_pairs(trajectories: list[Trajectory],
                              cnt: int,
                              tie_breaker: str | None = None,
                              threshold: int = 0) -> list[(int, int, int, int)]:
    preferences = []
    while len(preferences) < cnt:
        i = random.sample(range(len(trajectories)), 1)[0]
        j = random.sample(range(len(trajectories)), 1)[0]

        u1, u2 = get_preference_values(trajectories[i], trajectories[j], tie_breaker, threshold)
        if u1 != u2:
            preferences.append((i, j, u1, u2))

    return preferences

def get_preference_values(traj1: Trajectory, traj2: Trajectory, tie_breaker: str | None, threshold : int = 0) -> (
int, int):
    if abs(traj1.total_reward - traj2.total_reward) < threshold:
        return torch.tensor(0.5, dtype=torch.float32), torch.tensor(0.5, dtype=torch.float32)

    if traj1.total_reward == traj2.total_reward:
        if tie_breaker is not None:
            if tie_breaker == "shorter_trajectory":
                if len(traj1.trajectory) == len(traj2.trajectory):
                    u1 = 0.5
                    u2 = 0.5
                else:
                    u1 = 1 if len(traj1.trajectory) < len(traj2.trajectory) else 0
                    u2 = 1 - u1
        else:
            u1 = 0.5
            u2 = 0.5
    else:
        u1 = 1 if traj1.total_reward > traj2.total_reward else 0
        u2 = 1 - u1

    return u1, u2


def get_ratings_uniform(trajectories: list[Trajectory], min_reward, max_reward, n):
    bins = np.linspace(min_reward, max_reward, n + 1)
    ratings = []

    for traj in trajectories:
        rating = np.digitize(traj.total_reward, bins, right=False) - 1
        rating = min(rating, n - 1)

        ratings.append(rating)

    return ratings


def get_ratings_custom(trajectories: list[Trajectory], reward_range_ratings: list[(float, float)]):
    ratings = []
    for traj in trajectories:
        for rating, reward_range in enumerate(reward_range_ratings):
            low, high = reward_range
            if low <= traj.total_reward <= high:
                ratings.append(rating)
                break

    return ratings
