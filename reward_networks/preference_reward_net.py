from reward_networks.base_reward_net import BaseRewardNet
import torch
import gymnasium as gym
import numpy as np


class RewardNetPreference(BaseRewardNet):
    epsilon = 1e-8

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, margin=0):
        super().__init__(observation_space, action_space)

        self.preferences = []
        self.trajectory_tensors = []
        self.trajectory_tensor_index_map = {}

        self.reward_diff_significance_thresholds = np.array([10, 20, 30, 40])
        self.margin = margin

    def add_preference(self, traj1, traj2, u1, u2):
        self.preferences.append(
            (traj1, traj2, torch.tensor(u1, dtype=torch.float32), torch.tensor(u2, dtype=torch.float32)))

        if traj1 not in self.trajectory_tensor_index_map:
            tensor = self.trajectory_to_tensor(traj1)
            self.trajectory_tensor_index_map[traj1] = len(self.trajectory_tensors)
            self.trajectory_tensors.append(tensor)

        if traj2 not in self.trajectory_tensor_index_map:
            tensor = self.trajectory_to_tensor(traj2)
            self.trajectory_tensor_index_map[traj2] = len(self.trajectory_tensors)
            self.trajectory_tensors.append(tensor)

    def train_model(self, learning_rate, epochs, loss_threshold=None, patience=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-05)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            predicted_rewards = [self.forward(trajectory_tensor).sum() for trajectory_tensor in
                                 self.trajectory_tensors]

            optimizer.zero_grad()
            losses = []

            for traj1, traj2, u1, u2 in self.preferences:
                preference_tensors = (predicted_rewards[self.trajectory_tensor_index_map[traj1]],
                                      predicted_rewards[self.trajectory_tensor_index_map[traj2]],
                                      u1,
                                      u2,
                                      traj1.total_reward - traj2.total_reward)
                loss = self.preference_loss(*preference_tensors)
                losses.append(loss)

            if len(losses) == 0:
                break

            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()

            # print(f'Epoch {epoch + 1}, Loss: {total_loss.item()}')

            if loss_threshold is not None and total_loss.item() < loss_threshold:
                break

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # print('Early stopping triggered.')
                break

    def get_significance_score(self, score: torch.Tensor) -> torch.Tensor:
        # Find the position where the reward would be inserted to maintain order
        score_np = torch.abs(score).numpy()
        pos = np.searchsorted(self.reward_diff_significance_thresholds, score_np, side='right')

        # Calculate the rating as a float
        rating = pos / len(self.reward_diff_significance_thresholds)

        if score < 0:
            return -torch.tensor(rating, dtype=torch.float32)

        return torch.tensor(rating, dtype=torch.float32)

    def preference_loss(self, predicted_reward_traj1: torch.Tensor, predicted_reward_traj2: torch.Tensor,
                        u1: torch.Tensor, u2: torch.Tensor, diff) -> torch.Tensor:

        # diff bands -> [10, 20, 30, 40, 50]

        p1 = torch.sigmoid(predicted_reward_traj1 - predicted_reward_traj2 - self.get_significance_score(torch.tensor(diff, dtype=torch.float32) * self.margin))
        p2 = torch.sigmoid(predicted_reward_traj2 - predicted_reward_traj1 - self.get_significance_score(torch.tensor(-diff, dtype=torch.float32) * self.margin))

        return (u1 * -torch.log(p1 + RewardNetPreference.epsilon)) + (u2 * -torch.log(p2 + RewardNetPreference.epsilon))
