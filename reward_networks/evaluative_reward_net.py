import gymnasium as gym
import torch

from data_models.Trajectory import Trajectory
from reward_networks.base_reward_net import BaseRewardNet


class RewardNetEvaluative(BaseRewardNet):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__(observation_space, action_space)

        self.scalars = []
        self.tensor_trajectories = []

    def load_trajectories(self, trajectories: list[Trajectory], scalars):
        for traj in trajectories:
            self.tensor_trajectories.append(self.trajectory_to_tensor(traj))

        for scalar in scalars:
            self.scalars.append(torch.tensor(scalar, dtype=torch.float32))

    def train_model(self, learning_rate, epochs, loss_threshold=None, patience=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        mse_loss = torch.nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            predicted_rewards = [self.forward(trajectory_tensor).sum() for trajectory_tensor in
                                 self.tensor_trajectories]

            rewards_tensor = torch.stack(predicted_rewards)

            optimizer.zero_grad()
            losses = []

            for i in range(len(self.tensor_trajectories)):
                loss = mse_loss(rewards_tensor[i].unsqueeze(0), self.scalars[i].unsqueeze(0))
                losses.append(loss)

            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()

            #print(f'Epoch {epoch + 1}, Loss: {total_loss.item()}')

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                #print('Early stopping triggered.')
                break
