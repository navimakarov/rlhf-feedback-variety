import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import _trajectory_pair_includes_reward
from imitation.data import rollout
from imitation.data.types import TrajectoryPair, TrajectoryWithRew
from imitation.rewards import reward_nets
from typing import Tuple, Sequence, Optional, cast


class RatingPreferenceModel(preference_comparisons.PreferenceModel):
    def __init__(self, model: reward_nets.RewardNet, reward_diff_significance_thresholds, m_mul: float = 1.0,
                 noise_prob: float = 0.0, discount_factor: float = 1.0,
                 threshold: float = 50) -> None:
        super().__init__(model, noise_prob, discount_factor, threshold)

        self.m_mul = m_mul

        self.reward_diff_significance_thresholds = np.array(reward_diff_significance_thresholds)

    def get_significance_score(self, score: th.Tensor) -> th.Tensor:
        # Find the position where the reward would be inserted to maintain order
        score_np = th.abs(score).numpy()
        pos = np.searchsorted(self.reward_diff_significance_thresholds, score_np, side='right')

        # Calculate the rating as a float
        rating = pos / len(self.reward_diff_significance_thresholds)

        if score < 0:
            return -th.tensor(rating, dtype=th.float32)

        return th.tensor(rating, dtype=th.float32)


    def forward(
            self,
            fragment_pairs: Sequence[TrajectoryPair],
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        if gt_reward_available:
            for i, fragment in enumerate(fragment_pairs):
                frag1, frag2 = fragment
                frag1 = cast(TrajectoryWithRew, frag1)
                frag2 = cast(TrajectoryWithRew, frag2)


                true_rews1 = th.from_numpy(frag1.rews).sum()
                true_rews2 = th.from_numpy(frag2.rews).sum()


                trans1 = rollout.flatten_trajectories([frag1])
                trans2 = rollout.flatten_trajectories([frag2])
                predicted_rews1 = self.rewards(trans1)
                predicted_rews2 = self.rewards(trans2)

                m = self.get_significance_score(true_rews1 - true_rews2)
                probs[i] = th.sigmoid((predicted_rews1 - predicted_rews2).sum(axis=0) - self.m_mul * m)

        return probs, None

