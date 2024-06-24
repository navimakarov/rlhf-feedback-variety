from typing import Sequence

import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.data.types import (
    TrajectoryPair,
)
import random


class MSELoss(preference_comparisons.RewardLoss):

    def __init__(self) -> None:
        super().__init__()
        self.accuracies = []

        self.offset = [random.choice([-3, -2, -1, 0, 1, 2, 3]) for _ in range(1000)]

    def forward(
            self,
            fragment_pairs: Sequence[TrajectoryPair],
            preferences: np.ndarray,
            preference_model: preference_comparisons.PreferenceModel,
    ) -> preference_comparisons.LossAndMetrics:
        rews_pred = th.empty(len(fragment_pairs), dtype=th.float32)
        rews_true = th.empty(len(fragment_pairs), dtype=th.float32)

        for idx, fragment in enumerate(fragment_pairs):
            frag1, _ = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            rews_pred[idx] = preference_model.rewards(trans1).sum()

            #rew_modified = max(-32, min((frag1.rews.sum() // 100) + self.offset[idx], 0))
            rew_modified = frag1.rews.sum()
            rews_true[idx] = th.tensor(rew_modified, dtype=th.float32)

        loss = th.nn.MSELoss()(rews_pred, rews_true)

        metrics = {}
        metrics["accuracy"] = th.tensor(0)
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}
        return preference_comparisons.LossAndMetrics(
            loss=loss,
            metrics=metrics,
        )
