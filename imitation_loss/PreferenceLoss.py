from typing import Sequence

import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import LossAndMetrics, PreferenceModel, RewardLoss
from imitation.data import rollout
from imitation.data.types import (
    TrajectoryPair,
)

class PreferenceLoss(RewardLoss):
    """Compute the cross entropy reward loss."""

    def __init__(self) -> None:
        """Create cross entropy reward loss."""
        super().__init__()
        self.accuracies = []

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preference_model: PreferenceModel,
    ) -> LossAndMetrics:
        """Computes the loss.

        Args:
            fragment_pairs: Batch consisting of pairs of trajectory fragments.
            preferences: The probability that the first fragment is preferred
                over the second. Typically 0, 1 or 0.5 (tie).
            preference_model: model to predict the preferred fragment from a pair.

        Returns:
            The cross-entropy loss between the probability predicted by the
                reward model and the target probabilities in `preferences`. Metrics
                are accuracy, and gt_reward_loss, if the ground truth reward is
                available.
        """
        probs, gt_probs = preference_model(fragment_pairs)
        predictions = probs > 0.5
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = preferences_th > 0.5
        metrics = {}
        metrics["accuracy"] = (predictions == ground_truth).float().mean()
        if gt_probs is not None:
            metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
                gt_probs,
                preferences_th,
            )
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}

        self.accuracies.append(metrics["accuracy"])

        return LossAndMetrics(
            #loss=th.sum(-th.log(probs + 1e-6)),
            loss=th.nn.functional.binary_cross_entropy(probs, preferences_th),
            metrics=metrics,
        )