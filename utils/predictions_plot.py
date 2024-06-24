import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from reward_networks.base_reward_net import BaseRewardNet


def plot_predictions(predictions, filename, palette="Blues"):
    map_size = 4
    """Plot the last frame of the simulation and the policy learned."""
    predictions_max = predictions.max(axis=1).reshape(map_size, map_size)
    predictions_normalized = (predictions_max - np.min(predictions_max)) / (
            np.max(predictions_max) - np.min(predictions_max))

    qtable_val_max_rounded = np.round(predictions_max).astype(int)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # Plot the policy
    sns.heatmap(
        predictions_normalized,
        fmt="",
        ax=ax,
        annot=qtable_val_max_rounded,
        cmap=sns.color_palette(palette, as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
        cbar=False
    )
    fig.savefig(filename, bbox_inches="tight")
