import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_q_values_map(qtable, filename):
    map_size = 4
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max = qtable.reshape(map_size, map_size)
    qtable_val_max_normalized = (qtable_val_max - np.min(qtable_val_max)) / (
                np.max(qtable_val_max) - np.min(qtable_val_max))

    qtable_val_max_rounded = np.round(qtable_val_max).astype(int)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # Plot the policy
    sns.heatmap(
        qtable_val_max_normalized,
        fmt="",
        ax=ax,
        annot=qtable_val_max_rounded,
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
        cbar=False
    )
    fig.savefig(filename, bbox_inches="tight")


rewards = [-6, -5, -4, -3, -5, -4, -3, -2, -4, -3, -2, -1, -3, -2, -1, 0]
plot_q_values_map(np.array(rewards), "rewards.png")