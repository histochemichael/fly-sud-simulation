from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"untrained": "#6B7280", "paired": "#B45309", "unpaired": "#2563EB"}
LABELS = {"untrained": "Untrained", "paired": "Paired", "unpaired": "Unpaired"}


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140, "savefig.dpi": 180})


def generate_figures(results: pd.DataFrame, trajectories: pd.DataFrame, memory_history: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, order = [], ["untrained", "paired", "unpaired"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    for ax, condition in zip(axes, order):
        subset = trajectories[trajectories["condition"] == condition]
        for fly_id, path_data in subset.groupby("fly_id"):
            if fly_id >= 8:
                break
            ax.plot(path_data["x"], path_data["y"], alpha=0.55, lw=1)
        ax.plot([0, 0, -1.15], [-1.25, 0, 1.25], color="black", lw=1.4)
        ax.plot([0, 1.15], [0, 1.25], color="black", lw=1.4)
        ax.scatter([-1.15, 1.15], [1.25, 1.25], s=35, c=["#7C3AED", "#10B981"])
        ax.set_title(LABELS[condition])
        ax.set_aspect("equal")
        ax.set_xlabel("x (arena units)")
    axes[0].set_ylabel("y (arena units)")
    fig.suptitle("Figure 1. Reward-free Y-maze trajectories (examples)")
    fig.tight_layout()
    path = output_dir / "figure1_ymaze_trajectories.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    s = summary.set_index("condition").loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(order))
    errors = np.vstack([s["preference_index"] - s["pi_ci95_low"], s["pi_ci95_high"] - s["preference_index"]])
    ax.bar(x, s["preference_index"], color=[COLORS[c] for c in order], yerr=errors, capsize=5)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x, [LABELS[c] for c in order])
    ax.set_ylabel("Preference index for odor B")
    ax.set_ylim(-1, 1)
    ax.set_title("Figure 2. Conditioned preference with 95% normal CI")
    fig.tight_layout()
    path = output_dir / "figure2_preference_index.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    counts = results.groupby(["condition", "choice"]).size().unstack(fill_value=0).reindex(order)
    for key in ["A", "B"]:
        if key not in counts:
            counts[key] = 0
    proportions = counts[["A", "B"]].div(counts[["A", "B"]].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x, proportions["A"], color="#10B981", label="Odor A")
    ax.bar(x, proportions["B"], bottom=proportions["A"], color="#7C3AED", label="Odor B")
    ax.set_xticks(x, [LABELS[c] for c in order])
    ax.set_ylabel("Fraction of choices")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Figure 3. Choice distribution")
    fig.tight_layout()
    path = output_dir / "figure3_choice_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    mean_history = memory_history.groupby(["condition", "session"])[["weight_a", "weight_b"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, condition in zip(axes, order):
        frame = mean_history.loc[condition]
        ax.plot(frame.index, frame["weight_a"], marker="o", label="Odor A")
        ax.plot(frame.index, frame["weight_b"], marker="o", label="Odor B")
        ax.set_title(LABELS[condition])
        ax.set_xlabel("Conditioning session")
        ax.set_xticks(frame.index)
    axes[0].set_ylabel("Associative weight")
    axes[-1].legend(frameon=False)
    fig.suptitle("Figure 4. Reward-modulated memory across training")
    fig.tight_layout()
    path = output_dir / "figure4_memory_weights.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths

