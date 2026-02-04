"""
Attention visualization utilities.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class AttentionVisualizer:
    """Utility for plotting attention weights."""

    def __init__(self, variable_names: Optional[List[str]] = None):
        self.variable_names = variable_names

    def plot_variable_attention(
        self,
        attention_weights: np.ndarray,
        title: str = "Variable Attention Weights",
        save_path: Optional[str] = None,
    ) -> None:
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        while attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)

        plt.figure(figsize=(10, 8))

        labels = self.variable_names if self.variable_names else [
            f"Var {i}" for i in range(attention_weights.shape[0])
        ]

        sns.heatmap(
            attention_weights,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            square=True,
        )

        plt.title(title, fontsize=14)
        plt.xlabel("Key Variable", fontsize=12)
        plt.ylabel("Query Variable", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_temporal_attention(
        self,
        attention_weights: np.ndarray,
        timestamps: Optional[List] = None,
        title: str = "Temporal Attention Weights",
        save_path: Optional[str] = None,
    ) -> None:
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        while attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            attention_weights,
            cmap="viridis",
            square=True,
        )

        if timestamps is not None:
            plt.xticks(ticks=np.arange(len(timestamps)) + 0.5, labels=timestamps, rotation=45, ha="right")
            plt.yticks(ticks=np.arange(len(timestamps)) + 0.5, labels=timestamps, rotation=0)

        plt.title(title, fontsize=14)
        plt.xlabel("Key Time Step", fontsize=12)
        plt.ylabel("Query Time Step", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_variable_importance_over_time(
        self,
        attention_weights_list: np.ndarray,
        timestamps: List,
        title: str = "Variable Importance Over Time",
    ) -> None:
        # TODO: implement time-varying variable importance visualization.
        _ = attention_weights_list, timestamps, title
        raise NotImplementedError("Variable importance over time is not implemented yet.")
