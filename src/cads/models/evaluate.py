from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    metrics: dict[str, float] = {
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 2:
        try:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            pass
    return metrics


def binary_attack_metrics(
    y_true_attack_binary: np.ndarray,
    y_pred_attack_binary: np.ndarray,
) -> dict[str, float]:
    return {
        "precision_attack_binary": float(
            precision_score(y_true_attack_binary, y_pred_attack_binary, zero_division=0)
        ),
        "recall_attack_binary": float(recall_score(y_true_attack_binary, y_pred_attack_binary, zero_division=0)),
        "f1_attack_binary": float(f1_score(y_true_attack_binary, y_pred_attack_binary, zero_division=0)),
    }


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_binary_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", cbar=False, ax=ax, xticklabels=["benign", "attack"], yticklabels=["benign", "attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

