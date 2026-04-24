from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cads.config import AppConfig
from cads.models.evaluate import (
    binary_attack_metrics,
    multiclass_metrics,
    save_binary_confusion_matrix_plot,
    save_confusion_matrix_plot,
)

LOGGER = logging.getLogger(__name__)


def _read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(path)


def _build_models(config: AppConfig) -> dict[str, object]:
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=config.model.random_seed,
                    class_weight=config.model.class_weight,
                    n_jobs=None,
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=config.model.random_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    isolation = IsolationForest(
        n_estimators=config.model.iforest_estimators,
        contamination=config.model.iforest_contamination,
        random_state=config.model.random_seed,
        n_jobs=-1,
    )
    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
        "isolation_forest": isolation,
    }


def _attack_binary(y: np.ndarray, benign_index: int) -> np.ndarray:
    return (y != benign_index).astype(int)


def train_and_evaluate(config: AppConfig) -> dict[str, object]:
    train_df = _read_split(config.paths.data_processed / "train.csv")
    val_df = _read_split(config.paths.data_processed / "val.csv")
    test_df = _read_split(config.paths.data_processed / "test.csv")

    feature_cols = list(config.data.feature_columns)
    for col in feature_cols:
        if col not in train_df.columns:
            raise KeyError(f"Missing feature column '{col}'. Run `prepare-data` first.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"].astype(str))
    y_val = label_encoder.transform(val_df["label"].astype(str))
    y_test = label_encoder.transform(test_df["label"].astype(str))
    labels = label_encoder.classes_.tolist()

    X_train = train_df[feature_cols].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    models = _build_models(config)
    model_metrics: dict[str, dict[str, float]] = {}

    # Supervised models
    for name in ("logistic_regression", "random_forest"):
        model = models[name]
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = multiclass_metrics(y_test, y_pred_test, y_proba_test)

        try:
            benign_index = labels.index("benign")
        except ValueError:
            benign_index = 0

        y_true_attack = _attack_binary(y_test, benign_index)
        y_pred_attack = _attack_binary(y_pred_test, benign_index)
        metrics.update(binary_attack_metrics(y_true_attack, y_pred_attack))
        model_metrics[name] = metrics

        model_path = config.paths.artifacts_models / f"{name}.joblib"
        joblib.dump(model, model_path)
        save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=y_pred_test,
            labels=labels,
            out_path=config.paths.artifacts_plots / f"confusion_matrix_{name}.png",
            title=f"Confusion Matrix - {name}",
        )

    # Isolation forest (unsupervised): fit on benign-only train if available.
    iforest: IsolationForest = models["isolation_forest"]  # type: ignore[assignment]
    benign_mask = train_df["label"].astype(str).str.lower().isin(config.data.benign_labels)
    X_iforest_train = train_df.loc[benign_mask, feature_cols].to_numpy()
    if len(X_iforest_train) < 100:
        X_iforest_train = X_train
    iforest.fit(X_iforest_train)

    y_pred_iforest = iforest.predict(X_test)
    y_pred_attack_iforest = (y_pred_iforest == -1).astype(int)

    try:
        benign_index = labels.index("benign")
    except ValueError:
        benign_index = 0
    y_true_attack_iforest = _attack_binary(y_test, benign_index)

    iforest_metrics = binary_attack_metrics(y_true_attack_iforest, y_pred_attack_iforest)
    model_metrics["isolation_forest"] = iforest_metrics

    save_binary_confusion_matrix_plot(
        y_true_attack_iforest,
        y_pred_attack_iforest,
        out_path=config.paths.artifacts_plots / "confusion_matrix_isolation_forest_binary.png",
        title="Confusion Matrix - Isolation Forest (Attack vs Benign)",
    )

    joblib.dump(iforest, config.paths.artifacts_models / "isolation_forest.joblib")
    joblib.dump(label_encoder, config.paths.artifacts_models / "label_encoder.joblib")

    best_supervised = max(
        ("logistic_regression", "random_forest"),
        key=lambda model_name: model_metrics[model_name].get("f1_weighted", 0.0),
    )

    output = {
        "trained_models": list(model_metrics.keys()),
        "best_supervised_model": best_supervised,
        "metrics": model_metrics,
        "feature_columns": feature_cols,
        "labels": labels,
        "data_shapes": {
            "train": [int(X_train.shape[0]), int(X_train.shape[1])],
            "val": [int(X_val.shape[0]), int(X_val.shape[1])],
            "test": [int(X_test.shape[0]), int(X_test.shape[1])],
        },
    }

    metrics_path = config.paths.artifacts_reports / "model_metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2))

    summary_md = [
        "# Model Evaluation Summary",
        "",
        f"- Best supervised model: `{best_supervised}`",
        f"- Labels: `{', '.join(labels)}`",
        "",
        "## Metrics",
    ]
    for model_name, metrics in model_metrics.items():
        summary_md.append("")
        summary_md.append(f"### {model_name}")
        for key, value in metrics.items():
            summary_md.append(f"- {key}: `{value:.4f}`")
    summary_path = config.paths.artifacts_reports / "model_summary.md"
    summary_path.write_text("\n".join(summary_md))

    LOGGER.info("Training complete. Metrics: %s", metrics_path)
    return {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "best_supervised_model": best_supervised,
    }

