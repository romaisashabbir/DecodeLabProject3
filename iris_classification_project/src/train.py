"""Train and evaluate an Iris KNN classification pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load the Iris dataset and return features, target, feature names and class names."""
    iris = load_iris(as_frame=True)

    X = iris.data.copy()
    y = iris.target.copy()

    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    return X, y, feature_names, class_names


def build_pipeline() -> Pipeline:
    """Create the leakage-safe preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier()),
        ]
    )


def train() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    X, y, feature_names, class_names = load_dataset()

    # Save a human-readable copy of the dataset.
    iris_for_csv = X.copy()
    iris_for_csv["target"] = y
    iris_for_csv["target_name"] = y.map(dict(enumerate(class_names)))
    iris_for_csv.to_csv(DATA_DIR / "iris.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()

    # Tune k using training data only. The test set remains untouched.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__n_neighbors": list(range(1, 16, 2))},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = {
        "dataset": {
            "name": "Iris",
            "samples": int(len(X)),
            "features": int(X.shape[1]),
            "classes": class_names,
            "feature_names": feature_names,
        },
        "split": {
            "test_size": TEST_SIZE,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "model": {
            "algorithm": "K-Nearest Neighbors",
            "best_k": int(search.best_params_["classifier__n_neighbors"]),
            "cv_folds": 5,
            "cv_scoring": "f1_macro",
            "best_cv_f1_macro": float(search.best_score_),
        },
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    model_path = MODEL_DIR / "iris_knn_pipeline.joblib"
    report_path = REPORT_DIR / "metrics.json"

    joblib.dump(best_model, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 64)
    print("IRIS CLASSIFICATION — TRAINING COMPLETE")
    print("=" * 64)
    print(f"Dataset:          Iris ({len(X)} samples, {X.shape[1]} features)")
    print(f"Split:            {len(X_train)} train / {len(X_test)} test")
    print(f"Best k:           {metrics['model']['best_k']}")
    print(f"CV macro F1:      {metrics['model']['best_cv_f1_macro']:.4f}")
    print(f"Test accuracy:    {metrics['test_metrics']['accuracy']:.4f}")
    print(f"Test macro F1:    {metrics['test_metrics']['f1_macro']:.4f}")
    print(f"Confusion matrix: {metrics['test_metrics']['confusion_matrix']}")
    print(f"Model saved to:   {model_path}")
    print(f"Metrics saved to: {report_path}")

    return metrics


if __name__ == "__main__":
    train()
