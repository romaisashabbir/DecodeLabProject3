"""Predict an Iris class using the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_knn_pipeline.joblib"

CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )

    model = joblib.load(MODEL_PATH)

    X = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )

    prediction = int(model.predict(X)[0])
    return CLASS_NAMES[prediction]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Iris flower species.")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)
    args = parser.parse_args()

    result = predict(
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,
    )
    print(f"Predicted class: {result}")


if __name__ == "__main__":
    main()
