from pathlib import Path

import joblib
import pandas as pd

from src.train import build_pipeline, load_dataset


def test_dataset_shape():
    X, y, feature_names, class_names = load_dataset()
    assert X.shape == (150, 4)
    assert len(y) == 150
    assert len(feature_names) == 4
    assert len(class_names) == 3


def test_pipeline_has_scaler_and_classifier():
    pipeline = build_pipeline()
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_can_fit_and_predict():
    X, y, _, _ = load_dataset()
    pipeline = build_pipeline()
    pipeline.set_params(classifier__n_neighbors=5)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X.iloc[:5])
    assert len(predictions) == 5


def test_saved_model_has_expected_interface():
    model_path = Path("models/iris_knn_pipeline.joblib")
    if model_path.exists():
        model = joblib.load(model_path)
        sample = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2]],
            columns=[
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
        )
        assert model.predict(sample).shape == (1,)
