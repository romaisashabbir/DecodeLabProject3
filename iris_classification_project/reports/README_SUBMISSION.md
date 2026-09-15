# Submission Checklist

Before submitting:

1. Run:
   `python -m src.train`
2. Confirm `models/iris_knn_pipeline.joblib` exists.
3. Confirm `reports/metrics.json` exists.
4. Run:
   `pytest -q`
5. Run a sample prediction:
   `python -m src.predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2`
6. Include the repository/project folder and `PROJECT_REPORT.md`.

Suggested viva explanation:

> "I used the Iris dataset and performed a stratified 80/20 train-test split. Because KNN is distance-based, I standardized the features. I kept scaling inside a Pipeline to avoid data leakage. I selected k using 5-fold stratified cross-validation on the training set using macro F1, then evaluated the selected model once on the untouched test set. I reported accuracy, precision, recall, F1 and the confusion matrix, and saved the final pipeline for inference."
