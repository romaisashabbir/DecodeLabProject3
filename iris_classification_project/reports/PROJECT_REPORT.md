# Project 2 Report — Data Classification Using AI

## 1. Objective

Build a basic supervised classification model using a small dataset. The implementation follows the assignment's required flow: dataset understanding, train-test splitting, feature scaling, KNN classification, and output validation with a confusion matrix and F1 score.

## 2. Dataset

The Iris benchmark contains 150 samples, 3 classes, and 4 numerical dimensions:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The three target classes are:

- Setosa
- Versicolor
- Virginica

## 3. Methodology

### Step 1 — Load data

The Iris dataset is loaded through scikit-learn and exported to `data/iris.csv` for inspection and reproducibility.

### Step 2 — Train-test split

The dataset is divided into:

- 80% training data
- 20% test data

The split is stratified so that each class remains represented proportionally in both sets.

### Step 3 — Feature scaling

KNN is distance-based, so numerical features are standardized using `StandardScaler`.

The scaler is placed inside a scikit-learn `Pipeline`. This is important because it prevents information from the test set from influencing preprocessing.

### Step 4 — KNN classification

K-Nearest Neighbors classifies a sample based on nearby training examples. The project evaluates odd `k` values from 1 through 15.

Instead of selecting `k` using the test set, 5-fold stratified cross-validation is performed on the training set. Macro F1 is used as the tuning metric.

### Step 5 — Validation

The final selected model is evaluated once on the held-out test set using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix
- Per-class classification report

## 4. Why accuracy is not enough

The assignment material emphasizes that accuracy can be misleading when classes are imbalanced. Although Iris is balanced, the project still reports precision, recall, F1, and the confusion matrix to establish a stronger evaluation practice.

## 5. Reproducibility

A fixed random seed (`42`) is used for the train-test split and cross-validation shuffle. The final pipeline is serialized with Joblib, and metrics are stored as JSON.

## 6. Engineering quality

The implementation separates:

- training/evaluation logic (`src/train.py`)
- inference (`src/predict.py`)
- tests (`tests/`)
- generated model artifacts (`models/`)
- generated reports (`reports/`)

This makes the project easier to maintain and extend.

## 7. Conclusion

The resulting system demonstrates a complete supervised-learning workflow rather than only a single model-fitting statement. It is intentionally small enough for the assignment but follows professional practices around preprocessing, leakage prevention, validation, evaluation, artifact persistence, and testing.
