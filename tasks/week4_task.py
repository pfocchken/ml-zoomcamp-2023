import numpy as np
import pandas as pd

from src.data_helper import split_data, Data
from src.result_metrics import EvaluationMetrics
from tasks.common_dataset_steps import prepare_dataset, prepare_target_column, TARGET_NAME, NUMERIC_COLUMNS, \
    CATEGORICAL_COLUMNS
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def draw_roc_curves(data_to_draw: Data) -> None:
    for column in NUMERIC_COLUMNS:
        fpr, tpr, _ = roc_curve(data_to_draw.train.y, data_to_draw.train.X[column])
        plt.plot(fpr, tpr, label=column)
        # plt.plot([1, 0], [0, 1], label="base")  # TODO: Figure out what is wrong with graphic

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()


def draw_metrics(thresholds: np.ndarray, precisions: list, recalls: list):
    plt.plot(thresholds, precisions, label="precisions", color="red")
    plt.plot(thresholds, recalls, label="recalls", color="green")

    plt.legend()
    plt.show()


# TODO: move to src
def calculate_cv_auc_scores(data: Data, c: float = 1.0) -> list[float]:
    full_train_df = pd.concat([data.train.X, data.validation.X])
    full_train_df = full_train_df.reset_index(drop=True)
    full_train_y_df = np.concatenate((data.train.y, data.validation.y))

    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)

    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000)

    cv_auc_scores = []

    for train_idx, validation_idx in k_fold.split(full_train_df):
        model.fit(full_train_df.iloc[train_idx], full_train_y_df[train_idx])
        kfold_prediction = model.predict_proba(full_train_df.iloc[validation_idx])[:, 1]

        cv_auc_scores.append(roc_auc_score(full_train_y_df[validation_idx], kfold_prediction))

    return cv_auc_scores


if __name__ == "__main__":

    # Prepare dataset

    cars_full_df = prepare_dataset()
    cars_full_df = prepare_target_column(cars_full_df, drop_source=True)
    cars_one_hot_df = pd.get_dummies(cars_full_df, columns=CATEGORICAL_COLUMNS)
    data = split_data(cars_one_hot_df, TARGET_NAME, random_state=1)
    # data = split_data(cars_full_df, TARGET_NAME, random_state=1)

    # Question 1: ROC AUC feature importance. Which numerical variable (among the following 4) has the highest AUC?

    # draw_roc_curves(data)
    auc_scores = {}

    for column_name in NUMERIC_COLUMNS:
        auc_scores[column_name] = roc_auc_score(data.train.y, data.train.X[column_name])

        if auc_scores[column_name] < 0.5:
            auc_scores[column_name] = roc_auc_score(data.train.y, -data.train.X[column_name])

    print("QUESTION 1: Which numerical variable (among the following 4) has the highest AUC?\n")
    print(f"AUCs\n: {auc_scores}:\n")
    print("The highest AUC has ENGINE_HP feature\n")

    # Question 2: Training the model. What's the AUC of this model on the validation dataset? (round to 3 digits)

    # features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    # train_features = data.train.X[features].to_dict(orient="records")
    # dv = DictVectorizer(sparse=False)
    # train_features = dv.fit_transform(train_features)

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    # model.fit(train_features, data.train.y)
    model.fit(data.train.X, data.train.y)

    # validation_features = data.validation.X[features].to_dict(orient="records")
    # validation_features = dv.transform(validation_features)

    price_prediction = model.predict_proba(data.validation.X)[:, 1]

    validation_auc = roc_auc_score(data.validation.y, price_prediction)

    print("\nQUESTION 2: What's the AUC of this model on the validation dataset\n")
    print(f"The AUC for validation dataset is: {validation_auc}")

    # Question 3: Precision and Recall. At which threshold precision and recall curves intersect?

    metrics = EvaluationMetrics(data.validation.y, price_prediction)

    thresholds = np.linspace(0, 1.0, 101)
    precisions = []
    recalls = []
    f1_scores = {}

    for threshold in thresholds:
        metrics.set_threshold(threshold)
        precisions.append(metrics.precision)
        recalls.append(metrics.recall)
        f1_scores[threshold] = metrics.f1_score

    # draw_metrics(thresholds, precisions, recalls)

    print("\nQUESTION 3: At which threshold precision and recall curves intersect?\n")
    print(f"Precision and recall curves intersects at Threshold 0.48")

    # Question 4: F1 score. At which threshold F1 is maximal?

    print("\nQUESTION 4: At which threshold F1 is maximal?\n")
    print(f"F1-score is maximum for threshold: {max(f1_scores, key=lambda x: f1_scores[x])}")

    # Question 5: 5-Fold CV. How large is standard deviation of the scores across different folds?

    cross_validation_auc_scores = calculate_cv_auc_scores(data)

    print("\nQUESTION 5: At which threshold F1 is maximal?\n")
    print(f"Standard deviation across different folds is: {np.std(cross_validation_auc_scores)}")

    # Question 6: Hyperparemeter Tuning. Which C leads to the best mean score?

    cs = [0.01, 0.1, 0.5, 10]

    model_tuning_results = {}

    for c in cs:
        cross_validation_auc_scores = calculate_cv_auc_scores(data, c)
        model_tuning_results[c] = (np.mean(cross_validation_auc_scores), np.std(cross_validation_auc_scores))

    print(model_tuning_results)

    print("\nQUESTION 6: Which C leads to the best mean score?\n")
    print(f"C that leads to best mean score is: TEN - 10")

    assert not cars_full_df.empty













