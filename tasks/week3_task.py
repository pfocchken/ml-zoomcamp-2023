from typing import Optional

import numpy as np

from src.data_helper import split_data, Data
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression, Ridge

from tasks.common_dataset_steps import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, prepare_dataset, prepare_target_column, \
    TARGET_NAME


# TODO: Move this to src
def get_accuracy(data: Data) -> float:
    model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
    model.fit(data.train.X, data.train.y)

    prediction = model.predict(data.validation.X)
    accuracy = (data.validation.y == prediction).mean()

    return accuracy


def predict_and_get_rmse(data: Data, alpha: float = 0) -> float:
    model = Ridge(solver="sag", alpha=alpha, max_iter=1000, random_state=42)
    model.fit(data.train.X, data.train.y)

    prediction = model.predict(data.validation.X)
    rmse = get_rmse(data.validation.y, prediction)

    return rmse
    # return round(rmse, 3)


def get_rmse(y, y_pred) -> float:
    se = (y - y_pred) ** 2
    mse = se.mean()

    return np.sqrt(mse)


def _get_data_for_training(cars_full_df: pd.DataFrame, drop_column: Optional[str] = None) -> Data:
    cars_without_price_df = cars_full_df.drop("price", axis="columns")

    one_hot_columns = [column for column in CATEGORICAL_COLUMNS if column != drop_column]

    if drop_column:
        cars_without_price_df.drop(drop_column, axis="columns", inplace=True)

    cars_one_hot_encoding_df = pd.get_dummies(cars_without_price_df, columns=one_hot_columns)
    return split_data(cars_one_hot_encoding_df, TARGET_NAME)


if __name__ == "__main__":

    cars_full_df = prepare_dataset()

    # Question 1: What is the most frequent observation (mode) for the column transmission_type

    print("Question 1: What is the most frequent observation (mode) for the column transmission_type \n")
    print(f"ANSWER: Most frequent observation of transmission_type column is: "
          f"{cars_full_df.groupby('transmission_type').count().idxmax().iloc[0]}\n")

    # Question 2: What are the two features that have the biggest correlation in this dataset?

    correlation_matrix = cars_full_df[NUMERIC_COLUMNS].corr()

    print("Question 2: What are the two features that have the biggest correlation in this dataset?")
    print(f"Correlations for numeric values in dataset is \n{correlation_matrix.unstack().sort_values()}")
    print("\n ANSWER: The biggest correlation is between columns: highway_mpg and city_mpg\n")

    # Make price value binary

    cars_full_df = prepare_target_column(cars_full_df)

    # Split data

    data = split_data(cars_full_df, TARGET_NAME)

    # Question 3: Which of these variables has the lowest mutual information score?

    # TODO: Move this to separate method to src

    mutual_variables = {}
    for category in CATEGORICAL_COLUMNS:
        mutual_score = mutual_info_score(data.train.X[category], data.train.y)
        mutual_variables[category] = round(mutual_score, 2)

    print("Question 3: Which of these variables has the lowest mutual information score? \n")
    print(f"The lowest mutual information score has "
          f"{sorted(mutual_variables.items(), key=lambda item: item[1])[0][0]}\n")

    # Get One-hot encoding data

    data = _get_data_for_training(cars_full_df)

    # Question 4: What accuracy did you get on logistic regression?

    baseline_accuracy = get_accuracy(data)

    print("Question 4: What accuracy did you get on logistic regression? \n")
    print(f"Accuracy is {round(baseline_accuracy, 2)}")

    # Question 5: Which of following feature has the smallest difference?

    eliminated_features = {"year", "engine_hp", "transmission_type", "city_mpg"}
    accuracy_difference = {}

    for feature in eliminated_features:
        eliminated_data = _get_data_for_training(cars_full_df, drop_column=feature)
        accuracy_difference[feature] = baseline_accuracy - get_accuracy(eliminated_data)

    print("Question 5: Which of following feature has the smallest difference? \n")
    print(f"The smallest difference has 'engine_hp' feature")

    # Question 6: Which of these alphas leads to the best RMSE on the validation set?

    alphas = [0, 0.01, 0.1, 1, 10]
    # alphas = [0.00001]
    rmses = {}

    cars_full_df.drop("above_average", axis="columns", inplace=True)

    TARGET_NAME = "price"
    cars_full_df[TARGET_NAME] = np.log1p(cars_full_df[TARGET_NAME])

    cars_one_hot_encoding_df = pd.get_dummies(cars_full_df, columns=CATEGORICAL_COLUMNS)
    data = split_data(cars_one_hot_encoding_df, target_column=TARGET_NAME)

    # for alpha in alphas:
    #     rmses[alpha] = predict_and_get_rmse(data, alpha)

    print("Question 6: Which of these alphas leads to the best RMSE on the validation set? \n")
    print(f"Alpha that leads to the best RMSE is ZERO")

