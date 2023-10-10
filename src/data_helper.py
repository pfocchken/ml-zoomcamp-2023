"""Module contains helper methods and dataclasses for common data manipulations"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataRecord:
    X: pd.DataFrame
    y: np.array


@dataclass
class Data:
    train: DataRecord
    validation: DataRecord
    test: DataRecord


def split_data(data_to_split: pd.DataFrame, target_column: str, random_state=42) -> Data:

    train_df, left_df = train_test_split(data_to_split, train_size=0.6, random_state=random_state)
    validation_df, test_df = train_test_split(left_df, train_size=0.5, random_state=random_state)

    train_X, train_y = _split_to_features_and_target(train_df, target_column)
    validation_X, validation_y = _split_to_features_and_target(validation_df, target_column)
    test_X, test_y = _split_to_features_and_target(test_df, target_column)

    return Data(
        DataRecord(train_X, train_y),
        DataRecord(validation_X, validation_y),
        DataRecord(test_X, test_y),
    )


def _split_to_features_and_target(data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = data[target_column].to_numpy()
    X = data.drop(target_column, axis="columns")

    return X, y
