"""Module contains code for evaluating prediction result"""
import numpy as np


class EvaluationMetrics:
    """Main class responsibility is to calculate different metrics for model evaluation results"""

    def __init__(self, actual_y: np.array, predicted_y: np.ndarray):
        self._actual_y = actual_y
        self._predicted_y_proba = predicted_y
        self.__threshold = 0.5

    @property
    def precision(self) -> float:
        return self.__true_positive_count()/(self.__true_positive_count() + self.__false_positive_count())

    @property
    def recall(self) -> float:
        return self.__true_positive_count()/(self.__true_positive_count() + self.__false_negative_count())

    @property
    def f1_score(self) -> float:
        return 2*self.precision*self.recall/(self.precision + self.recall)

    def set_threshold(self, threshold: float) -> None:
        self.__threshold = threshold

    def __true_positive_count(self) -> int:
        return (self.__predicted_positive() & self.__actual_positive()).sum()

    def __true_negative_count(self) -> int:
        return (self.__predicted_negative() & self.__actual_negative()).sum()

    def __false_positive_count(self) -> int:
        return (self.__predicted_positive() & self.__actual_negative()).sum()

    def __false_negative_count(self) -> int:
        return (self.__predicted_negative() & self.__actual_positive()).sum()

    def __actual_positive(self) -> np.ndarray:
        return self._actual_y == 1

    def __actual_negative(self) -> np.ndarray:
        return self._actual_y == 0

    def __predicted_positive(self) -> np.ndarray:
        return self._predicted_y_proba >= self.__threshold

    def __predicted_negative(self) -> np.ndarray:
        return self._predicted_y_proba < self.__threshold

    def __str__(self):
        output = f"\nTrue positive rate is: {self.__true_positive_rate()}\n"
        output += f"True negative rate is: {self.__true_negative_rate()}\n"
        output += f"False positive rate is: {self.__false_positive_rate()}\n"
        output += f"False negative rate is: {self.__false_negative_rate()}\n"

        return output
