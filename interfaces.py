import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from enum import Enum,auto

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, f1_score,
                             classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)

class MLClassifierInterface(ABC):
    def __init__(self, data_file_path, target_column, test_size=0.3):
        self.data_file_path = data_file_path
        self.target_column = target_column
        self.model = None
        self.model_name = None
        self.X = pd.read_csv(
            self.data_file_path).drop(
            self.target_column,
            axis=1)
        self.y = pd.read_csv(self.data_file_path)[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=101)

    @abstractmethod
    def create_model(self, **kwargs):
        # kwargs isn't needed in interface, but it's useful for the future implementation.
        # Return a sci-kit model
        pass

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def metrics(self):
        return classification_report(self.y_test, self.predict())

class MLRegressorInterface(ABC):
    def __init__(self, data_file_path, target_column, test_size=0.3):
        self.data_file_path = data_file_path
        self.target_column = target_column
        self.model = None
        self.X = pd.read_csv(
            self.data_file_path).drop(
            self.target_column,
            axis=1)
        self.y = pd.read_csv(self.data_file_path)[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=101)

    @abstractmethod
    def create_model(self, **kwargs):
        # kwargs isn't needed in interface, but it's useful for the future implementation.
        # Return a sci-kit model
        pass

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def metrics(self):
        # Regression metrics
        return {
            "MAE": mean_absolute_error(self.y_test, self.y_pred),
            "MSE": mean_squared_error(self.y_test, self.y_pred),
            "R2": r2_score(self.y_test, self.y_pred)
        }