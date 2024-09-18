import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, accuracy_score)
from context import ModelContext


class MLInterface(ABC):
    def __init__(self, model_context: ModelContext) -> None:
        if not isinstance(model_context, ModelContext):
            raise TypeError("model_context must be an instance of ModelContext.")
        self.context = model_context
        self.X = model_context.X
        self.y = model_context.y
        self.X_train = model_context.X_train
        self.X_test = model_context.X_test
        self.y_train = model_context.y_train
        self.y_test = model_context.y_test
        self.model = None

    @abstractmethod
    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        # kwargs isn't needed in interface, but it's useful for the future implementation.
        # Return a sci-kit model
        pass

    def train_model(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def predict(self) -> pd.DataFrame:
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    @abstractmethod
    def metrics(self, filename: str) -> pd.DataFrame:
        pass


class MLClassifierInterface(MLInterface):
    @abstractmethod
    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        # kwargs isn't needed in interface, but it's useful for the future implementation.
        # Return a sci-kit model
        pass

    def metrics(self, filename: str) -> pd.DataFrame:
        # Calculate metrics
        #! use metrics() only after predict()
        # Generate the classification report as a dictionary
        report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)

        # Extract precision, recall, and f1-score for the 'macro avg'
        precision = report_dict['macro avg']['precision']
        recall = report_dict['macro avg']['recall']
        f1_score_ = report_dict['macro avg']['f1-score']

        # Get the name of the specific model
        model_name = self.model.estimator.steps[-1][0]

        # Calculate accuracy separately
        accuracy = accuracy_score(self.y_test, self.y_pred)

        # Save metrics to the classification_metrics DataFrame
        data = {'type': model_name,
                'refit time': self.model.refit_time_,
                'precision': precision,
                'recall': recall,
                'f1 score': f1_score_,
                'accuracy': accuracy}

        # Read existing metrics from CSV file
        try:
            metrics_df = pd.read_csv(filename)
        except FileNotFoundError:
            metrics_df = pd.DataFrame()

        # Append new data
        metrics_df = pd.concat([metrics_df, pd.DataFrame([data])], ignore_index=True)

        # Write updated metrics to CSV file
        metrics_df.to_csv(filename, index=False)

        return metrics_df


class MLRegressorInterface(MLInterface):
    @abstractmethod
    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        # kwargs isn't needed in interface, but it's useful for the future implementation.
        # Creates a estimator/model in self.model attribute
        pass

    def metrics(self, filename: str) -> pd.DataFrame:
        # Calculate metrics
        #! use metrics() only after predict()
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)

        # Get the name of the specific estimator model
        model_name = self.model.estimator.steps[-1][0]

        # Save metrics to the regression_metrics DataFrame
        data = {'type': model_name,
                'refit time': self.model.refit_time_,
                'mae': mae,
                'rmse': rmse,
                'r2': r2}

        # Read existing metrics from CSV file
        try:
            metrics_df = pd.read_csv(filename)
        except FileNotFoundError:
            metrics_df = pd.DataFrame()

        # Append new data
        metrics_df = pd.concat([metrics_df, pd.DataFrame([data])], ignore_index=True)

        # Write updated metrics to CSV file
        metrics_df.to_csv(filename, index=False)

        return metrics_df
