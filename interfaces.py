"""
NAME
    interfaces

DESCRIPTION
    This module provides abstract base classes for machine learning models, including classifiers and regressors.
    It defines interfaces for creating, training, predicting, and evaluating models.

CLASSES
    MLInterface
        Abstract base class for machine learning models.
        
        Methods defined here:
        
        __init__(self, model_context: ModelContext) -> None
            Initializes the MLInterface with the given model context.
        
        create_model(self, param_grid: dict|None = None) -> None
            Abstract method to create a machine learning model.
        
        train_model(self) -> None
            Trains the machine learning model.
        
        predict(self) -> pd.DataFrame
            Makes predictions using the trained model.
        
        metrics(self, filename: str) -> pd.DataFrame
            Abstract method to calculate and save model metrics.

    MLClassifierInterface
        Abstract base class for machine learning classifiers.
        
        Methods defined here:
        
        create_model(self, param_grid: dict|None = None) -> None
            Abstract method to create a classifier model.
        
        metrics(self, filename: str) -> pd.DataFrame
            Calculates and saves classification metrics.

    MLRegressorInterface
        Abstract base class for machine learning regressors.
        
        Methods defined here:
        
        create_model(self, param_grid: dict|None = None) -> None
            Abstract method to create a regressor model.
        
        metrics(self, filename: str) -> pd.DataFrame
            Calculates and saves regression metrics.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, accuracy_score)
from context import ModelContext


class MLInterface(ABC):
    """
    Abstract base class for machine learning models.
    
    Attributes:
        context (ModelContext): The context in which the model is created.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        X_train (pd.DataFrame): The training feature matrix.
        X_test (pd.DataFrame): The test feature matrix.
        y_train (pd.Series): The training target vector.
        y_test (pd.Series): The test target vector.
        model (object): The machine learning model.
    
    Methods:
        __init__(self, model_context: ModelContext) -> None
            Initializes the MLInterface with the given model context.
        
        create_model(self, param_grid: dict|None = None) -> None
            Abstract method to create a machine learning model.
        
        train_model(self) -> None
            Trains the machine learning model.
        
        predict(self) -> pd.DataFrame
            Makes predictions using the trained model.
        
        metrics(self, filename: str) -> pd.DataFrame
            Abstract method to calculate and save model metrics.
    """
    
    def __init__(self, model_context: ModelContext) -> None:
        """
        Initializes the MLInterface with the given model context.
        
        Args:
            model_context (ModelContext): The context in which the model is created.
        
        Raises:
            TypeError: If model_context is not an instance of ModelContext.
        """
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
    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Abstract method to create a machine learning model.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Returns:
            None
        """
        pass

    def train_model(self) -> None:
        """
        Trains the machine learning model.
        
        Returns:
            None
        """
        self.model.fit(self.X_train, self.y_train)

    def predict(self) -> pd.DataFrame:
        """
        Makes predictions using the trained model.
        
        Returns:
            pd.DataFrame: The predicted values.
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    @abstractmethod
    def metrics(self, filename: str) -> pd.DataFrame:
        """
        Abstract method to calculate and save model metrics.
        
        Args:
            filename (str): The name of the file to save the metrics.
        
        Returns:
            pd.DataFrame: The metrics DataFrame.
        """
        pass


class MLClassifierInterface(MLInterface):
    """
    Abstract base class for machine learning classifiers.
    
    Methods:
        create_model(self, param_grid: dict|None = None) -> None
            Abstract method to create a classifier model.
        
        metrics(self, filename: str) -> pd.DataFrame
            Calculates and saves classification metrics.
    """
    
    @abstractmethod
    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Abstract method to create a classifier model.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Returns:
            None
        """
        pass

    def metrics(self, filename: str) -> pd.DataFrame:
        """
        Calculates and saves classification metrics.
        
        Args:
            filename (str): The name of the file to save the metrics.
        
        Returns:
            pd.DataFrame: The metrics DataFrame.
        """
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
    """
    Abstract base class for machine learning regressors.
    
    Methods:
        create_model(self, param_grid: dict|None = None) -> None
            Abstract method to create a regressor model.
        
        metrics(self, filename: str) -> pd.DataFrame
            Calculates and saves regression metrics.
    """
    
    @abstractmethod
    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Abstract method to create a regressor model.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Returns:
            None
        """
        pass

    def metrics(self, filename: str) -> pd.DataFrame:
        """
        Calculates and saves regression metrics.
        
        Args:
            filename (str): The name of the file to save the metrics.
        
        Returns:
            pd.DataFrame: The metrics DataFrame.
        """
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
