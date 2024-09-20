"""
NAME
    context

DESCRIPTION
    This module provides the ModelContext class for managing the context of machine learning models.
    It includes methods for checking missing values and feature types.

CLASSES
    ModelContext
        Manages the context for machine learning models, including data splitting and preprocessing.
        
        Methods defined here:
        
        __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.3, 
                 is_pipeline: bool = True, scaler=StandardScaler()) -> None
            Initializes the ModelContext with the given parameters.
        
        check_missing(self) -> None
            Checks for missing values in the DataFrame and raises a warning if any are found.
        
        check_feature_types(self) -> None
            Checks for object type features in the DataFrame and raises a warning if any are found.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import inspect
import sklearn.preprocessing

class ModelContext:
    """
    Manages the context for machine learning models, including data splitting and preprocessing.
    
    Attributes:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        is_pipeline (bool): Whether to use a pipeline for preprocessing.
        scaler (object): The scaler object for preprocessing.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        X_train (pd.DataFrame): The training feature matrix.
        X_test (pd.DataFrame): The test feature matrix.
        y_train (pd.Series): The training target vector.
        y_test (pd.Series): The test target vector.
    
    Methods:
        __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.3, 
                 is_pipeline: bool = True, scaler=StandardScaler()) -> None
            Initializes the ModelContext with the given parameters.
        
        check_missing(self) -> None
            Checks for missing values in the DataFrame and raises a warning if any are found.
        
        check_feature_types(self) -> None
            Checks for object type features in the DataFrame and raises a warning if any are found.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target_column: str,
                 test_size: float = 0.3,
                 is_pipeline: bool = True,
                 scaler=StandardScaler()) -> None:
        """
        Initializes the ModelContext with the given parameters.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.
            test_size (float): The proportion of the dataset to include in the test split.
            is_pipeline (bool): Whether to use a pipeline for preprocessing.
            scaler (object): The scaler object for preprocessing.
        
        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If target_column is not a string or does not exist in the DataFrame.
            ValueError: If test_size is not a float between 0 and 1.
            TypeError: If is_pipeline is not a boolean.
            ValueError: If scaler is not an instance of a class from sklearn.preprocessing.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(target_column, str) or target_column not in df.columns:
            raise ValueError("target_column must be a string and exist in the DataFrame.")
        if not isinstance(test_size, float) or not 0 < test_size < 1:
            raise ValueError("test_size must be a float between 0 and 1.")
        if not isinstance(is_pipeline, bool):
            raise TypeError("is_pipeline must be a boolean.")

        # Get all classes from sklearn.preprocessing
        preprocessing_classes = [
            cls for _, cls in inspect.getmembers(sklearn.preprocessing, inspect.isclass)
        ]
        # Check if the provided scaler is an instance of any of the preprocessing classes
        if not any(isinstance(scaler, cls) for cls in preprocessing_classes):
            raise ValueError("scaler must be an instance of a class from sklearn.preprocessing.")

        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.is_pipeline = is_pipeline
        self.scaler = scaler

        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=101)
        # check_missing and check_feature_types are not called in the constructor,
        # but they are called in the app code.

    def check_missing(self) -> None:
        """
        Checks for missing values in the DataFrame and raises a warning if any are found.
        
        Raises:
            UserWarning: If any columns have missing values.
        """
        missing_values = self.df.isna().sum()
        for column, count in missing_values.items():
            if count > 0:
                raise UserWarning(f"Warning: Column {column} has {count} missing values.")

    def check_feature_types(self) -> None:
        """
        Checks for object type features in the DataFrame and raises a warning if any are found.
        
        Raises:
            UserWarning: If any columns are of object type and may need encoding.
        """
        object_columns = self.X.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            raise UserWarning(
                f"Warning: Columns {list(object_columns)} are of object type and may need encoding."
            )
