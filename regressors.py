"""
NAME
    regressors

DESCRIPTION
    This module provides factory classes for creating various regression models.
    It includes factories for Linear Regression, ElasticNet, SVR, and ANN regressors.

CLASSES
    LinearFactory
        Factory class for creating a Linear Regression model.
        
        Methods defined here:
        
        create_model(self, param_grid: dict|None = None) -> None
            Creates a Linear Regression model with optional parameter grid for GridSearchCV.

    ElasticNetFactory
        Factory class for creating an ElasticNet Regression model.
        
        Methods defined here:
        
        create_model(self, param_grid: dict|None = None) -> None
            Creates an ElasticNet Regression model with optional parameter grid for GridSearchCV.

    SVRFactory
        Factory class for creating an SVR model.
        
        Methods defined here:
        
        create_model(self, param_grid: dict|None = None) -> None
            Creates an SVR model with optional parameter grid for GridSearchCV.

    ANNRegressorFactory
        Factory class for creating an ANN Regressor model.
        
        Methods defined here:
        
        create_model(self, param_grid: dict|None = None) -> None
            Creates an ANN Regressor model with optional parameter grid for GridSearchCV.
        
        build_model(self, neuron_layers: tuple[int] = (64, 64), dropout_layers: tuple[float] = (0.2, 0.2), activation: str = 'relu', optimizer: str = 'adam') -> Sequential
            Builds a Sequential ANN model with the specified parameters.
        
        train_model(self) -> None
            Trains the ANN model with early stopping.
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR

from scikeras.wrappers import KerasRegressor

from interfaces import MLRegressorInterface
from build_model import build_model_factory # to get callable build_model function

class LinearFactory(MLRegressorInterface):
    """
    Factory class for creating a Linear Regression model.
    
    Attributes:
        default_param_grid (dict): Default parameter grid for GridSearchCV.
    
    Methods:
        create_model(self, param_grid: dict|None = None) -> None
            Creates a Linear Regression model with optional parameter grid for GridSearchCV.
    """
    
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8)
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Creates a Linear Regression model with optional parameter grid for GridSearchCV.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Raises:
            ValueError: If param_grid is not a non-empty dictionary.
        """
        if self.context.is_pipeline is False:  # use a basic model
            self.model = LinearRegression()

        else:  # use the polyfeatures-scaler-estimator pipeline in a gridsearch
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,  # StandardScaler() or MinMaxScaler() or RobustScaler()
                LinearRegression()
            )

            self.model = GridSearchCV(
                pipeline,  # use the pipeline
                param_grid,  # use the provided param_grid
                n_jobs=4,  # number of cores to use
                cv=10,  # number of cross-validation folds
                scoring="neg_mean_squared_error"  # scoring metric
            )


class ElasticNetFactory(MLRegressorInterface):
    """
    Factory class for creating an ElasticNet Regression model.
    
    Attributes:
        default_param_grid (dict): Default parameter grid for GridSearchCV.
    
    Methods:
        create_model(self, param_grid: dict|None = None) -> None
            Creates an ElasticNet Regression model with optional parameter grid for GridSearchCV.
    """
    
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8),
        'elasticnet__alpha': np.logspace(-1, 1, 10),
        'elasticnet__l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1]
        # l1_ratio = 0 is ridge (L2 penalty), l1_ratio = 1 is lasso (L1 penalty)
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Creates an ElasticNet Regression model with optional parameter grid for GridSearchCV.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Raises:
            ValueError: If param_grid is not a non-empty dictionary.
        """
        if self.context.is_pipeline is False:
            self.model = ElasticNet()
        else:
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,
                ElasticNet()
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )


class SVRFactory(MLRegressorInterface):
    """
    Factory class for creating an SVR model.
    
    Attributes:
        default_param_grid (dict): Default parameter grid for GridSearchCV.
    
    Methods:
        create_model(self, param_grid: dict|None = None) -> None
            Creates an SVR model with optional parameter grid for GridSearchCV.
    """
    
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8),
        'svr__C': np.logspace(0, 1, 10),
        'svr__epsilon': [0, 0.001, 0.01, 0.1, 0.5, 1, 2],
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svr__gamma': ['scale', 'auto']
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Creates an SVR model with optional parameter grid for GridSearchCV.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Raises:
            ValueError: If param_grid is not a non-empty dictionary.
        """
        if self.context.is_pipeline is False:
            self.model = SVR()
        else:
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,
                SVR()
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )


class ANNRegressorFactory(MLRegressorInterface):
    """
    Factory class for creating an ANN Regressor model.
    
    Attributes:
        default_param_grid (dict): Default parameter grid for GridSearchCV.
    
    Methods:
        create_model(self, param_grid: dict|None = None) -> None
            Creates an ANN Regressor model with optional parameter grid for GridSearchCV.
        
        build_model(self, neuron_layers: tuple[int] = (64, 64), dropout_layers: tuple[float] = (0.2, 0.2), activation: str = 'relu', optimizer: str = 'adam') -> Sequential
            Builds a Sequential ANN model with the specified parameters.
        
        train_model(self) -> None
            Trains the ANN model with early stopping.
    """
    
    #! This is a very heavy parameter grid
    default_param_grid = {
        'kerasregressor__epochs': [50, 100, 200],
        'kerasregressor__batch_size': [16, 32, 64],
        'kerasregressor__neuron_layers': [ # shapes of the hidden layers
            (64, 64, 64),  # rectangle
            (32, 64, 128),  # expanding_triangle
            (128, 64, 32),  # diminishing_triangle
            (128, 32, 128),  # squeezed center
            (32, 128, 32)  # enlarged center
        ],
        'kerasregressor__dropout_layers': [
            (0, 0, 0), # no dropout
            (0.1, 0.1, 0.2),
            (0.2, 0.2, 0.2),
            (0.3, 0.3, 0.3),
            (0.1, 0.2, 0.3)
        ],
        'kerasregressor__optimizer': ['adam', 'rmsprop']
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        """
        Creates an ANN Regressor model with optional parameter grid for GridSearchCV.
        
        Args:
            param_grid (dict|None): The parameter grid for model tuning.
        
        Raises:
            ValueError: If param_grid is not a non-empty dictionary.
        """
        input_dim = self.context.X_train.shape[1]
        if self.context.is_pipeline is False:
            self.model = KerasRegressor(build_fn=build_model_factory(input_dim))
        else:
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")
            
            pipeline = make_pipeline(
                self.context.scaler,
                KerasRegressor(
                    build_fn=build_model_factory(input_dim),
                    neuron_layers=(64, 64),
                    dropout_layers=(0.2, 0.2)
                )
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )