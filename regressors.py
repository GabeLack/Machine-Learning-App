import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR

from Interfaces import MLRegressorInterface

class LinearFactory(MLRegressorInterface):
    default_param_grid = {
                'polynomialfeatures__degree': np.arange(1, 8)
                }

    def create_model(self, pipeline=False, param_grid=None, scaler=StandardScaler(), **kwargs):

        if pipeline is False: # use a basic model
            self.model = LinearRegression(**kwargs) #! needs validation of kwargs

        else: # use the polyfeatures-scaler-estimator pipeline in a gridsearch
            if param_grid is None: # use the default param_grid
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                scaler, # StandardScaler() or MinMaxScaler() or RobustScaler()
                LinearRegression(**kwargs)
            )

            self.model = GridSearchCV(
                pipeline, # use the pipeline
                param_grid, # use the provided param_grid
                n_jobs=4, # number of cores to use
                cv=10, # number of cross-validation folds
                scoring="neg_mean_squared_error" # scoring metric
            )

class ElasticNetFactory(MLRegressorInterface):
    default_param_grid = {
                'polynomialfeatures__degree': np.arange(1, 8),
                'elasticnet__alpha': np.logspace(-1, 1, 10),
                'elasticnet__l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1] 
                # l1_ratio = 0 is ridge (L2 penalty), l1_ratio = 1 is lasso (L1 penalty)
                }
    
    def create_model(self, pipeline=False, param_grid=None, scaler=StandardScaler(), **kwargs):

        if pipeline is False:
            self.model = ElasticNet(**kwargs) #! needs validation of kwargs

        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                scaler,
                ElasticNet(**kwargs)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )

class SVRFactory(MLRegressorInterface):
    default_param_grid = {
                'polynomialfeatures__degree': np.arange(1, 8),
                'svr__C': np.logspace(0, 1, 10),
                'svr__epsilon': [0, 0.001, 0.01, 0.1, 0.5, 1, 2],
                'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'svr__gamma': ['scale', 'auto']
                }

    def create_model(self, pipeline=False, param_grid=None, scaler=StandardScaler(), **kwargs):

        if pipeline is False:
            self.model = SVR(**kwargs) #! needs validation of kwargs

        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                scaler,
                SVR(**kwargs)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )
