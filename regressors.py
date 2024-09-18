import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor

from interfaces import MLRegressorInterface

class LinearFactory(MLRegressorInterface):
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8)
    }

    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        if self.context.is_pipeline is False:  # use a basic model
            self.model = LinearRegression(**kwargs)  #! needs validation of kwargs
        else:  # use the polyfeatures-scaler-estimator pipeline in a gridsearch
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,  # StandardScaler() or MinMaxScaler() or RobustScaler()
                LinearRegression(**kwargs)
            )

            self.model = GridSearchCV(
                pipeline,  # use the pipeline
                param_grid,  # use the provided param_grid
                n_jobs=4,  # number of cores to use
                cv=10,  # number of cross-validation folds
                scoring="neg_mean_squared_error"  # scoring metric
            )


class ElasticNetFactory(MLRegressorInterface):
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8),
        'elasticnet__alpha': np.logspace(-1, 1, 10),
        'elasticnet__l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1]
        # l1_ratio = 0 is ridge (L2 penalty), l1_ratio = 1 is lasso (L1 penalty)
    }

    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        if self.context.is_pipeline is False:
            self.model = ElasticNet(**kwargs)  #! needs validation of kwargs
        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,
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

    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        if self.context.is_pipeline is False:
            self.model = SVR(**kwargs)  #! needs validation of kwargs
        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,
                SVR(**kwargs)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )

class ANNRegressorFactory(MLRegressorInterface):
    #! Used a new simpler approach rather than old KerasANN exam question
    # I think it could theoritcally be coupled to that later?
    # Albeit with a very heavy GridSearchCV
    default_param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 200],
        'optimizer': ['adam', 'rmsprop'],
        'dropout_rate': [0.0, 0.2, 0.5],
        'neurons': [32, 64, 128],
        'metrics': [['mean_squared_error'], 
                    ['mean_absolute_error'], 
                    ['mean_squared_error', 'mean_absolute_error'], 
                    ['mean_squared_error', 'RootMeanSquaredError']]
    }

    def create_model(self, param_grid: dict|None = None, **kwargs) -> None:
        if param_grid is None:
            param_grid = self.default_param_grid

        if self.context.is_pipeline is False:
            self.model = KerasRegressor(build_fn=self.build_model, **kwargs)
        else:
            pipeline = make_pipeline(
                self.context.scaler,
                KerasRegressor(build_fn=self.build_model, **kwargs)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )

    def build_model(self,
                    optimizer: str = 'adam',
                    dropout_rate: float = 0.0,
                    neurons: int = 64,
                    metrics: list[str] = ['mean_squared_error']) -> Sequential:
        model = Sequential()
        model.add(Dense(neurons, input_dim=self.context.X_train.shape[1], activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=metrics)
        return model
