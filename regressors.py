import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Optimizer # imported for validation

from interfaces import MLRegressorInterface

class LinearFactory(MLRegressorInterface):
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8)
    }

    def create_model(self, param_grid: dict|None = None) -> None:
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
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8),
        'elasticnet__alpha': np.logspace(-1, 1, 10),
        'elasticnet__l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1]
        # l1_ratio = 0 is ridge (L2 penalty), l1_ratio = 1 is lasso (L1 penalty)
    }

    def create_model(self, param_grid: dict|None = None) -> None:
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
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 8),
        'svr__C': np.logspace(0, 1, 10),
        'svr__epsilon': [0, 0.001, 0.01, 0.1, 0.5, 1, 2],
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svr__gamma': ['scale', 'auto']
    }

    def create_model(self, param_grid: dict|None = None) -> None:
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
    #! This is a very heavy parameter grid
    default_param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 200],
        'optimizer': ['adam', 'rmsprop'],
        'neuron_layers': [ # shapes of the hidden layers
            (64, 64, 64),  # rectangle
            (32, 64, 128),  # expanding_triangle
            (128, 64, 32),  # diminishing_triangle
            (128, 32, 128),  # squeezed center
            (32, 128, 32)  # enlarged center
        ],
        'dropout_layers': [
            (0, 0, 0), # no dropout
            (0.1, 0.1, 0.2),
            (0.2, 0.2, 0.2),
            (0.3, 0.3, 0.3),
            (0.1, 0.2, 0.3)
        ]
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        if self.context.is_pipeline is False:
            self.model = KerasRegressor(build_fn=self.build_model)
        else:
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")
            pipeline = make_pipeline(
                self.context.scaler,
                KerasRegressor(build_fn=self.build_model)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="neg_mean_squared_error"
            )

    def build_model(self,
                    neuron_layers: tuple[int] = (64, 64),
                    dropout_layers: tuple[float] = (0.2, 0.2),
                    activation: str = 'relu',
                    optimizer: str = 'adam') -> Sequential:
        if not isinstance(neuron_layers, tuple) or any(not isinstance(layer, int) for layer in neuron_layers):
            raise ValueError("neuron_layers must be a tuple of integers.")
        if not isinstance(dropout_layers, tuple) or any(not isinstance(layer, float) for layer in dropout_layers):
            raise ValueError("dropout_layers must be a tuple of floats.")
        if not isinstance(optimizer, (str, Optimizer)):
            raise ValueError("Optimizer must be a string or a Keras optimizer instance.")
        # activation, and metrics are checked by Keras, they can also be non-string types

        model = Sequential()
        input_dim = self.context.X_train.shape[1]

        # Combine neuron and dropout layers
        combined_layers = []
        for neurons, dropout in zip(neuron_layers, dropout_layers):
            combined_layers.append(neurons)
            combined_layers.append(dropout)

        # Add input layer
        model.add(Dense(combined_layers[0], input_dim=input_dim, activation=activation))

        # Add hidden layers
        for i in combined_layers[1:]:
            if i >= 1:
                try: # Use error management in Dense to raise error further.
                    model.add(Dense(i, activation=activation))
                except Exception as e:
                    raise ValueError(f"An error occurred (layer): {e}") from e
            elif 0 <= i < 1: # dropout layer, cannot be i<=1.
                model.add(Dropout(i))
            else:
                raise ValueError(f"error in layer construction, invalid layer value: {i}\n" +\
                    f"value {i} should be a positive integer or float between 0 and 1.")

        # Add output layer
        model.add(Dense(1, activation='linear'))

        # Compile model
        try:
            model.compile(loss='mean_squared_error', optimizer=optimizer)
        except ValueError as e:
            raise ValueError(f"Error compiling model: {e}") from e
        return model

    def train_model(self) -> None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train, callbacks=[early_stopping])
