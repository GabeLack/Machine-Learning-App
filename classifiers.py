import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Optimizer # imported for validation

from interfaces import MLClassifierInterface


class LogisticFactory(MLClassifierInterface):
    default_param_grid = {
                'polynomialfeatures__degree': np.arange(1,6), # degrees of polynomial features
                'logisticregression__C': np.logspace(0,2,5), # regularization strength
                'logisticregression__penalty': ['l1','l2','elasticnet'], # regularization type
                'logisticregression__solver': ['lbfgs','liblinear','newton-cg', 
                                               'newton-cholesky','saga'], # optimization algorithm
                'logisticregression__class_weight': [None,'balanced'] # class weights
                }

    def create_model(self, param_grid: dict|None = None) -> None:

        if self.context.is_pipeline is False: # use a basic model
            self.model = LogisticRegression()

        else: # use the polyfeatures-scaler-estimator pipeline in a gridsearch
            if param_grid is None: # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler, # StandardScaler() or MinMaxScaler() or RobustScaler()
                LogisticRegression(max_iter=1000)
            )

            self.model = GridSearchCV(
                pipeline, # use the pipeline
                param_grid, # use the provided param_grid
                n_jobs=4, # number of cores to use
                cv=10, # number of cross-validation folds
                scoring="accuracy" # scoring metric
            )


class SVCFactory(MLClassifierInterface):
    default_param_grid = {
        'svc__degree': np.arange(1, 11),
        'svc__C': np.logspace(0, 1, 10),
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svc__gamma': ['scale', 'auto'],
        'svc__class_weight': [None, 'balanced']
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        if self.context.is_pipeline is False:
            self.model = SVC()
        else:
            if param_grid is None:
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                self.context.scaler,
                SVC()
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="accuracy"
            )


class RandomForestFactory(MLClassifierInterface):
    default_param_grid = {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
        'randomforestclassifier__max_depth': [None, 10, 20, 30],
        'randomforestclassifier__class_weight': [None, 'balanced']
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        if self.context.is_pipeline is False:
            self.model = RandomForestClassifier()
        else:
            if param_grid is None:
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                self.context.scaler,
                RandomForestClassifier()
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="accuracy"
            )


class KNNFactory(MLClassifierInterface):
    default_param_grid = {
        'polynomialfeatures__degree': np.arange(1, 11),
        'kneighborsclassifier__n_neighbors': np.arange(1, 31),
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__metric': ['manhattan', 'euclidean', 'minkowski', 'chebyshev', 'mahalanobis', 'seuclidean']
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        if self.context.is_pipeline is False:
            self.model = KNeighborsClassifier()
        else:
            if param_grid is None:
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,
                KNeighborsClassifier()
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="accuracy"
            )


class GradientBoostingFactory(MLClassifierInterface):
    default_param_grid = {
        'gradientboostingclassifier__n_estimators': [100, 200],
        'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
        'gradientboostingclassifier__max_depth': [3, 5, 7]
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        if self.context.is_pipeline is False:
            self.model = GradientBoostingClassifier()
        else:
            if param_grid is None:
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                self.context.scaler,
                GradientBoostingClassifier()
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="accuracy"
            )


class ANNClassifierFactory(MLClassifierInterface):
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
            self.model = KerasClassifier(build_fn=self.build_model)
        else:
            if param_grid is None:
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")

            pipeline = make_pipeline(
                self.context.scaler,
                KerasClassifier(build_fn=self.build_model)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
            )
    
    def build_model(self,
                    neuron_layers: tuple = (64, 64),
                    dropout_layers: tuple = (0.2, 0.2),
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
        output_dim = self.context.y_train.shape[1]

        # Determine if the task is binary or multi-class
        num_classes = np.unique(self.context.y_train).shape[0]
        if num_classes > 2:
            output_activation = 'softmax'
        else:
            output_activation = 'sigmoid'

        # Combine neuron and dropout layers
        combined_layers = []
        for neurons, dropout in zip(neuron_layers, dropout_layers):
            combined_layers.append(neurons)
            combined_layers.append(dropout)

        # Add input layer
        model.add(Dense(combined_layers[0], input_dim=input_dim, activation=activation))

        # Add hidden layers
        for i in combined_layers[1:]:
            if i > 0:
                try: # Use error managemant in Dense to raise error further.
                    model.add(Dense(i, activation=activation))
                except Exception as e:
                    raise ValueError(f"An error occurred (layer): {e}") from e
            elif 0 <= i < 1: # dropout layer, cannot be i<=1.
                model.add(Dropout(i))
            else:
                raise ValueError(f"error in layer construction, invalid layer value: {i}\n" +\
                    f"value {i} should be a positive integer or float between 0 and 1.")

        # Add output layer
        model.add(Dense(output_dim, activation=output_activation))

        # Determine loss function based on output activation
        if output_activation == 'softmax':
            loss = 'categorical_crossentropy'
        elif output_activation == 'sigmoid':
            loss = 'binary_crossentropy'
        else:
            raise ValueError(f"Invalid output activation: {output_activation}. Must be 'softmax' or 'sigmoid'.")

        try: # Try to compile the model
            model.compile(loss=loss, optimizer=optimizer)
        except ValueError as e:
            raise ValueError(f"Error compiling model: {e}") from e
        return model

    def train_model(self) -> None:
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train, callbacks=[early_stopping])
