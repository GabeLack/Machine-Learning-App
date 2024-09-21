import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from scikeras.wrappers import KerasClassifier

from interfaces import MLClassifierInterface
from build_model import build_model_factory # to get callable build_model function

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
        'kerasclassifier__batch_size': [16, 32, 64],
        'kerasclassifier__epochs': [50, 100, 200],
        'kerasclassifier__optimizer': ['adam', 'rmsprop'],
        'kerasclassifier__neuron_layers': [ # shapes of the hidden layers
            (64, 64, 64),  # rectangle
            (32, 64, 128),  # expanding_triangle
            (128, 64, 32),  # diminishing_triangle
            (128, 32, 128),  # squeezed center
            (32, 128, 32)  # enlarged center
        ],
        'kerasclassifier__dropout_layers': [
            (0, 0, 0), # no dropout
            (0.1, 0.1, 0.2),
            (0.2, 0.2, 0.2),
            (0.3, 0.3, 0.3),
            (0.1, 0.2, 0.3)
        ]
    }

    def create_model(self, param_grid: dict|None = None) -> None:
        # Get the input and output dimensions
        input_dim = self.context.X_train.shape[1]
        # Check if the output layer should be a softmax or sigmoid
        num_classes = len(set(self.context.y_train))
        if num_classes > 2:
            output_activation = 'softmax'
            output_dim = num_classes
        else:
            output_activation = 'sigmoid'
            output_dim = 1

        if self.context.is_pipeline is False:
            self.model = KerasClassifier(build_fn=build_model_factory(input_dim, output_activation, output_dim))
        else:
            if param_grid is None:  # use the default param_grid
                param_grid = self.default_param_grid
            if not isinstance(param_grid, dict) or not param_grid:
                raise ValueError("param_grid must be a non-empty dictionary.")
            
            pipeline = make_pipeline(
                self.context.scaler,
                KerasClassifier(
                    build_fn=build_model_factory(input_dim, output_activation, output_dim),
                    neuron_layers=(64, 64), # just so param_grid can be used, these specific inputs are not used
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
