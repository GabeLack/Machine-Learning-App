import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from Interfaces import MLClassifierInterface

class LogisticFactory(MLClassifierInterface):
    default_param_grid = {
                'polynomialfeatures__degree': np.arange(1,6), # degrees of polynomial features
                'logisticregression__C': np.logspace(0,2,5), # regularization strength
                'logisticregression__penalty': ['l1','l2','elasticnet'], # regularization type
                'logisticregression__solver': ['lbfgs','liblinear','newton-cg', 
                                               'newton-cholesky','saga'], # optimization algorithm
                'logisticregression__class_weight': [None,'balanced'] # class weights
                }

    def create_model(self, param_grid:dict=None, **kwargs):

        if self.context.is_pipeline is False: # use a basic model
            self.model = LogisticRegression(**kwargs) #! needs validation of kwargs

        else: # use the polyfeatures-scaler-estimator pipeline in a gridsearch
            if param_grid is None: # use the default param_grid
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler, # StandardScaler() or MinMaxScaler() or RobustScaler()
                LogisticRegression(**kwargs)
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

    def create_model(self, param_grid:dict=None, **kwargs):
        if self.context.is_pipeline is False:
            self.model = SVC(**kwargs) #! needs validation of kwargs
        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                self.context.scaler,
                SVC(**kwargs)
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

    def create_model(self, param_grid:dict=None, **kwargs):
        if self.context.is_pipeline is False:
            self.model = RandomForestClassifier(**kwargs) #! needs validation of kwargs
        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                self.context.scaler,
                RandomForestClassifier(**kwargs)
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

    def create_model(self, param_grid:dict=None, **kwargs):
        if self.context.is_pipeline is False:
            self.model = KNeighborsClassifier(**kwargs) #! needs validation of kwargs
        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                PolynomialFeatures(include_bias=False),
                self.context.scaler,
                KNeighborsClassifier(**kwargs)
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

    def create_model(self, param_grid:dict=None, **kwargs):
        if self.context.is_pipeline is False:
            self.model = GradientBoostingClassifier(**kwargs) #! needs validation of kwargs
        else:
            if param_grid is None:
                param_grid = self.default_param_grid

            pipeline = make_pipeline(
                self.context.scaler,
                GradientBoostingClassifier(**kwargs)
            )

            self.model = GridSearchCV(
                pipeline,
                param_grid,
                n_jobs=4,
                cv=10,
                scoring="accuracy"
            )
