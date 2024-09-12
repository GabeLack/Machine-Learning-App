import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum,auto

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import (LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, f1_score,
                             classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)

from Regressors import LinearFactory, ElasticNetFactory, SVRFactory
from Classifiers import LogisticFactory, SVCFactory, RandomForestFactory, KNNFactory, GradientBoostingFactory

class ProblemType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()

class ModelType(Enum):
    # Classifiers
    LOGISTIC = auto()
    SVC = auto()
    RANDOMFOREST = auto()
    KNEARESTNEIGHBORS = auto()
    GRADIENTBOOSTING = auto()
    # Regressors
    LINEAR = auto()
    ELASTICNET = auto()
    SVR = auto()

class ModelContext:
    def __init__(self,
                 data_file_path:str,
                 target_column:str,
                 model_type:str,
                 problem_type:str,
                 test_size:float=0.3,
                 is_pipeline:bool=False,
                 scalar=StandardScaler()):
        self.data_file_path = data_file_path
        self.target_column = target_column
        self.test_size = test_size
        self.model_type = model_type
        self.problem_type = problem_type
        self.is_pipeline = is_pipeline
        self.scaler = scalar
        self.model = ModelFactory().get_model(model_type, problem_type)
    
class ModelFactory:
    def get_model(self, model_type:ModelType, problem_type:ProblemType):
        if problem_type == problem_type.CLASSIFICATION:
            if model_type == model_type.LOGISTIC:
                return LogisticFactory()
            elif model_type == model_type.SVC:
                return SVCFactory()
            elif model_type == model_type.RANDOMFOREST:
                return RandomForestFactory()
            elif model_type == model_type.KNEARESTNEIGHBORS:
                return KNNFactory()
            elif model_type == model_type.GRADIENTBOOSTING:
                return GradientBoostingFactory
            else:
                raise ValueError("Invalid model type")
        elif problem_type == problem_type.REGRESSION:
            if model_type == model_type.LINEAR:
                return LinearFactory()
            elif model_type == model_type.ELASTICNET:
                return ElasticNetFactory()
            elif model_type == model_type.SVR:
                return SVRFactory()
            else:
                raise ValueError("Invalid model type")
        else:
            raise ValueError("Invalid problem type")
