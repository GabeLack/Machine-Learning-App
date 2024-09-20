"""
NAME
    factory

DESCRIPTION
    This module provides classes for creating machine learning models using a factory pattern.
    It includes enums for problem types and model types, and a factory class for creating models.

CLASSES
    ProblemType
        Enum representing the type of problem (classification or regression).

    ModelType
        Enum representing the type of model to be created.

    ModelFactory
        Factory class for creating machine learning models.

        Methods defined here:
        
        create_model(self, model_type: ModelType, problem_type: ProblemType, model_context: ModelContext) -> object
            Creates a machine learning model based on the specified type and context.
"""

from enum import Enum, auto
from classifiers import (LogisticFactory, SVCFactory, RandomForestFactory, 
                         KNNFactory, GradientBoostingFactory, ANNClassifierFactory)
from regressors import LinearFactory, ElasticNetFactory, SVRFactory, ANNRegressorFactory
from context import ModelContext

class ProblemType(Enum):
    """
    Enum representing the type of problem (classification or regression).
    
    Attributes:
        CLASSIFICATION: Represents a classification problem.
        REGRESSION: Represents a regression problem.
    """
    
    CLASSIFICATION = auto()
    REGRESSION = auto()


class ModelType(Enum):
    """
    Enum representing the type of model to be created.
    
    Attributes:
        LOGISTIC: Represents a logistic regression model.
        SVC: Represents a support vector classifier.
        RANDOMFOREST: Represents a random forest classifier.
        KNEARESTNEIGHBORS: Represents a k-nearest neighbors classifier.
        GRADIENTBOOSTING: Represents a gradient boosting classifier.
        ANNCLASSIFIER: Represents an artificial neural network classifier.
        LINEAR: Represents a linear regression model.
        ELASTICNET: Represents an elastic net regression model.
        SVR: Represents a support vector regressor.
        ANNREGRESSOR: Represents an artificial neural network regressor.
    """
    
    # Classifiers
    LOGISTIC = auto()
    SVC = auto()
    RANDOMFOREST = auto()
    KNEARESTNEIGHBORS = auto()
    GRADIENTBOOSTING = auto()
    ANNCLASSIFIER = auto()
    # Regressors
    LINEAR = auto()
    ELASTICNET = auto()
    SVR = auto()
    ANNREGRESSOR = auto()


class ModelFactory:
    """
    Factory class for creating machine learning models.
    
    Methods:
        create_model: Creates a machine learning model based on the specified type and context.
    """
    
    def create_model(self,
                     model_type: ModelType,
                     problem_type: ProblemType,
                     model_context: ModelContext) -> object:
        """
        Creates a machine learning model based on the specified type and context.
        
        Args:
            model_type (ModelType): The type of model to create.
            problem_type (ProblemType): The type of problem (classification or regression).
            model_context (ModelContext): The context in which the model will be created.
        
        Returns:
            object: An instance of the created model.
        
        Raises:
            ValueError: If an invalid model type or problem type is specified.
        """
        
        if problem_type == ProblemType.CLASSIFICATION:
            if model_type == ModelType.LOGISTIC:
                logistic = LogisticFactory(model_context)
                logistic.create_model()
                return logistic
            elif model_type == ModelType.SVC:
                svc = SVCFactory(model_context)
                svc.create_model()
                return svc
            elif model_type == ModelType.RANDOMFOREST:
                random_forest = RandomForestFactory(model_context)
                random_forest.create_model()
                return random_forest
            elif model_type == ModelType.KNEARESTNEIGHBORS:
                knn = KNNFactory(model_context)
                knn.create_model()
                return knn
            elif model_type == ModelType.GRADIENTBOOSTING:
                gradient_boosting = GradientBoostingFactory(model_context)
                gradient_boosting.create_model()
                return gradient_boosting
            elif model_type == ModelType.ANNCLASSIFIER:
                ann = ANNClassifierFactory(model_context)
                ann.create_model()
                return ann
            else:
                raise ValueError("Invalid model type")
        elif problem_type == ProblemType.REGRESSION:
            if model_type == ModelType.LINEAR:
                linear = LinearFactory(model_context)
                linear.create_model()
                return linear
            elif model_type == ModelType.ELASTICNET: # ElasticNet includes Ridge and Lasso in GridSearchCV
                elastic_net = ElasticNetFactory(model_context)
                elastic_net.create_model()
                return elastic_net
            elif model_type == ModelType.SVR:
                svr = SVRFactory(model_context)
                svr.create_model()
                return svr
            elif model_type == ModelType.ANNREGRESSOR:
                ann = ANNRegressorFactory(model_context)
                ann.create_model()
                return ann
            else:
                raise ValueError("Invalid model type")
        else:
            raise ValueError("Invalid problem type")
