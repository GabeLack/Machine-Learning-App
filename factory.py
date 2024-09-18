from enum import Enum, auto
from classifiers import (LogisticFactory, SVCFactory, RandomForestFactory, 
                         KNNFactory, GradientBoostingFactory, ANNClassifierFactory)
from regressors import LinearFactory, ElasticNetFactory, SVRFactory, ANNRegressorFactory
from context import ModelContext

class ProblemType(Enum):
    """Enum class representing different types of machine learning problems.

    Attributes:
        CLASSIFICATION (auto): Represents a classification problem where the goal is to predict discrete labels.
        REGRESSION (auto): Represents a regression problem where the goal is to predict continuous values.
    """
    
    CLASSIFICATION = auto()
    REGRESSION = auto()


class ModelType(Enum):
    """ModelType is an enumeration that defines various types of machine learning models.

    Attributes:
        LOGISTIC (auto): Logistic Regression classifier.
        SVC (auto): Support Vector Classifier.
        RANDOMFOREST (auto): Random Forest classifier.
        KNEARESTNEIGHBORS (auto): K-Nearest Neighbors classifier.
        GRADIENTBOOSTING (auto): Gradient Boosting classifier.
        ANNCLASSIFIER (auto): Artificial Neural Network classifier.
        LINEAR (auto): Linear Regression model.
        ELASTICNET (auto): Elastic Net regression model.
        SVR (auto): Support Vector Regressor.
        ANNREGRESSOR (auto): Artificial Neural Network regressor.
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
    def create_model(self,
                     model_type: ModelType,
                     problem_type: ProblemType,
                     model_context: ModelContext) -> object:
        """Creates and returns a machine learning model based on the specified model, problem, and context.
        
        Parameters:
        model_type (ModelType): The type of model to create (e.g., LOGISTIC, SVC, RANDOMFOREST, etc.).
        problem_type (ProblemType): The type of problem to solve (e.g., CLASSIFICATION, REGRESSION).
        model_context (ModelContext): The context in which the model will be used, providing necessary configuration and data.
        Returns:
        object: An instance of the created model factory, which has already created the model.
        Raises:
        ValueError: If an invalid model type or problem type is provided.
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
