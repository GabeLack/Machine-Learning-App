from enum import Enum, auto
from classifiers import LogisticFactory, SVCFactory, RandomForestFactory, KNNFactory, GradientBoostingFactory
from regressors import LinearFactory, ElasticNetFactory, SVRFactory
from context import ModelContext


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


class ModelFactory:
    def create_model(self, model_type: ModelType, problem_type: ProblemType, model_context: ModelContext):
        if problem_type == ProblemType.CLASSIFICATION:
            if model_type == ModelType.LOGISTIC:
                logistic = LogisticFactory(model_context)
                return logistic.create_model()
            elif model_type == ModelType.SVC:
                svc = SVCFactory(model_context)
                return svc.create_model()
            elif model_type == ModelType.RANDOMFOREST:
                random_forest = RandomForestFactory(model_context)
                return random_forest.create_model()
            elif model_type == ModelType.KNEARESTNEIGHBORS:
                knn = KNNFactory(model_context)
                return knn.create_model()
            elif model_type == ModelType.GRADIENTBOOSTING:
                gradient_boosting = GradientBoostingFactory(model_context)
                return gradient_boosting.create_model()
            else:
                raise ValueError("Invalid model type")
        elif problem_type == ProblemType.REGRESSION:
            if model_type == ModelType.LINEAR:
                linear = LinearFactory(model_context)
                return linear.create_model()
            elif model_type == ModelType.ELASTICNET:
                elastic_net = ElasticNetFactory(model_context)
                return elastic_net.create_model()
            elif model_type == ModelType.SVR:
                svr = SVRFactory(model_context)
                return svr.create_model()
            else:
                raise ValueError("Invalid model type")
        else:
            raise ValueError("Invalid problem type")
