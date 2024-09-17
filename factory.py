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
            else:
                raise ValueError("Invalid model type")
        elif problem_type == ProblemType.REGRESSION:
            if model_type == ModelType.LINEAR:
                linear = LinearFactory(model_context)
                linear.create_model()
                return linear
            elif model_type == ModelType.ELASTICNET:
                elastic_net = ElasticNetFactory(model_context)
                elastic_net.create_model()
                return elastic_net
            elif model_type == ModelType.SVR:
                svr = SVRFactory(model_context)
                svr.create_model()
                return svr
            else:
                raise ValueError("Invalid model type")
        else:
            raise ValueError("Invalid problem type")
