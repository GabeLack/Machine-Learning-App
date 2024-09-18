import unittest
from factory import ModelFactory, ModelType, ProblemType
from context import ModelContext
import pandas as pd

class TestProblemType(unittest.TestCase):

    def test_enum_members(self):
        self.assertTrue(hasattr(ProblemType, 'CLASSIFICATION'))
        self.assertTrue(hasattr(ProblemType, 'REGRESSION'))


class TestModelType(unittest.TestCase):

    def test_enum_members(self):
        self.assertTrue(hasattr(ModelType, 'LOGISTIC'))
        self.assertTrue(hasattr(ModelType, 'SVC'))
        self.assertTrue(hasattr(ModelType, 'RANDOMFOREST'))
        self.assertTrue(hasattr(ModelType, 'KNEARESTNEIGHBORS'))
        self.assertTrue(hasattr(ModelType, 'GRADIENTBOOSTING'))
        self.assertTrue(hasattr(ModelType, 'ANNCLASSIFIER'))
        self.assertTrue(hasattr(ModelType, 'LINEAR'))
        self.assertTrue(hasattr(ModelType, 'ELASTICNET'))
        self.assertTrue(hasattr(ModelType, 'SVR'))
        self.assertTrue(hasattr(ModelType, 'ANNREGRESSOR'))


class TestModelFactory(unittest.TestCase):

    def setUp(self) -> None:
        # Context tested in its own file
        df_regression = pd.read_csv('test_csv/regression_data.csv')
        self.context_regression = ModelContext(df_regression, 'target')

        df_classification = pd.read_csv('test_csv/binary_classification_data.csv')
        self.context_classification = ModelContext(df_classification, 'target')

    def test_modelfactory(self):
        model_factory = ModelFactory()
        self.assertIsNotNone(model_factory)
        self.assertTrue(isinstance(model_factory, ModelFactory))

    def test_create_model_all_types_classification(self):
        model_factory = ModelFactory()
        for model_type in [ModelType.LOGISTIC,
                           ModelType.SVC,
                           ModelType.RANDOMFOREST,
                           ModelType.KNEARESTNEIGHBORS,
                           ModelType.GRADIENTBOOSTING,
                           ModelType.ANNCLASSIFIER]:
            model = model_factory.create_model(model_type,
                                               ProblemType.CLASSIFICATION,
                                               self.context_classification)
            self.assertIsNotNone(model)

    def test_create_model_all_types_regression(self):
        model_factory = ModelFactory()
        for model_type in [ModelType.LINEAR,
                           ModelType.ELASTICNET,
                           ModelType.SVR,
                           ModelType.ANNREGRESSOR]:
            model = model_factory.create_model(model_type,
                                               ProblemType.REGRESSION,
                                               self.context_regression)
            self.assertIsNotNone(model)

    def test_create_model_invalid_model_type_classification(self):
        model_factory = ModelFactory()

        with self.assertRaises(ValueError):
            model_factory.create_model(ModelType.LINEAR,
                                       ProblemType.CLASSIFICATION,
                                       self.context_classification)

    def test_create_model_invalid_model_type_regression(self):
        model_factory = ModelFactory()

        with self.assertRaises(ValueError):
            model_factory.create_model(ModelType.LOGISTIC,
                                       ProblemType.REGRESSION,
                                       self.context_regression)

    def test_create_model_invalid_problem_type(self):
        model_factory = ModelFactory()

        with self.assertRaises(ValueError):
            model_factory.create_model(ModelType.LOGISTIC,
                                       'INVALID',
                                       self.context_classification)

    def test_create_model_invalid_context(self):
        model_factory = ModelFactory()
        invalid_context = None
        with self.assertRaises(TypeError):
            model_factory.create_model(ModelType.LOGISTIC,
                                       ProblemType.CLASSIFICATION,
                                       invalid_context)

if __name__ == '__main__':
    unittest.main()
