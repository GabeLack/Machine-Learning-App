import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasClassifier

from classifiers import LogisticFactory, SVCFactory, KNNFactory, GradientBoostingFactory, RandomForestFactory, ANNClassifierFactory
from context import ModelContext

class TestClassifiers(unittest.TestCase):

    def setUp(self):
        # Mock context with necessary attributes
        self.df_classification = pd.read_csv('test_csv/binary_classification_data.csv')
        self.mock_context = ModelContext(self.df_classification, 'target')

class TestLogisticFactory(TestClassifiers):
    def test_factory(self):
        factory = LogisticFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], LogisticRegression)

    def test_factory_no_pipeline(self):
        no_pipeline_context = ModelContext(self.df_classification, 'target', is_pipeline=False)
        factory = LogisticFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, LogisticRegression)

    def test_factory_nondefault_scaler(self):
        diff_scaler_context = ModelContext(self.df_classification, 'target', scaler=RobustScaler())
        factory = LogisticFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = LogisticFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = LogisticFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = LogisticFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = LogisticFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestSVCFactory(TestClassifiers):
    def test_factory(self):
        factory = SVCFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], SVC)

    def test_factory_no_pipeline(self):
        no_pipeline_context = ModelContext(self.df_classification, 'target', is_pipeline=False)
        factory = SVCFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, SVC)

    def test_factory_nondefault_scaler(self):
        diff_scaler_context = ModelContext(self.df_classification, 'target', scaler=RobustScaler())
        factory = SVCFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = SVCFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = SVCFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = SVCFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = SVCFactory(self.mock_context)
        factory.create_model(param_grid={'svc__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestRandomForestFactory(TestClassifiers):
    def test_factory(self):
        factory = RandomForestFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RandomForestClassifier)

    def test_factory_no_pipeline(self):
        no_pipeline_context = ModelContext(self.df_classification, 'target', is_pipeline=False)
        factory = RandomForestFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, RandomForestClassifier)

    def test_factory_nondefault_scaler(self):
        diff_scaler_context = ModelContext(self.df_classification, 'target', scaler=RobustScaler())
        factory = RandomForestFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = RandomForestFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = RandomForestFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = RandomForestFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = RandomForestFactory(self.mock_context)
        factory.create_model(param_grid={'randomforestclassifier__n_estimators': [100, 200]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestKNNFactory(TestClassifiers):
    def test_factory(self):
        factory = KNNFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], KNeighborsClassifier)

    def test_factory_no_pipeline(self):
        no_pipeline_context = ModelContext(self.df_classification, 'target', is_pipeline=False)
        factory = KNNFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, KNeighborsClassifier)

    def test_factory_nondefault_scaler(self):
        diff_scaler_context = ModelContext(self.df_classification, 'target', scaler=RobustScaler())
        factory = KNNFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = KNNFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = KNNFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = KNNFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = KNNFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestGradientBoostingFactory(TestClassifiers):
    def test_factory(self):
        factory = GradientBoostingFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], GradientBoostingClassifier)

    def test_factory_no_pipeline(self):
        no_pipeline_context = ModelContext(self.df_classification, 'target', is_pipeline=False)
        factory = GradientBoostingFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GradientBoostingClassifier)

    def test_factory_nondefault_scaler(self):
        diff_scaler_context = ModelContext(self.df_classification, 'target', scaler=RobustScaler())
        factory = GradientBoostingFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = GradientBoostingFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = GradientBoostingFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = GradientBoostingFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = GradientBoostingFactory(self.mock_context)
        factory.create_model(param_grid={'gradientboostingclassifier__n_estimators': [100, 200]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestANNClassifierFactory(TestClassifiers):
    def test_classifier_factory(self):
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], Normalizer)
        self.assertIsInstance(factory.model.estimator.steps[1][1], KerasClassifier)
        self.assertTrue(callable(factory.model.estimator.steps[1][1].build_fn))

    def test_build_model(self):
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
        model = factory.build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 5)  # 3 Dense layers and 2 Dropout layers
        self.assertEqual(model.loss, 'binary_crossentropy')  # Default loss function for classification

    def test_factory_str_paramgrid(self):
        # scaler doesn't matter here so didnt bother replacing it
        factory = ANNClassifierFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = ANNClassifierFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = ANNClassifierFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = ANNClassifierFactory(self.mock_context)
        factory.create_model(param_grid={'batch_size': [16, 32, 64]})
        self.assertIsInstance(factory.model, GridSearchCV)

if __name__ == '__main__':
    unittest.main()