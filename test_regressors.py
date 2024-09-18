import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures, RobustScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor

from regressors import LinearFactory, ElasticNetFactory, SVRFactory, ANNRegressorFactory
from context import ModelContext

class TestRegressors(unittest.TestCase):

    def setUp(self):
        # Mock context with necessary attributes
        self.df_regression = pd.read_csv('test_csv/regression_data.csv')
        self.mock_context = ModelContext(self.df_regression, 'target')

class TestLinearFactory(TestRegressors):
    def test_factory(self):
        factory = LinearFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], LinearRegression)

    def test_factory_no_pipeline(self):
        # Pipeline is technically already tested in test_context, so we just need a simple check here
        no_pipeline_context = ModelContext(self.df_regression, 'target', is_pipeline=False)
        factory = LinearFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, LinearRegression)

    def test_factory_nondefault_scaler(self):
        # Alternate scalers are tested in test_context, so we just need a simple check here
        diff_scaler_context = ModelContext(self.df_regression, 'target', scaler=RobustScaler())
        factory = LinearFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = LinearFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = LinearFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            # This should raise an error because the pipeline should fail to fit the model
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = LinearFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = LinearFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestElasticNetFactory(TestRegressors):
    def test_factory(self):
        factory = ElasticNetFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], ElasticNet)

    def test_factory_no_pipeline(self):
        factory = ElasticNetFactory(self.mock_context)
        factory.context.is_pipeline = False
        factory.create_model()
        self.assertIsInstance(factory.model, ElasticNet)

    def test_factory_nondefault_scaler(self):
        factory = ElasticNetFactory(self.mock_context)
        factory.context.scaler = RobustScaler()
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

    def tes_factory_str_paramgrid(self):
        factory = ElasticNetFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = ElasticNetFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            # This should raise an error because the pipeline should fail to fit the model
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = ElasticNetFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = ElasticNetFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestSVRFactory(TestRegressors):
    def testfactory(self):
        factory = SVRFactory(self.mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], SVR)

    def test_factory_no_pipeline(self):
        factory = SVRFactory(self.mock_context)
        factory.context.is_pipeline = False
        factory.create_model()
        self.assertIsInstance(factory.model, SVR)

    def test_factory_nondefault_scaler(self):
        factory = SVRFactory(self.mock_context)
        factory.context.scaler = RobustScaler()
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

    def test_factory_str_paramgrid(self):
        factory = SVRFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = SVRFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            # This should raise an error because the pipeline should fail to fit the model
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = SVRFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = SVRFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)


class TestANNRegressorFactory(TestRegressors):
    def test_regressor_factory(self):
        # Correct scaler for ANN is Normalizer
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], Normalizer)
        self.assertIsInstance(factory.model.estimator.steps[1][1], KerasRegressor)
        self.assertTrue(callable(factory.model.estimator.steps[1][1].build_fn))

    def test_build_model(self):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        model = factory.build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 5)  # 3 Dense layers and 2 Dropout layers
        self.assertEqual(model.loss, 'mean_squared_error') # Default loss function

    def test_factory_str_paramgrid(self):
        # scaler doesn't matter here so didnt bother replacing it
        factory = ANNRegressorFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid='invalid')

    def test_factory_invalid_paramgrid(self):
        factory = ANNRegressorFactory(self.mock_context)
        invalid_param_grid = {'invalid_param': [1, 2, 3]}  # Invalid parameter name
        factory.create_model(param_grid=invalid_param_grid)
        with self.assertRaises(ValueError):
            # This should raise an error because the pipeline should fail to fit the model
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = ANNRegressorFactory(self.mock_context)
        factory.create_model(param_grid={})
        self.assertIsInstance(factory.model, GridSearchCV)

    def test_factory_custom_paramgrid(self):
        factory = ANNRegressorFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

if __name__ == '__main__':
    unittest.main()