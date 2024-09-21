from parameterized import parameterized
import unittest
from unittest.mock import patch, PropertyMock
import pandas as pd

from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures, RobustScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

from regressors import LinearFactory, ElasticNetFactory, SVRFactory, ANNRegressorFactory
from context import ModelContext

def invalid_param_grids():
    return [
        ("invalid_string", 'str'),
        ("invalid_float", 1.23),
        ("invalid_int", 123),
        ("invalid_tuple", (1, 2, 3)),
        ("invalid_list", [1, 2, 3]),
        ("invalid_empty_dict", {})
    ]

def invalid_param_grids_for_fit():
    return [
        ("invalid_param_name", {'invalid_param': [1, 2, 3]})
    ]

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

    def test_factory_custom_paramgrid(self):
        factory = LinearFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = LinearFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = LinearFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            # This should raise an error because the pipeline should fail to fit the model
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)


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

    def test_factory_custom_paramgrid(self):
        factory = ElasticNetFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = ElasticNetFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = ElasticNetFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)


class TestSVRFactory(TestRegressors):
    def test_factory(self):
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

    def test_factory_custom_paramgrid(self):
        factory = SVRFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = SVRFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = SVRFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)


class TestANNRegressorFactory(TestRegressors):
    def test_factory(self):
        # Correct scaler for ANN is Normalizer
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], Normalizer)
        self.assertIsInstance(factory.model.estimator.steps[1][1], KerasRegressor)
        self.assertTrue(callable(factory.model.estimator.steps[1][1].build_fn))

    def test_factory_custom_paramgrid(self):
        factory = ANNRegressorFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = ANNRegressorFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand([
        ("invalid_param_name", {'invalid_param': [1, 2, 3]})
    ])
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = ANNRegressorFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_train_model(self):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)

        # Smaller param grid for faster testing
        small_param_grid = {
            'kerasregressor__batch_size': [16],
            'kerasregressor__epochs': [10],
            'kerasregressor__optimizer': ['adam'],
            'kerasregressor__neuron_layers': [(32, 32)],
            'kerasregressor__dropout_layers': [(0.1, 0.1)]
        }
        factory.create_model(param_grid=small_param_grid)
        #! by all rights this method SHOULD WORK, I even have a working example in kerasregressor.py
        #! however it refuses to work within the ANNRegressorFactory class, and the error message
        #! is not helpful at all. I'm tired, KerasClassifier probably has the same problem and I
        #! havn't even bothered dealing with it yet. Not enough time to fix the bug, even if it's
        #! right there.

        # this test somehow passed now, after putting the build_model method in a method of its own so
        # it remains a callable (yet I can still send in input_dim). By all rights it should've worked
        # when it was part of the ANNRegressorFactory class too, but it didn't, so whatever, now it works.

        # In the fixing process also ended up removing EarlyStopping from the model, which is a shame, but
        # apparently it gets too funky with GridSearchCV, so I've opted to just not having it.
        
        #* I'm leaving my rambling here cause it was funny in retrospect.

        factory.train_model()
        self.assertTrue(hasattr(factory, 'model'))
        # Check if the model's best parameters and best score are not None after training
        self.assertIsNotNone(factory.model.best_params_)
        self.assertIsNotNone(factory.model.best_score_)

if __name__ == '__main__':
    unittest.main()