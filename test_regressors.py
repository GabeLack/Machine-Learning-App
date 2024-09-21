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
        ("invalid_param_name", {'invalid_param': [1, 2, 3]}),
        ("invalid_none", None)
    ])
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = ANNRegressorFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_build_model(self):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        model = factory.build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 5)  # 4 Dense layers and 2 Dropout layers
        self.assertEqual(model.loss, 'mean_squared_error') # Default loss function

    def test_build_model_custom_params(self):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        model = factory.build_model(neuron_layers=(128, 64, 32),
                                    dropout_layers=(0.1, 0.2, 0.3),
                                    activation='tanh',
                                    optimizer='rmsprop')
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 7)  # 4 Dense layers (including output) and 3 Dropout layers

    @parameterized.expand([
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_float", 1.23),
        ("invalid_list", [64, 64]),
        ("invalid_dict", {'layer1': 64, 'layer2': 64}),
        ("invalid_none", None)
    ])
    def test_build_model_neuron_layers(self, name, invalid_input):
        # Only valid input is a tuple of integers
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        with self.assertRaises(ValueError):
            factory.build_model(neuron_layers=invalid_input)

    @parameterized.expand([
        ("invalid_string_in_tuple", ('invalid',)),
        ("invalid_float_in_tuple", (1.23,)),
        ("invalid_list_in_tuple", ([64, 64],)),
        ("invalid_dict_in_tuple", ({'layer1': 64},)),
        ("invalid_none_in_tuple", (None,))
    ])
    def test_build_model_tuple_neuron_layers(self, name, invalid_input):
        # Only the tuple input is tested here because the individual elements are tested in the previous test
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        with self.assertRaises(ValueError):
            factory.build_model(neuron_layers=invalid_input)

    @parameterized.expand([
        ("invalid_string", 'invalid'),
        ("invalid_int", 1),
        ("invalid_list", [0.2, 0.2]),
        ("invalid_dict", {'layer1': 0.2, 'layer2': 0.2}),
        ("invalid_none", None)
    ])
    def test_build_model_dropout_layers(self, name, invalid_input):
        # Only valid input is a tuple of floats
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        with self.assertRaises(ValueError):
            factory.build_model(dropout_layers=invalid_input)

    @parameterized.expand([
        ("invalid_string_in_tuple", ('invalid',)),
        ("invalid_int_in_tuple", (1,)),
        ("invalid_list_in_tuple", ([0.2, 0.2],)),
        ("invalid_dict_in_tuple", ({'layer1': 0.2},)),
        ("invalid_none_in_tuple", (None,))
    ])
    def test_build_model_tuple_dropout_layers(self, name, invalid_input):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        with self.assertRaises(ValueError):
            factory.build_model(dropout_layers=invalid_input)

    @parameterized.expand([
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_list", ['relu', 'relu']),
        ("invalid_dict", {'layer1': 'relu', 'layer2': 'relu'})
    ])
    def test_build_model_activation(self, name, invalid_input):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        with self.assertRaises(ValueError):
            # Invalid activation function
            model = factory.build_model(activation=invalid_input)

    @parameterized.expand([
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_list", ['invalid',]),
        ("invalid_tuple", ('invalid',)),
        ("invalid_dict", {'layer1': 'invalid', 'layer2': 'invalid'})
    ])
    def test_build_model_optimizer(self, name, invalid_input):
        mock_context = ModelContext(self.df_regression, 'target', scaler=Normalizer())
        factory = ANNRegressorFactory(mock_context)
        with self.assertRaises(ValueError):
            # Invalid optimizer
            model = factory.build_model(optimizer=invalid_input)

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
        
        factory.train_model()
        self.assertTrue(hasattr(factory, 'model'))
        # Check if the model's best parameters and best score are not None after training
        self.assertIsNotNone(factory.model.best_params_)
        self.assertIsNotNone(factory.model.best_score_)

if __name__ == '__main__':
    unittest.main()