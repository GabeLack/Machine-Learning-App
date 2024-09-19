from parameterized import parameterized
import unittest
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

    def test_factory_custom_paramgrid(self):
        factory = LogisticFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = LogisticFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = LogisticFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

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

    def test_factory_custom_paramgrid(self):
        factory = SVCFactory(self.mock_context)
        factory.create_model(param_grid={'svc__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = SVCFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = SVCFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

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

    def test_factory_custom_paramgrid(self):
        factory = RandomForestFactory(self.mock_context)
        factory.create_model(param_grid={'randomforestclassifier__n_estimators': [100, 200]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = RandomForestFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = RandomForestFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

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

    def test_factory_custom_paramgrid(self):
        factory = KNNFactory(self.mock_context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = KNNFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = KNNFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)


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

    @parameterized.expand(invalid_param_grids())
    def test_factory_invalid_paramgrid(self, name, invalid_param_grid):
        factory = GradientBoostingFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand(invalid_param_grids_for_fit())
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = GradientBoostingFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_factory_empty_paramgrid(self):
        factory = GradientBoostingFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid={})

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

    def test_factory_custom_paramgrid(self):
        factory = ANNClassifierFactory(self.mock_context)
        factory.create_model(param_grid={'batch_size': [16, 32, 64]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(invalid_param_grids())
    def test_factory_paramgrid(self, name, invalid_param_grid):
        factory = ANNClassifierFactory(self.mock_context)
        with self.assertRaises(ValueError):
            factory.create_model(param_grid=invalid_param_grid)

    @parameterized.expand([
        ("invalid_param_name", {'invalid_param': [1, 2, 3]}),
        ("invalid_none", None)
    ])
    def test_factory_paramgrid_for_fit(self, name, bad_param_grid):
        factory = ANNClassifierFactory(self.mock_context)
        factory.create_model(param_grid=bad_param_grid)
        with self.assertRaises(ValueError):
            factory.model.fit(self.mock_context.X_train, self.mock_context.y_train)

    def test_build_model_custom_params(self):
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
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
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
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
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
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
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
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
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
        with self.assertRaises(ValueError):
            factory.build_model(dropout_layers=invalid_input)

    @parameterized.expand([
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_list", ['relu', 'relu']),
        ("invalid_dict", {'layer1': 'relu', 'layer2': 'relu'})
    ])
    def test_build_model_activation(self, name, invalid_input):
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
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
        mock_context = ModelContext(self.df_classification, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
        with self.assertRaises(ValueError):
            # Invalid optimizer
            model = factory.build_model(optimizer=invalid_input)

if __name__ == '__main__':
    unittest.main()