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


def contexts():
    return [
        ("binary_context", ModelContext(pd.read_csv('test_csv/binary_classification_data.csv'), 'target')),
        ("multi_context", ModelContext(pd.read_csv('test_csv/multi_classification_data.csv'), 'target'))
    ]

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
    # Not a strictly necessary method, but it makes the test cases easier to read and modify in the future
    # for other cases where the param_grid should be invalid
    return [
        ("invalid_param_name", {'invalid_param': [1, 2, 3]})
    ]


class TestClassifiers(unittest.TestCase):

    def setUp(self):
        # Load binary classification data
        self.df_binary_classification = pd.read_csv('test_csv/binary_classification_data.csv')
        self.binary_context = ModelContext(self.df_binary_classification, 'target')

        # Load multi-class classification data
        self.df_multi_classification = pd.read_csv('test_csv/multi_classification_data.csv')
        self.multi_context = ModelContext(self.df_multi_classification, 'target')

class TestLogisticFactory(TestClassifiers):
    @parameterized.expand(contexts())
    def test_create_model(self, name, context):
        factory = LogisticFactory(context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], LogisticRegression)

    @parameterized.expand(contexts())
    def test_create_model_no_pipeline(self, name, context):
        no_pipeline_context = ModelContext(context.df, 'target', is_pipeline=False)
        factory = LogisticFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, LogisticRegression)

    @parameterized.expand(contexts())
    def test_create_model_nondefault_scaler(self, name, context):
        diff_scaler_context = ModelContext(context.df, 'target', scaler=RobustScaler())
        factory = LogisticFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

    @parameterized.expand(contexts())
    def test_create_model_custom_paramgrid(self, name, context):
        factory = LogisticFactory(context)
        factory.create_model(param_grid={'polynomialfeatures__degree': [1, 2, 3]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(contexts())
    def test_create_model_invalid_paramgrid(self, name, context):
        invalid_param_grids = [
            ("invalid_string", 'str'),
            ("invalid_float", 1.23),
            ("invalid_int", 123),
            ("invalid_tuple", (1, 2, 3)),
            ("invalid_list", [1, 2, 3]),
            ("invalid_empty_dict", {})
        ]
        for invalid_param_grid in invalid_param_grids:
            with self.subTest(invalid_param_grid=invalid_param_grid):
                factory = LogisticFactory(context)
                with self.assertRaises(ValueError):
                    factory.create_model(param_grid=invalid_param_grid[1])

    @parameterized.expand(contexts())
    def test_create_model_paramgrid_for_fit(self, name, context):
        invalid_param_grids_for_fit = [
            ("invalid_param_name", {'invalid_param': [1, 2, 3]})
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = LogisticFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)


class TestSVCFactory(TestClassifiers):
    @parameterized.expand(contexts())
    def test_create_model(self, name, context):
        factory = SVCFactory(context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], SVC)

    @parameterized.expand(contexts())
    def test_create_model_no_pipeline(self, name, context):
        no_pipeline_context = ModelContext(context.df, 'target', is_pipeline=False)
        factory = SVCFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, SVC)

    @parameterized.expand(contexts())
    def test_create_model_nondefault_scaler(self, name, context):
        diff_scaler_context = ModelContext(context.df, 'target', scaler=RobustScaler())
        factory = SVCFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    @parameterized.expand(contexts())
    def test_create_model_custom_paramgrid(self, name, context):
        factory = SVCFactory(context)
        factory.create_model(param_grid={'svc__C': [0.1, 1, 10]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(contexts())
    def test_create_model_invalid_paramgrid(self, name, context):
        invalid_param_grids = [
            ("invalid_string", 'str'),
            ("invalid_float", 1.23),
            ("invalid_int", 123),
            ("invalid_tuple", (1, 2, 3)),
            ("invalid_list", [1, 2, 3]),
            ("invalid_empty_dict", {})
        ]
        for invalid_param_grid in invalid_param_grids:
            with self.subTest(invalid_param_grid=invalid_param_grid):
                factory = SVCFactory(context)
                with self.assertRaises(ValueError):
                    factory.create_model(param_grid=invalid_param_grid[1])

    @parameterized.expand(contexts())
    def test_create_model_paramgrid_for_fit(self, name, context):
        invalid_param_grids_for_fit = [
            ("invalid_param_name", {'invalid_param': [1, 2, 3]})
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = SVCFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)


class TestRandomForestFactory(TestClassifiers):
    @parameterized.expand(contexts())
    def test_create_model(self, name, context):
        factory = RandomForestFactory(context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], RandomForestClassifier)

    @parameterized.expand(contexts())
    def test_create_model_no_pipeline(self, name, context):
        no_pipeline_context = ModelContext(context.df, 'target', is_pipeline=False)
        factory = RandomForestFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, RandomForestClassifier)

    @parameterized.expand(contexts())
    def test_create_model_nondefault_scaler(self, name, context):
        diff_scaler_context = ModelContext(context.df, 'target', scaler=RobustScaler())
        factory = RandomForestFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    @parameterized.expand(contexts())
    def test_create_model_custom_paramgrid(self, name, context):
        factory = RandomForestFactory(context)
        factory.create_model(param_grid={'randomforestclassifier__n_estimators': [100, 200]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(contexts())
    def test_create_model_invalid_paramgrid(self, name, context):
        invalid_param_grids = [
            ("invalid_string", 'str'),
            ("invalid_float", 1.23),
            ("invalid_int", 123),
            ("invalid_tuple", (1, 2, 3)),
            ("invalid_list", [1, 2, 3]),
            ("invalid_empty_dict", {})
        ]
        for invalid_param_grid in invalid_param_grids:
            with self.subTest(invalid_param_grid=invalid_param_grid):
                factory = RandomForestFactory(context)
                with self.assertRaises(ValueError):
                    factory.create_model(param_grid=invalid_param_grid[1])

    @parameterized.expand(contexts())
    def test_create_model_paramgrid_for_fit(self, name, context):
        invalid_param_grids_for_fit = [
            ("invalid_param_name", {'invalid_param': [1, 2, 3]})
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = RandomForestFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)


class TestKNNFactory(TestClassifiers):
    @parameterized.expand(contexts())
    def test_create_model(self, name, context):
        factory = KNNFactory(context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], KNeighborsClassifier)

    @parameterized.expand(contexts())
    def test_create_model_no_pipeline(self, name, context):
        no_pipeline_context = ModelContext(context.df, 'target', is_pipeline=False)
        factory = KNNFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, KNeighborsClassifier)

    @parameterized.expand(contexts())
    def test_create_model_nondefault_scaler(self, name, context):
        diff_scaler_context = ModelContext(context.df, 'target', scaler=RobustScaler())
        factory = KNNFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    @parameterized.expand(contexts())
    def test_create_model_custom_paramgrid(self, name, context):
        factory = KNNFactory(context)
        factory.create_model(param_grid={'kneighborsclassifier__n_neighbors': [3, 5, 7]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(contexts())
    def test_create_model_invalid_paramgrid(self, name, context):
        invalid_param_grids = [
            ("invalid_string", 'str'),
            ("invalid_float", 1.23),
            ("invalid_int", 123),
            ("invalid_tuple", (1, 2, 3)),
            ("invalid_list", [1, 2, 3]),
            ("invalid_empty_dict", {})
        ]
        for invalid_param_grid in invalid_param_grids:
            with self.subTest(invalid_param_grid=invalid_param_grid):
                factory = KNNFactory(context)
                with self.assertRaises(ValueError):
                    factory.create_model(param_grid=invalid_param_grid[1])

    @parameterized.expand(contexts())
    def test_create_model_paramgrid_for_fit(self, name, context):
        invalid_param_grids_for_fit = [
            ("invalid_param_name", {'invalid_param': [1, 2, 3]})
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = KNNFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)


class TestGradientBoostingFactory(TestClassifiers):
    @parameterized.expand(contexts())
    def test_create_model(self, name, context):
        factory = GradientBoostingFactory(context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[1][1], GradientBoostingClassifier)

    @parameterized.expand(contexts())
    def test_create_model_no_pipeline(self, name, context):
        no_pipeline_context = ModelContext(context.df, 'target', is_pipeline=False)
        factory = GradientBoostingFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GradientBoostingClassifier)

    @parameterized.expand(contexts())
    def test_create_model_nondefault_scaler(self, name, context):
        diff_scaler_context = ModelContext(context.df, 'target', scaler=RobustScaler())
        factory = GradientBoostingFactory(diff_scaler_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], RobustScaler)

    @parameterized.expand(contexts())
    def test_create_model_custom_paramgrid(self, name, context):
        factory = GradientBoostingFactory(context)
        factory.create_model(param_grid={'gradientboostingclassifier__n_estimators': [100, 200]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(contexts())
    def test_create_model_invalid_paramgrid(self, name, context):
        invalid_param_grids = [
            ("invalid_string", 'str'),
            ("invalid_float", 1.23),
            ("invalid_int", 123),
            ("invalid_tuple", (1, 2, 3)),
            ("invalid_list", [1, 2, 3]),
            ("invalid_empty_dict", {})
        ]
        for invalid_param_grid in invalid_param_grids:
            with self.subTest(invalid_param_grid=invalid_param_grid):
                factory = GradientBoostingFactory(context)
                with self.assertRaises(ValueError):
                    factory.create_model(param_grid=invalid_param_grid[1])

    @parameterized.expand(contexts())
    def test_create_model_paramgrid_for_fit(self, name, context):
        invalid_param_grids_for_fit = [
            ("invalid_param_name", {'invalid_param': [1, 2, 3]})
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = GradientBoostingFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)


class TestANNClassifierFactory(TestClassifiers):
    @parameterized.expand(contexts())
    def test_create_model(self, name, context):
        ann_context = ModelContext(context.df, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(ann_context)
        factory.create_model()
        self.assertIsInstance(factory.model, GridSearchCV)
        self.assertIsInstance(factory.model.estimator.steps[0][1], Normalizer)
        self.assertIsInstance(factory.model.estimator.steps[1][1], KerasClassifier)
        self.assertTrue(callable(factory.model.estimator.steps[1][1].build_fn))

    @parameterized.expand(contexts())
    def test_create_model_no_pipeline(self, name, context):
        no_pipeline_context = ModelContext(context.df, 'target', is_pipeline=False)
        factory = ANNClassifierFactory(no_pipeline_context)
        factory.create_model()
        self.assertIsInstance(factory.model, KerasClassifier)

    @parameterized.expand(contexts())
    def test_create_model_custom_paramgrid(self, name, context):
        factory = ANNClassifierFactory(context)
        factory.create_model(param_grid={'batch_size': [16, 32, 64]})
        self.assertIsInstance(factory.model, GridSearchCV)

    @parameterized.expand(contexts())
    def test_create_model_invalid_paramgrid(self, name, context):
        invalid_param_grids = [
            ("invalid_string", 'str'),
            ("invalid_float", 1.23),
            ("invalid_int", 123),
            ("invalid_tuple", (1, 2, 3)),
            ("invalid_list", [1, 2, 3]),
            ("invalid_empty_dict", {})
        ]
        for invalid_param_grid in invalid_param_grids:
            with self.subTest(invalid_param_grid=invalid_param_grid):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.create_model(param_grid=invalid_param_grid[1])

    @parameterized.expand(contexts())
    def test_create_model_paramgrid_for_fit(self, name, context):
        invalid_param_grids_for_fit = [
            ("invalid_param_name", {'invalid_param': [1, 2, 3]}),
            ("invalid_none", None)
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = ANNClassifierFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)

    @parameterized.expand(contexts())
    def test_build_model(self, name, context):
        factory = ANNClassifierFactory(context)
        model = factory.build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 5)
        # 3 Dense layers and 2 Dropout layers
        if name == "binary_context":
            self.assertEqual(model.loss, 'binary_crossentropy')
            # Default loss function for binary classification
        else:
            self.assertEqual(model.loss, 'categorical_crossentropy')
            # Default loss function for multi-class classification

    @parameterized.expand(contexts())
    def test_build_model_custom_params(self, name, context):
        factory = ANNClassifierFactory(context)
        model = factory.build_model(neuron_layers=(128, 64, 32),
                                    dropout_layers=(0.1, 0.2, 0.3),
                                    activation='tanh',
                                    optimizer='rmsprop')
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 7) # 4 Dense layers (including output) and 3 Dropout layers

    @parameterized.expand(contexts())
    def test_build_model_invalid_neuron_layers(self, name, context):
        invalid_inputs = [
            ("invalid_string", 'invalid'),
            ("invalid_int", 123),
            ("invalid_list", [64, 64]),
            ("invalid_dict", {'layer1': 64, 'layer2': 64}),
            ("invalid_none", None)
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.build_model(neuron_layers=invalid_input[1])

    @parameterized.expand(contexts())
    def test_build_model_invalid_tuple_neuron_layers(self, name, context):
        invalid_inputs = [
            ("invalid_string_in_tuple", ('invalid',)),
            ("invalid_float_in_tuple", (1.23,)),
            ("invalid_list_in_tuple", ([64, 64],)),
            ("invalid_dict_in_tuple", ({'layer1': 64},)),
            ("invalid_none_in_tuple", (None,))
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.build_model(neuron_layers=invalid_input[1])

    @parameterized.expand(contexts())
    def test_build_model_invalid_dropout_layers(self, name, context):
        invalid_inputs = [
            ("invalid_string", 'invalid'),
            ("invalid_int", 1),
            ("invalid_list", [0.2, 0.2]),
            ("invalid_dict", {'layer1': 0.2, 'layer2': 0.2}),
            ("invalid_none", None)
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.build_model(dropout_layers=invalid_input[1])

    @parameterized.expand(contexts())
    def test_build_model_invalid_tuple_dropout_layers(self, name, context):
        invalid_inputs = [
            ("invalid_string_in_tuple", ('invalid',)),
            ("invalid_int_in_tuple", (1,)),
            ("invalid_list_in_tuple", ([0.2, 0.2],)),
            ("invalid_dict_in_tuple", ({'layer1': 0.2},)),
            ("invalid_none_in_tuple", (None,))
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.build_model(dropout_layers=invalid_input[1])

    @parameterized.expand(contexts())
    def test_build_model_invalid_activation(self, name, context):
        invalid_inputs = [
            ("invalid_string", 'invalid'),
            ("invalid_int", 123),
            ("invalid_list", ['relu', 'relu']),
            ("invalid_dict", {'layer1': 'relu', 'layer2': 'relu'})
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.build_model(activation=invalid_input[1])

    @parameterized.expand(contexts())
    def test_build_model_invalid_optimizer(self, name, context):
        invalid_inputs = [
            ("invalid_string", 'invalid'),
            ("invalid_int", 123),
            ("invalid_list", ['invalid',]),
            ("invalid_tuple", ('invalid',)),
            ("invalid_dict", {'layer1': 'invalid', 'layer2': 'invalid'})
        ]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                factory = ANNClassifierFactory(context)
                with self.assertRaises(ValueError):
                    factory.build_model(optimizer=invalid_input[1])

if __name__ == '__main__':
    unittest.main()