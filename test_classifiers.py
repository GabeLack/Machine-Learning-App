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
        self.assertIsInstance(factory.model.estimator.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(factory.model.estimator.steps[1][1], StandardScaler)
        self.assertIsInstance(factory.model.estimator.steps[-1][1], KNeighborsClassifier)

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
        self.assertIsInstance(factory.model.estimator.steps[1][1], RobustScaler)

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
            ("invalid_param_name", {'invalid_param': [1, 2, 3]})
        ]
        for bad_param_grid in invalid_param_grids_for_fit:
            with self.subTest(bad_param_grid=bad_param_grid):
                factory = ANNClassifierFactory(context)
                factory.create_model(param_grid=bad_param_grid[1])
                with self.assertRaises(ValueError):
                    factory.model.fit(context.X_train, context.y_train)

    @parameterized.expand(contexts())
    def test_train_model(self, name, context):
        mock_context = ModelContext(context.df, 'target', scaler=Normalizer())
        factory = ANNClassifierFactory(mock_context)
        #! I don't want to fix multi-ANNClassifier for now, everything else works,
        #! too many changes required. This test is written correctly, just that the
        #! multi-ANNClassifier is not working.

        # Smaller param grid for faster testing
        small_param_grid = {
            'kerasclassifier__batch_size': [16],
            'kerasclassifier__epochs': [10],
            'kerasclassifier__optimizer': ['adam'],
            'kerasclassifier__neuron_layers': [(32, 32)],
            'kerasclassifier__dropout_layers': [(0.1, 0.1)]
        }
        factory.create_model(param_grid=small_param_grid)

        factory.train_model()
        self.assertTrue(hasattr(factory, 'model'))
        # Check if the model's best parameters and best score are not None after training
        self.assertIsNotNone(factory.model.best_params_)
        self.assertIsNotNone(factory.model.best_score_)

if __name__ == '__main__':
    unittest.main()