import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from interfaces import MLClassifierInterface, MLRegressorInterface
from context import ModelContext

class TestMLClassifier(MLClassifierInterface):
    """Test class for the MLClassifierInterface"""
    def create_model(self, param_grid=None):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        self.model = GridSearchCV(pipeline, param_grid or {}, cv=10)

class TestMLRegressor(MLRegressorInterface):
    """Test class for the MLRegressorInterface"""
    def create_model(self, param_grid=None):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        self.model = GridSearchCV(pipeline, param_grid or {}, cv=10)

class BaseTestInterface(unittest.TestCase):
    """Base class for the interface tests"""
    def setUp(self) -> None:
        # Context tested in its own file
        self.df_regression = pd.read_csv('test_csv/regression_data.csv')
        self.context_regression = ModelContext(self.df_regression, 'target')

        self.df_classification = pd.read_csv('test_csv/binary_classification_data.csv')
        self.context_classification = ModelContext(self.df_classification, 'target')

class TestMLClassifierInterface(BaseTestInterface):

    def test_init(self):
        classifier = TestMLClassifier(self.context_classification)
        self.assertIsNotNone(classifier.context)
        self.assertIsNotNone(classifier.X)
        self.assertIsNotNone(classifier.y)
        self.assertIsNotNone(classifier.X_train)
        self.assertIsNotNone(classifier.X_test)
        self.assertIsNotNone(classifier.y_train)
        self.assertIsNotNone(classifier.y_test)

    def test_init_invalid(self):
        with self.assertRaises(TypeError):
            TestMLClassifier('invalid')

    def test_create_model(self):
        # properly created create_model() is tested in test_regressors.py
        # and test_classifiers.py, this is just a filler method for the interface testing
        classifier = TestMLClassifier(self.context_classification)
        classifier.create_model()
        self.assertIsNotNone(classifier.model)

    def test_train_model(self):
        classifier = TestMLClassifier(self.context_classification)
        classifier.create_model()
        self.assertIsNotNone(classifier.model)
        classifier.train_model()
        # Check if the model's best parameters and best score are not None after training
        self.assertIsNotNone(classifier.model.best_params_)
        self.assertIsNotNone(classifier.model.best_score_)

    def test_predict(self):
        classifier = TestMLClassifier(self.context_classification)
        classifier.create_model()
        classifier.train_model()
        predictions = classifier.predict()
        self.assertIsNotNone(classifier.y_pred)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(classifier.y_pred))
        self.assertEqual(len(predictions), len(classifier.y_test))
        self.assertListEqual(predictions.tolist(), classifier.y_pred.tolist())

    def test_metrics(self):
        classifier = TestMLClassifier(self.context_classification)
        classifier.create_model()
        classifier.train_model()
        classifier.predict()
        metrics_df = classifier.metrics('classification_metrics.csv')
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertIsNotNone(metrics_df['type'][0])
        self.assertIsNotNone(metrics_df['refit time'][0])
        self.assertIsNotNone(metrics_df['precision'][0])
        self.assertIsNotNone(metrics_df['recall'][0])
        self.assertIsNotNone(metrics_df['f1 score'][0])
        self.assertIsNotNone(metrics_df['accuracy'][0])

class TestMLRegressorInterface(BaseTestInterface):

    def test_init(self):
        regressor = TestMLRegressor(self.context_regression)
        self.assertIsNotNone(regressor.context)
        self.assertIsNotNone(regressor.X)
        self.assertIsNotNone(regressor.y)
        self.assertIsNotNone(regressor.X_train)
        self.assertIsNotNone(regressor.X_test)
        self.assertIsNotNone(regressor.y_train)
        self.assertIsNotNone(regressor.y_test)

    def test_init_invalid(self):
        with self.assertRaises(TypeError):
            TestMLRegressor('invalid')

    def test_reate_model(self):
        regressor = TestMLRegressor(self.context_regression)
        regressor.create_model()
        self.assertIsNotNone(regressor.model)

    def test_train_model(self):
        regressor = TestMLRegressor(self.context_regression)
        regressor.create_model()
        regressor.train_model()
        self.assertTrue(hasattr(regressor, 'model'))
        # Check if the model's best parameters and best score are not None after training
        self.assertIsNotNone(regressor.model.best_params_)
        self.assertIsNotNone(regressor.model.best_score_)

    def test_predict(self):
        regressor = TestMLRegressor(self.context_regression)
        regressor.create_model()
        regressor.train_model()
        predictions = regressor.predict()
        self.assertIsNotNone(regressor.y_pred)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(regressor.y_pred))
        self.assertEqual(len(predictions), len(regressor.y_test))
        self.assertListEqual(predictions.tolist(), regressor.y_pred.tolist())

    def test_metrics(self):
        regressor = TestMLRegressor(self.context_regression)
        regressor.create_model()
        regressor.train_model()
        regressor.predict()
        metrics_df = regressor.metrics('regression_metrics.csv')
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertIsNotNone(metrics_df['type'][0])
        self.assertIsNotNone(metrics_df['refit time'][0])
        self.assertIsNotNone(metrics_df['mae'][0])
        self.assertIsNotNone(metrics_df['rmse'][0])
        self.assertIsNotNone(metrics_df['r2'][0])

if __name__ == '__main__':
    unittest.main()
