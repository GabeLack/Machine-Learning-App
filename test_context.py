import unittest
import pandas as pd
from context import ModelContext
from sklearn.preprocessing import Normalizer

class TestModelContext(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing, file has a length of 50 lines
        self.df = pd.read_csv('test_csv/binary_classification_data.csv')
        self.target_column = 'target'

    def test_init(self):
        context = ModelContext(df=self.df, target_column=self.target_column)
        self.assertIsNotNone(context.df)
        self.assertEqual(context.target_column, self.target_column)
        self.assertEqual(context.test_size, 0.3)
        self.assertTrue(context.is_pipeline)
        self.assertIsNotNone(context.scaler)
        self.assertEqual(len(context.X), 50)
        self.assertEqual(len(context.y), 50)
        self.assertEqual(len(context.X_train), 35)
        self.assertEqual(len(context.X_test), 15)
        self.assertEqual(len(context.y_train), 35)
        self.assertEqual(len(context.y_test), 15)

    def test_init_with_invalid_df(self):
        with self.assertRaises(TypeError):
            ModelContext(df='invalid', target_column=self.target_column)
    def test_init_with_invalid_target_column(self):
        with self.assertRaises(ValueError):
            ModelContext(df=self.df, target_column='invalid')

    def test_init_with_invalid_test_size_0(self):
        with self.assertRaises(ValueError):
            ModelContext(df=self.df, target_column=self.target_column, test_size=0)
    def test_init_with_invalid_test_size_1(self):
        with self.assertRaises(ValueError):
            ModelContext(df=self.df, target_column=self.target_column, test_size=1)

    def test_init_with_invalid_is_pipeline(self):
        with self.assertRaises(TypeError):
            ModelContext(df=self.df, target_column=self.target_column, is_pipeline='invalid')

    def test_init_with_valid_scaler(self):
        context = ModelContext(df=self.df, target_column=self.target_column, scaler=Normalizer())
        self.assertTrue(isinstance(context.scaler, Normalizer))
    def test_init_with_invalid_scaler(self):
        with self.assertRaises(ValueError):
            ModelContext(df=self.df, target_column=self.target_column, scaler='invalid')

    def test_check_missing_no_missing_values(self):
        context = ModelContext(df=self.df, target_column=self.target_column)
        try:
            context.check_missing()
        except UserWarning:
            self.fail("check_missing() raised UserWarning unexpectedly!")

    def test_check_missing_with_missing_values(self):
        df_with_missing = self.df.copy()
        # Add a missing value to the first row of 'feature1'
        df_with_missing.loc[0, 'feature1'] = None
        context = ModelContext(df=df_with_missing, target_column=self.target_column)
        with self.assertRaises(UserWarning):
            context.check_missing()

    def test_check_feature_types_no_object_columns(self):
        context = ModelContext(df=self.df, target_column=self.target_column)
        try:
            context.check_feature_types()
        except UserWarning:
            self.fail("check_feature_types() raised UserWarning unexpectedly!")

    def test_check_feature_types_with_object_columns(self):
        df_with_object = self.df.copy()
        # Add a column of object type
        df_with_object.loc[0, 'feature1'] = 'a'
        context = ModelContext(df=df_with_object, target_column=self.target_column)
        with self.assertRaises(UserWarning):
            context.check_feature_types()

if __name__ == '__main__':
    unittest.main()
