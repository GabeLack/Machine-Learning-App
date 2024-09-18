import unittest
import pandas as pd
from context import ModelContext

class TestModelContext(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        example_df = {'feature1': [1, 2, 3, 4, 5],
                      'feature2': [6, 7, 8, 9, 10],
                      'target': [0, 1, 0, 1, 0]}
        self.df = pd.DataFrame(example_df)
        self.target_column = 'target'

    def test_initialization(self):
        context = ModelContext(df=self.df, target_column=self.target_column)
        self.assertEqual(context.target_column, self.target_column)
        self.assertEqual(context.test_size, 0.3)
        self.assertTrue(context.is_pipeline)
        self.assertIsNotNone(context.scaler)
        self.assertEqual(len(context.X_train), 3)
        self.assertEqual(len(context.X_test), 2)

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
        df_with_object['feature3'] = ['a', 'b', 'c', 'd', 'e']
        context = ModelContext(df=df_with_object, target_column=self.target_column)
        with self.assertRaises(UserWarning):
            context.check_feature_types()

if __name__ == '__main__':
    unittest.main()