import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
from app import MLApp
import pandas as pd
from sklearn.pipeline import Pipeline

import logging #! Remove when complete
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from context import ModelContext
from classifiers import (LogisticFactory, SVCFactory, RandomForestFactory, 
                         KNNFactory, GradientBoostingFactory, ANNClassifierFactory)
from regressors import LinearFactory, ElasticNetFactory, SVRFactory, ANNRegressorFactory

class TestMLApp(unittest.TestCase):

    def setUp(self):
        # Read multiple CSV files and store them as DataFrames
        self.df_regression = pd.read_csv('test_csv/regression_data.csv')
        self.df_regression_missing = pd.read_csv('test_csv/regression_missing_data.csv')
        self.df_multi_class = pd.read_csv('test_csv/multi_classification_data.csv')
        self.df_binary_class = pd.read_csv('test_csv/binary_classification_data.csv')
        self.df_object_class = pd.read_csv('test_csv/classification_object_data.csv')

    @patch.object(MLApp, 'create_widgets', MagicMock())
    def test_init(self):
        app = MLApp()
        self.assertIsInstance(app, MLApp)
        self.assertIsInstance(app.root, tk.Tk)
        self.assertEqual(app.root.title(), "MLApp")

        # Variables to store user input
        self.assertIsInstance(app.data_type, tk.StringVar)
        self.assertIsInstance(app.file_path, tk.StringVar)
        self.assertIsInstance(app.target_column, tk.StringVar)

        # Text widget for output display
        self.assertIsInstance(app.output_text, tk.Text)
        # Ensure create_widgets was called with MagicMock
        app.create_widgets.assert_called_once()

    @patch('app.tk.Button')
    @patch('app.tk.Radiobutton')
    @patch('app.tk.Label')
    def test_create_widgets(self, mock_label, mock_radiobutton, mock_button):
        app = MLApp() # create_widgets is called in __init__

        # Check that the correct widgets were created
        mock_label.assert_any_call(app.root, text="Choose data type:")
        mock_radiobutton.assert_any_call(app.root, text="Regression", variable=app.data_type, value="regression")
        mock_radiobutton.assert_any_call(app.root, text="Classification", variable=app.data_type, value="classification")
        mock_label.assert_any_call(app.root, text="Select CSV file:")
        mock_button.assert_any_call(app.root, text="Browse", command=app.browse_file)
        mock_button.assert_any_call(app.root, text="Show Column List", command=app.show_column_list)
        mock_button.assert_any_call(app.root, text="Initialize Models", command=app.initialize_models)
        mock_button.assert_any_call(app.root, text="Get and plot Results", command=app.plot_results)
        mock_button.assert_any_call(app.root, text="Recommended Model", command=app.recommended_model)
        mock_button.assert_any_call(app.root, text="Choose Model", command=app.choose_model)
        mock_button.assert_any_call(app.root, text="Close", command=app.root.destroy)
        
        # Ensure that each widget was packed
        self.assertEqual(mock_label.return_value.pack.call_count, 2)
        self.assertEqual(mock_radiobutton.return_value.pack.call_count, 2)
        self.assertEqual(mock_button.return_value.pack.call_count, 7)

    @patch('app.filedialog.askopenfilename')
    def test_browse_file(self, mock_askopenfilename):
        # Mock the file dialog to return a file path
        mock_askopenfilename.return_value = 'test_csv/regression_data.csv'
        app = MLApp()
        app.browse_file()
        self.assertEqual(app.file_path.get(), 'test_csv/regression_data.csv')

    @patch('app.pd.read_csv')
    @patch('app.tk.OptionMenu')
    def test_show_column_list(self, mock_optionmenu, mock_read_csv):
        mock_read_csv.return_value = self.df_regression
        app = MLApp()
        app.file_path.set('test_csv/regression_data.csv')
        
        # Mock the OptionMenu to simulate user selection
        mock_optionmenu_instance = MagicMock()
        mock_optionmenu.return_value = mock_optionmenu_instance
        app.show_column_list()
        
        # Simulate user selecting a column
        selected_column = self.df_regression.columns[-1] # Select the last column which is 'target'
        app.target_column.set(selected_column)
        
        # Validate that the selected column was saved correctly
        self.assertEqual(app.target_column.get(), selected_column)
        mock_optionmenu_instance.pack.assert_called_once()

    @patch('app.tk.messagebox.showerror')
    def test_show_column_list_no_file(self, mock_showerror):
        app = MLApp()
        app.show_column_list()
        mock_showerror.assert_called_once_with("Error", "Please select a CSV file first.")

    @patch('app.pd.read_csv')
    def test_get_models_regressor(self, mock_read_csv):
        #! test get_models before initialize_models
        # Use one of the DataFrames read in setUp
        mock_read_csv.return_value = self.df_regression

        # Create an instance of the application
        app = MLApp()
        app.data_type.set('regression')
        app.file_path.set('test_csv/regression_data1.csv')
        app.target_column.set('target')

        # Create a real ModelContext object
        context = ModelContext(
            df = pd.read_csv(app.file_path.get()),
            target_column = app.target_column.get(),
            is_pipeline=True
        )

        # Call the get_models method
        models = app.get_models(context)

        # Validate the number of models returned
        self.assertEqual(len(models), 4)  # Assuming 4 regression models
        logging.debug(models)
        self.assertTrue(isinstance(models[0], LinearFactory))
        self.assertTrue(isinstance(models[1], ElasticNetFactory))
        self.assertTrue(isinstance(models[2], SVRFactory))
        self.assertTrue(isinstance(models[3], ANNRegressorFactory))

    @patch('app.pd.read_csv')
    def test_get_models_binary_classifier(self, mock_read_csv):
        #! test get_models before initialize_models
        # Use one of the DataFrames read in setUp
        mock_read_csv.return_value = self.df_binary_class

        # Create an instance of the application
        app = MLApp()
        app.data_type.set('classification')
        app.file_path.set('test_csv/binary_classification_data.csv')
        app.target_column.set('target')

        # Create a real ModelContext object
        context = ModelContext(
            df = pd.read_csv(app.file_path.get()),
            target_column = app.target_column.get(),
            is_pipeline=True
        )

        # Call the get_models method
        models = app.get_models(context)

        # Validate the number of models returned
        self.assertEqual(len(models), 6) # Assuming 6 classification models
        logging.debug(models)
        self.assertTrue(isinstance(models[0], LogisticFactory))
        self.assertTrue(isinstance(models[1], SVCFactory))
        self.assertTrue(isinstance(models[2], RandomForestFactory))
        self.assertTrue(isinstance(models[3], KNNFactory))
        self.assertTrue(isinstance(models[4], GradientBoostingFactory))
        self.assertTrue(isinstance(models[5], ANNClassifierFactory))

    @patch('app.pd.read_csv')
    def test_get_models_multi_classifier(self, mock_read_csv):
        #! test get_models before initialize_models
        # Use one of the DataFrames read in setUp
        mock_read_csv.return_value = self.df_multi_class

        # Create an instance of the application
        app = MLApp()
        app.data_type.set('classification')
        app.file_path.set('test_csv/multi_classification_data.csv')
        app.target_column.set('target')

        # Create a real ModelContext object
        context = ModelContext(
            df = pd.read_csv(app.file_path.get()),
            target_column = app.target_column.get(),
            is_pipeline=True
        )

        # Call the get_models method
        models = app.get_models(context)

        # Validate the number of models returned
        self.assertEqual(len(models), 6) # Assuming 6 classification models
        logging.debug(models)
        self.assertTrue(isinstance(models[0], LogisticFactory))
        self.assertTrue(isinstance(models[1], SVCFactory))
        self.assertTrue(isinstance(models[2], RandomForestFactory))
        self.assertTrue(isinstance(models[3], KNNFactory))
        self.assertTrue(isinstance(models[4], GradientBoostingFactory))
        self.assertTrue(isinstance(models[5], ANNClassifierFactory))

    @patch('app.pd.read_csv')
    def test_initialize_models(self, mock_read_csv):
        # Use one of the DataFrames read in setUp
        mock_read_csv.return_value = self.df_regression
        
        # Create an instance of the application
        app = MLApp()
        app.data_type.set('regression')
        app.file_path.set('test_csv/regression_data1.csv')
        app.target_column.set('target')
        
        # Call the initialize_models method
        app.initialize_models()
        # context created in the app code from data_type, file_path, and target_column
        
        # Check if the factories and context attributes are set
        self.assertTrue(hasattr(app, 'factories'))
        self.assertTrue(hasattr(app, 'context'))

    @patch('app.tk.messagebox.showerror')
    def test_initialize_models_no_selection(self, mock_showerror):
        app = MLApp()
        app.initialize_models()
        self.assertFalse(hasattr(app, 'factories'))
        self.assertFalse(hasattr(app, 'context'))
        self.assertIn("Please select data type, file path, and target column.\n", app.output_text.get("1.0", tk.END))

    @patch('app.pd.read_csv')
    @patch('app.tk.messagebox.askquestion')
    def test_initialize_models_missing_values_yes(self, mock_askquestion, mock_read_csv):
        # Mock the read_csv to return the DataFrame with missing values
        mock_read_csv.return_value = self.df_regression_missing
        # Mock the askquestion to return 'yes'
        mock_askquestion.return_value = 'yes'
        
        # Create an instance of the application
        app = MLApp()
        app.data_type.set('regression')
        app.file_path.set('test_csv/regression_missing_data.csv')
        app.target_column.set('target')

        # Create a copy of the DataFrame with missing values for comparison
        df_comparison = self.df_regression_missing.copy()
        
        # Call the initialize_models method
        app.initialize_models()

        # Verify that the missing values were handled and models initialized
        self.assertIn("Models initialized successfully.\n", app.output_text.get("1.0", tk.END))
        self.assertIsNotNone(app.context)
        self.assertTrue(len(app.context.df) < len(df_comparison)) # Check that missing values were removed

    @patch('app.pd.read_csv')
    @patch('app.tk.messagebox.askquestion')
    @patch('app.MLApp.browse_file')
    def test_initialize_models_missing_values_no(self, mock_browse_file, mock_askquestion, mock_read_csv):
        # Mock the read_csv to return the DataFrame with missing values initially
        mock_read_csv.return_value = self.df_regression_missing
        # Mock the askquestion to return 'no'
        mock_askquestion.return_value = 'no'
        
        # Create an instance of the application
        app = MLApp()
        app.data_type.set('regression')
        app.file_path.set('test_csv/regression_missing_data.csv')
        app.target_column.set('target')
        
        # Call the initialize_models method
        app.initialize_models()
        
        # Verify that browse_file is called to replace the DataFrame with missing values
        mock_browse_file.assert_called_once()

    @patch('app.pd.read_csv')
    @patch('app.tk.messagebox.askquestion')
    def test_initialize_models_object_values_yes(self, mock_askquestion, mock_read_csv):
        # Mock the read_csv to return the DataFrame with object values
        mock_read_csv.return_value = self.df_object_class
        # Mock the askquestion to return 'yes'
        mock_askquestion.return_value = 'yes'

        # Create an instance of the application
        app = MLApp()
        app.data_type.set('classification')
        app.file_path.set('test_csv/classification_object_data.csv')
        app.target_column.set('target')

        # Create a copy of the DataFrame with object values for comparison
        df_comparison = self.df_object_class.copy()

        # Call the initialize_models method
        app.initialize_models()
        
        # Verify that the object values were handled and models initialized
        self.assertIn("Models initialized successfully.\n", app.output_text.get("1.0", tk.END))
        self.assertIsNotNone(app.context)
        # Check that object values were encoded
        self.assertTrue(len(app.context.df.columns) > len(df_comparison.columns))

    @patch('app.pd.read_csv')
    @patch('app.tk.messagebox.askquestion')
    @patch('app.MLApp.browse_file')
    def test_initialize_models_object_values_no(self, mock_browse_file, mock_askquestion, mock_read_csv):
        # Mock the read_csv to return the DataFrame with object values
        mock_read_csv.return_value = self.df_object_class
        # Mock the askquestion to return 'no'
        mock_askquestion.return_value = 'no'
        
        # Create an instance of the application
        app = MLApp()
        app.data_type.set('classification')
        app.file_path.set('test_csv/classification_object_data.csv')
        app.target_column.set('target')
        
        # Call the initialize_models method
        app.initialize_models()
        
        # Verify that browse_file is called to replace the DataFrame with object values
        mock_browse_file.assert_called_once()

    def test_train_models(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context

        # Put a linear model in the factories list
        linear = LinearFactory(context)
        linear.create_model() # This method tested in test_regressors.py
        app.factories = [linear]

        # Individual method used to train each model is tested in test_interfaces.py
        app.train_models()

        # Check that the models were trained
        self.assertIn("Models trained successfully.\n", app.output_text.get("1.0", tk.END))

    def test_predict_models(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context

        # Put a linear model in the factories list
        linear = LinearFactory(context)
        linear.create_model() # This method tested in test_regressors.py
        app.factories = [linear]
        # Train the model before predicting
        app.train_models()

        # Call the predict_models method
        app.predict_models()
        
        # Check that the models were predicted
        self.assertIn("Predictions made successfully.\n", app.output_text.get("1.0", tk.END))

    def test_process_metrics(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context

        # Put a linear model in the factories list
        linear = LinearFactory(context)
        linear.create_model() # This method tested in test_regressors.py
        app.factories = [linear]
        # Train the model before predicting
        app.train_models()
        # Call the predict_models method
        app.predict_models()
        
        # Call the process_metrics method
        metrics_df = app.process_metrics()
        
        # Check that the metrics were processed
        self.assertIn("Metrics processed successfully.\n", app.output_text.get("1.0", tk.END))
        self.assertIsNotNone(metrics_df)

    @patch('plotting.Plotting.plot_residuals')
    def test_plot_results_regression(self, mock_plot_residuals):
        # Mock the plot methods to return a mock figure
        mock_fig = MagicMock()
        mock_plot_residuals.return_value = mock_fig

        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        app.plot_results()

        # check that plots were shown
        mock_fig.show.assert_called_once()
        # Check that the plots were plotted successfully
        self.assertIn("Results plotted successfully.\n", app.output_text.get("1.0", tk.END))

    @patch('plotting.Plotting.plot_confusion_matrix')
    def test_plot_results_classification(self, mock_plot_confusion_matrix):
        # Mock the plot methods to return a mock figure
        mock_fig = MagicMock()
        mock_plot_confusion_matrix.return_value = mock_fig

        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_binary_class, target_column='target')
        app.context = context
        app.data_type.set('classification')

        # Put a linear model in the factories list
        linear = LogisticFactory(context)
        linear.create_model()
        app.factories = [linear]

        app.plot_results()

        # check that plots were shown
        mock_fig.show.assert_called_once()
        # Check that the plots were plotted successfully
        self.assertIn("Results plotted successfully.\n", app.output_text.get("1.0", tk.END))

    def test_get_best_type(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear and elasticnet model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        # get_best_type depends on metrics so we need to call all the methods
        app.train_models()
        app.predict_models()
        app.process_metrics()

        # Call the get_best_type method
        best_type = app.get_best_type()

        # Check that the best type was returned
        self.assertEqual(best_type, "linearregression")

    def test_get_best_params(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear and elasticnet model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        # get_best_params depends on metrics so we need to call all the methods
        app.train_models()
        app.predict_models()
        app.process_metrics()

        # Call the get_best_params method
        best_params = app.get_best_params('linearregression')

        # Check that the best params were returned
        self.assertTrue(isinstance(best_params, dict))

    def test_get_cv_results(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear and elasticnet model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        # get_cv_results depends on metrics so we need to call all the methods
        app.train_models()
        app.predict_models()
        app.process_metrics()

        # Call the get_cv_results method
        cv_results = app.get_cv_results('linearregression')

        # Check that the cv results were returned
        self.assertTrue(isinstance(cv_results, dict))

    def test_get_best_model(self):
        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear and elasticnet model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        # get_best_model depends on metrics so we need to call all the methods
        app.train_models()
        app.predict_models()
        app.process_metrics()

        # Call the get_best_model method
        best_model = app.get_best_model('linearregression')

        # Check that the best model was returned
        self.assertTrue(isinstance(best_model, Pipeline))

    @patch('matplotlib.pyplot.show')
    @patch('plotting.Plotting.corr_w_test_score_plot')
    def test_recommended_model(self, mock_corr_plot, mock_show):
        # Mock the plot method to return a mock figure
        mock_fig = MagicMock()
        mock_corr_plot.return_value = mock_fig

        # Create an instance of the application
        app = MLApp()
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        # get_best_type depends on metrics so we need to call all the methods
        app.train_models()
        app.predict_models()
        app.process_metrics()

        # Call the recommended_model method
        app.recommended_model()

        # Check that the recommended model was displayed
        self.assertIn("Recommended Model: linearregression\n", app.output_text.get("1.0", tk.END))
        # Check they were called
        mock_corr_plot.assert_called_once()
        mock_show.assert_called_once()

    @patch('app.tk.StringVar')
    @patch('app.tk.Label')
    @patch('app.tk.OptionMenu')
    @patch('app.filedialog.asksaveasfilename')
    @patch('app.messagebox.showinfo')
    @patch('app.dump')
    def test_choose_model(self, mock_dump, mock_showinfo, mock_asksaveasfilename, mock_optionmenu, mock_label, mock_stringvar):
        """Test that the user can choose a model and save it successfully."""
        # Mock the StringVar to simulate user selection
        mock_stringvar_instance = MagicMock()
        mock_stringvar.return_value = mock_stringvar_instance
        mock_stringvar_instance.get.return_value = 'linearregression'

        # Mock the asksaveasfilename to return a file path
        mock_asksaveasfilename.return_value = 'test_model.joblib'

        # Mock the dump function to do nothing
        mock_dump.return_value = None

        # Mock the showinfo function to do nothing
        mock_showinfo.return_value = None

        # Mock the Tk instance to prevent the GUI from popping up
        mock_root = MagicMock()

        # Create an instance of the application
        app = MLApp()
        app.root = mock_root
        context = ModelContext(df=self.df_regression, target_column='target')
        app.context = context
        app.data_type.set('regression')

        # Put a linear model in the factories list
        linear = LinearFactory(context)
        linear.create_model()
        app.factories = [linear]

        # choose_mdoel depends on metrics so we need to call all the methods
        app.train_models()
        app.predict_models()
        app.process_metrics()

        # Call the choose_model method
        app.choose_model()

        # Check that the model was chosen and saved
        mock_dump.assert_called_once_with(app.get_best_model('linearregression'), 'test_model.joblib')
        mock_showinfo.assert_called_once_with("Success", "Model saved successfully to test_model.joblib.")

    def test_methods_no_factories(self):
        """Test all methods that require factories to be initialized."""
        
        # Create an instance of the application
        app = MLApp()

        # List of methods to test
        methods_to_test = [
            app.train_models,
            app.predict_models,
            app.process_metrics,
            lambda: app.get_best_type(),
            lambda: app.get_best_params('linearregression'),
            lambda: app.get_cv_results('linearregression'),
            lambda: app.get_best_model('linearregression'),
            app.recommended_model,
            app.choose_model
        ]

        # Expected output for each method
        expected_outputs = [
            "Please initialize models first.\n",
            "Please initialize models first.\n",
            "Please initialize models first.\n",
            None,
            None,
            None,
            None,
            "Please initialize models first.\n",
            "Please initialize models first.\n"
        ]

        # Test each method
        for method, expected_output in zip(methods_to_test, expected_outputs):
            with self.subTest(method=method):
                if expected_output is not None:
                    method()
                    self.assertIn(expected_output, app.output_text.get("1.0", tk.END))
                else:
                    result = method()
                    self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()