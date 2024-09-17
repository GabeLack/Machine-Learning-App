import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from joblib import dump
import pandas as pd
from plotting import Plotting

class MLApp:
    plots = []  # To store all plots
    predictions = []  # To store all predictions

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MLApp")

        # Variables to store user inputs
        self.data_type = tk.StringVar()
        self.file_path = tk.StringVar()
        self.target_column = tk.StringVar()

        # Create a Text widget for displaying output
        self.output_text = tk.Text(self.root, height=20, width=60)
        self.output_text.pack()

        # Create a progress bar
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress_bar.pack()

        # UI elements
        self.create_widgets()

    def create_widgets(self):
        # Choose data type (Regression/Classification)
        tk.Label(self.root, text="Choose data type:").pack()
        tk.Radiobutton(self.root, text="Regression", variable=self.data_type, value="regression").pack()
        tk.Radiobutton(self.root, text="Classification", variable=self.data_type, value="classification").pack()

        # Select CSV file
        tk.Label(self.root, text="Select CSV file:").pack()
        tk.Button(self.root, text="Browse", command=self.browse_file).pack()

        # Show column list and choose target column
        tk.Button(self.root, text="Show Column List", command=self.show_column_list).pack()

        # Validate and initialize MLClass
        tk.Button(self.root, text="Initialize Models", command=self.initialize_models).pack()

        # Get results and show plots
        tk.Button(self.root, text="Show Results", command=self.plot_results).pack()

        # Show recommended model
        tk.Button(self.root, text="Recommended Model", command=self.recommended_model).pack()

        # Choose model
        tk.Button(self.root, text="Choose Model", command=self.choose_model).pack()

        # Close button
        tk.Button(self.root, text="Close", command=self.root.destroy).pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.file_path.set(file_path)

    def show_column_list(self):
        # Load CSV file and show column list
        file_path = self.file_path.get()
        if file_path:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
            messagebox.showinfo("Column List", "\n".join(columns))
        else:
            messagebox.showerror("Error", "Please select a CSV file first.")

    def initialize_models(self):
        # Validate and initialize models
        if self.data_type.get() and self.file_path.get() and self.target_column.get():
            df = pd.read_csv(self.file_path.get())
            from context import ModelContext
            context = ModelContext(
                df=df,
                target_column=self.target_column.get(),
                is_pipeline=True
            )

            # Check for missing values
            try:
                context.check_missing()
            except UserWarning as e:
                # Ask the user about fixing/dropping missing values
                response = tk.messagebox.askquestion("Missing Values",
                            f"{str(e)}\n\nDo you want to fix/drop missing values?")
                if response == 'yes':
                    # Reset the index after dropping rows with missing values
                    df.dropna(inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    # Reinitialize with corrected df
                    context = ModelContext(
                        df=df,
                        target_column=self.target_column.get(),
                        is_pipeline=True
                    )
                else:
                    # Ask user to choose new file
                    self.browse_file()
                    return  # Stop execution here if the user chooses a new file

            # Check for object dtype columns
            try:
                context.check_feature_types()
            except UserWarning as e:
                # Ask the user about using get_dummies on object feature columns
                response = tk.messagebox.askquestion("Object Feature Columns",
                            f"{str(e)}\n\nDo you want to use get_dummies on object feature columns?")
                if response == 'yes':
                    # run get dummies on columns other than target column
                    df = pd.get_dummies(df, columns=[col for col in df.columns if \
                                                     col != self.target_column.get()],
                                        drop_first=True, dtype='int')
                    context = ModelContext(
                        df=df,
                        target_column=self.target_column.get(),
                        is_pipeline=True
                    )
                else:
                    # Ask user to choose new file
                    self.browse_file()
                    return

            # Initialize models using the new get_models method
            self.factories = self.get_models(context)
            self.context = context
            self.output_text.insert(tk.END, "Models initialized successfully.\n")
        else:
            self.output_text.insert(tk.END, "Please select data type, file path, and target column.\n")

    def get_models(self, context):
        from factory import ModelFactory, ModelType, ProblemType
        factory = ModelFactory()
        if self.data_type.get() == "regression":
            return [
                factory.create_model(ModelType.LINEAR, ProblemType.REGRESSION, context),
                factory.create_model(ModelType.ELASTICNET, ProblemType.REGRESSION, context),
                factory.create_model(ModelType.SVR, ProblemType.REGRESSION, context)
            ]
        else:
            return [
                factory.create_model(ModelType.LOGISTIC, ProblemType.CLASSIFICATION, context),
                factory.create_model(ModelType.SVC, ProblemType.CLASSIFICATION, context),
                factory.create_model(ModelType.RANDOMFOREST, ProblemType.CLASSIFICATION, context),
                factory.create_model(ModelType.KNEARESTNEIGHBORS, ProblemType.CLASSIFICATION, context),
                factory.create_model(ModelType.GRADIENTBOOSTING, ProblemType.CLASSIFICATION, context)
            ]

    def train_models(self):
        # Train the models
        if hasattr(self, 'factories'):
            for factory in self.factories:
                factory.train_model()
            self.output_text.insert(tk.END, "Models trained successfully.\n")
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def predict_models(self):
        # Predict using the models
        if hasattr(self, 'factories'):
            self.predictions = [factory.predict() for factory in self.factories]
            self.output_text.insert(tk.END, "Predictions made successfully.\n")
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def process_metrics(self):
        # Process metrics
        if hasattr(self, 'factories'):
            metrics_filename = "metrics.csv"
            for factory in self.factories:
                factory.metrics(metrics_filename)
            self.output_text.insert(tk.END, "Metrics processed successfully.\n")
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def plot_results(self):
        self.progress_bar.config(mode="determinate", maximum=100, value=0)

        # Check if the models have been initialized
        if hasattr(self, 'factories'):
            self.train_models()
            self.predict_models()
            self.process_metrics()

            # Plot results
            fig1 = Plotting.corr_w_test_score_plot(self.factories[0].model.cv_results_)
            self.plots.append(fig1)
            fig2 = Plotting.plot_confusion_matrix(self.context.y_test, self.factories, self.predictions)
            self.plots.append(fig2)
            fig3 = Plotting.plot_residuals(self.context.y_test, self.factories, self.predictions)
            self.plots.append(fig3)

            for fig in self.plots:
                fig.show()

            self.output_text.insert(tk.END, "Results plotted successfully.\n")
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

        # Stop the progress bar when the process is done
        self.progress_bar.stop()

    def get_best_type(self) -> str:
        # Get the best model type based on the data type
        metrics_filename = "metrics.csv"
        metrics_df = pd.read_csv(metrics_filename)  # Read the metrics from the CSV file

        # Determine the best model based on the data type
        if self.data_type.get() == "regression":
            best_model = metrics_df.loc[metrics_df['r2'].idxmax()]
        else:
            best_model = metrics_df.loc[metrics_df['accuracy'].idxmax()]

        return best_model['type']

    def get_best_params(self, model_name: str) -> dict:
        # Get the best hyperparameters for a specific model
        if hasattr(self, 'factories'):
            for factory in self.factories:
                if factory.model.estimator.steps[-1][0] == model_name:
                    return factory.model.best_params_
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def get_cv_results(self, model_name: str) -> dict:
        # Get the best hyperparameters for a specific model
        if hasattr(self, 'factories'):
            for factory in self.factories:
                if factory.model.estimator.steps[-1][0] == model_name:
                    return factory.model.cv_results_
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def recommended_model(self):
        # Recommend the best-performing model based on metrics
        if hasattr(self, 'factories'):
            best_model_type = self.get_best_type()
            best_params = self.get_best_params(best_model_type)
            self.output_text.insert(tk.END, f"Recommended Model: {best_model_type}\n")
            self.output_text.insert(tk.END, f"Best Parameters: {best_params}\n")
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def choose_model(self):
        # Allow the user to choose from a list of available models
        if hasattr(self, 'factories'):
            # Get the names of all available models
            model_names = [model.estimator.steps[-1][0] for model in self.factories]

            # Create a variable to store the selected model type
            selected_model_type = tk.StringVar(self.root)
            selected_model_type.set(model_names[0])  # Set the default value

            # Create an OptionMenu to choose the model type
            tk.Label(self.root, text="Choose Model Type:").pack()
            model_type_menu = tk.OptionMenu(self.root, selected_model_type, *model_names)
            model_type_menu.pack()

            # Wait for the user to choose a model type
            self.root.wait_variable(selected_model_type)

            # Get the selected model type
            selected_model_type = selected_model_type.get()

            # Get the best parameters for the selected model type
            best_params = self.get_best_params(selected_model_type)

            # Display the best parameters in the text box
            self.output_text.insert(tk.END, f"Best Parameters for {selected_model_type}:\n{best_params}\n\n")

            # Get cross-validation results for the selected model type
            cv_results = self.get_cv_results(selected_model_type)

            # Plot the correlation with mean test score for parameter columns
            fig = Plotting.corr_w_test_score_plot(cv_results)
            plt.show()

            # Ask the user for the file path and name
            file_path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                     filetypes=[("Joblib files", "*.joblib"),
                                                                ("All files", "*.*")])

            # Dump the best model to the chosen file
            dump(self.get_best_model(selected_model_type), file_path)

            # Inform the user of the successful save
            messagebox.showinfo("Success", f"Model saved successfully to {file_path}.")
        else:
            self.output_text.insert(tk.END, "Please initialize models first.\n")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MLApp()
    app.run()
