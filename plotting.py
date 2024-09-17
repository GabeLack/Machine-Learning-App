import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Plotting:
    @staticmethod
    def corr_w_test_score_plot(cv_results_:dict):
        # Get stats of all estimators, dummify
        estimators = pd.DataFrame(cv_results_)
        estimators.drop('params', axis=1, inplace=True)
        estimators = pd.get_dummies(estimators, dtype='int')

        # Corr with mean_test_score column
        correlation_with_mean_test_score = estimators.corrwith(estimators['mean_test_score'])
        param_correlations = correlation_with_mean_test_score.filter(like='param_')

        # Visualize the correlation matrix using a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=param_correlations, y=param_correlations.index, ax=ax)
        ax.set_title(f'Correlation with Mean Test Score for Parameter Columns')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('param_ Columns')
        return fig

    @staticmethod
    def plot_confusion_matrix(y_test, factories: list, predictions: list[pd.DataFrame]):
        # Determine the number of rows and columns for subplots
        n_factories = len(factories)
        n_cols = 2
        n_rows = (n_factories + 1) // n_cols

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2 * n_factories))
        axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing

        for idx, (factory, y_pred) in enumerate(zip(factories, predictions)):
            # Create a ConfusionMatrixDisplay object
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
            # Plot the confusion matrix
            disp.plot(ax=axes[idx])
            # Get the name of the specific model
            model_name = factory.model.estimator.steps[-1][1].__class__.__name__
            # Set the title for the subplot
            axes[idx].set_title(f"Confusion Matrix for {model_name}")

        # Hide any unused subplots
        for ax in axes[n_factories:]:
            ax.axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_residuals(y_test, factories: list, predictions: list[pd.DataFrame]):
        # Determine the number of rows and columns for subplots
        n_factories = len(factories)
        n_cols = 2
        n_rows = (n_factories + 1) // n_cols

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing

        for i, (factory, y_pred) in enumerate(zip(factories, predictions)):
            # Plot residuals using seaborn's regplot
            sns.regplot(x=y_test, y=y_pred, ax=axes[i])
            # Get the name of the specific model
            model_name = factory.model.estimator.steps[-1][1].__class__.__name__
            # Set the title for the subplot
            axes[i].set_title(f"Residuals for {model_name}")

        # Hide any unused subplots
        for ax in axes[n_factories:]:
            ax.axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig
