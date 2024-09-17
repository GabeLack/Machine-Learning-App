import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

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
    def plot_confusion_matrix(y_test, models: list[GridSearchCV], predictions: list[pd.DataFrame]):
        # Plot confusion matrix for each model
        fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 4 * len(models)))
        for idx, (model, y_pred) in enumerate(zip(models, predictions)):
            # Create a ConfusionMatrixDisplay object
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
            # Plot the confusion matrix
            disp.plot(ax=axes[idx])
            # Get the name of the specific model
            model_name = model.estimator.steps[-1][1].__class__.__name__
            # Set the title for the subplot
            axes[idx].set_title(f"Confusion Matrix for {model_name}")
        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_residuals(y_test, models: list[GridSearchCV], predictions: list[pd.DataFrame]):
        # Plot residuals for each model
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        for i, (model, y_pred) in enumerate(zip(models, predictions)):
            # Plot residuals using seaborn's regplot
            sns.regplot(x=y_test, y=y_pred, ax=axes[i])
            # Get the name of the specific model
            model_name = model.estimator.steps[-1][1].__class__.__name__
            # Set the title for the subplot
            axes[i].set_title(f"Residuals for {model_name}")
        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig
