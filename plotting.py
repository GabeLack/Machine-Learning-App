import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Plotting:
    @staticmethod
    def corr_w_test_score_plot(cv_results_:pd.DataFrame):
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
    def classification_plot_conf(y_test, models, predictions):
        fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 4 * len(models)))
        for idx, model_name in enumerate(models):
            y_pred = predictions[model_name]
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=axes[idx])
            axes[idx].set_title(f"Confusion Matrix for {model_name}")
        plt.tight_layout()
        return fig

    @staticmethod
    def regression_plot_resid(y_test, models, predictions):
        all_y_preds = pd.DataFrame(index=y_test.index)
        for model_name in models:
            y_pred = predictions[model_name]
            all_y_preds[model_name] = y_pred
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        for i, model_name in enumerate(models):
            sns.regplot(x=y_test, y=all_y_preds[model_name], ax=axes[i])
            axes[i].set_title(f"Residuals for {model_name}")
        plt.tight_layout()
        return fig
