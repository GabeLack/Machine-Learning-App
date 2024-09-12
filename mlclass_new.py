import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import (LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier

#! future additions?
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, f1_score,
                             classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)

from validation import Validation as val

# REGRESSIONS = ['LiR','Lasso','Ridge','ElasticNet','SVR']
# CLASSIFICATIONS = ['LoR','KNN','SVC']

class MLDirector:
    def __init__(self, builder) -> None:
        self.builder = builder

    def construct(self, data_file_path: str, model, test_size: float, random_state: int):
        self.builder.set_data(data_file_path)
        self.builder.set_model(model)
        self.builder.split_data(test_size, random_state)
        return self.builder.get_result()

class MachineLearning: # Interface
    def __init__(self) -> None:
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self, data_file_path: str):
        self.df = pd.read_csv(data_file_path)
        return self.df

class MLRegressorBuilder(ABC):
    @abstractmethod
    def set_data(self, data_file_path: str):
        pass

    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def split_data(self, test_size: float, random_state: int):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict_and_metrics(self):
        pass

    @abstractmethod
    def plot_residuals(self):
        pass

    @abstractmethod
    def get_result(self):
        pass

class MLClassifierBuilder(ABC):
    @abstractmethod
    def set_data(self, data_file_path: str):
        pass

    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def split_data(self, test_size: float, random_state: int):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict_and_metrics(self):
        pass

    @abstractmethod
    def plot_confusion_matrix(self):
        pass

    @abstractmethod
    def get_result(self):
        pass


class ConcreteMLRegressorBuilder(MLRegressorBuilder):
    def __init__(self):
        self.ml = MachineLearning()

    def set_data(self, data_file_path: str):
        self.ml.load_data(data_file_path)

    def set_model(self, model):
        self.ml.model = model

    def split_data(self, test_size: float, random_state: int):
        self.ml.X = self.ml.df.drop('target', axis=1)
        self.ml.y = self.ml.df['target']
        self.ml.X_train, self.ml.X_test, self.ml.y_train, self.ml.y_test = train_test_split(
            self.ml.X, self.ml.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.ml.model.fit(self.ml.X_train, self.ml.y_train)

    def predict_and_metrics(self):
        y_pred = self.ml.model.predict(self.ml.X_test)
        mse = mean_squared_error(self.ml.y_test, y_pred)
        r2 = r2_score(self.ml.y_test, y_pred)
        print(f'MSE: {mse}, R2: {r2}')
        return y_pred

    def plot_residuals(self):
        y_pred = self.ml.model.predict(self.ml.X_test)
        residuals = self.ml.y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.show()

    def get_result(self):
        return self.ml

class ConcreteMLClassifierBuilder(MLClassifierBuilder):
    def __init__(self):
        self.ml = MachineLearning()

    def set_data(self, data_file_path: str):
        self.ml.load_data(data_file_path)

    def set_model(self, model):
        self.ml.model = model

    def split_data(self, test_size: float, random_state: int):
        self.ml.X = self.ml.df.drop('target', axis=1)
        self.ml.y = self.ml.df['target']
        self.ml.X_train, self.ml.X_test, self.ml.y_train, self.ml.y_test = train_test_split(
            self.ml.X, self.ml.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.ml.model.fit(self.ml.X_train, self.ml.y_train)

    def predict_and_metrics(self):
        y_pred = self.ml.model.predict(self.ml.X_test)
        accuracy = accuracy_score(self.ml.y_test, y_pred)
        f1 = f1_score(self.ml.y_test, y_pred, average='weighted')
        print(f'Accuracy: {accuracy}, F1 Score: {f1}')
        return y_pred

    def plot_confusion_matrix(self):
        y_pred = self.ml.model.predict(self.ml.X_test)
        cm = confusion_matrix(self.ml.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def get_result(self):
        return self.ml
