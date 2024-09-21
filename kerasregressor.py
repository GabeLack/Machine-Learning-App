import numpy as np
import pandas as pd
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Create a synthetic dataset
np.random.seed(42)
X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
y_train = pd.Series(np.random.rand(100))

def create_model(neurons=64, dropout=0.5, optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap the Keras model with KerasRegressor
model = KerasRegressor(build_fn=create_model, neurons=64, dropout=0.5)

# Create a pipeline with Normalizer and KerasRegressor
pipeline = make_pipeline(Normalizer(), model)

# Use GridSearchCV to find the best parameters
param_grid = {
    'kerasregressor__epochs': [50, 100],
    'kerasregressor__batch_size': [10, 20],
    'kerasregressor__neurons': [64, 128],
    'kerasregressor__dropout': [0.5, 0.7],
    'kerasregressor__optimizer': ['adam', 'rmsprop']}
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best params: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")
