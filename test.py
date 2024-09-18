from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
import numpy as np


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("model created")
    return model

# Create the KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)

# Create a pipeline
pipeline = make_pipeline(StandardScaler(), model)

# Define the grid search parameters
param_grid = {'kerasclassifier__batch_size': [10, 20], 'kerasclassifier__epochs': [10, 20]}

# Create GridSearchCV
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=3)

# Dummy data for testing
X = np.random.random((100, 8))
y = np.random.randint(2, size=(100, 1))

# Fit the grid search
grid_result = grid.fit(X, y)

# Print the best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))