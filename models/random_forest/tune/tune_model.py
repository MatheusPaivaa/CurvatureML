import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_csv_features, read_csv_output

# Directories
input_dir = '../../../data/in_out_processed/input'  # Input directory containing feature files
output_dir = '../../../data/in_out_processed/output'  # Output directory containing target files

X_all_csv = []
y_all_csv = []

# Reading data
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        features_path = os.path.join(input_dir, filename)
        X = read_csv_features(features_path)
        X_all_csv.extend(X)

        output_filename = 'output_' + filename.split('input_')[1]
        output_path = os.path.join(output_dir, output_filename)
        y = read_csv_output(output_path)
        y_all_csv.extend(y)

X_all_csv = np.array(X_all_csv)
y_all_csv = np.array(y_all_csv)

# print(X_all_csv.shape)
# print(y_all_csv.shape)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all_csv, y_all_csv, test_size=0.3, random_state=42)

# Creating and training the RF model
rf_model = RandomForestRegressor(random_state=16843)

# Defining parameters for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Config GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Executing GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Training model with best parameters
best_rf_model = grid_search.best_estimator_

# Evaluating the model
y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
score = best_rf_model.score(X_test, y_test)
print("Acuracy:", score)

# Save the trained model, if needed
with open('rf_model_pouro_tuned.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)
