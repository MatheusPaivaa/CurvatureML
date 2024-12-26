import numpy as np
import pandas as pd
import os
import csv
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
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

# Training model with best features
rf_model = RandomForestRegressor(random_state=16843, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Feature importances
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame(importances, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

print("Features importances:")
print(feature_importances)

# Selected features
n_features_to_select = 3  # Número de características desejadas
rfe = RFE(estimator=rf_model, n_features_to_select=n_features_to_select)
X_rfe = rfe.fit_transform(X_train, y_train)

# Best parameters
print("Features selected:")
print(X_train.columns[rfe.support_])

# Training model with best features
rf_model_rfe = RandomForestRegressor(random_state=16843, n_jobs=-1)
rf_model_rfe.fit(X_rfe, y_train)

# Evaluating the model
X_test_rfe = rfe.transform(X_test)
y_pred = rf_model_rfe.predict(X_test_rfe)
mse = mean_squared_error(y_test, y_pred)
score = rf_model_rfe.score(X_test_rfe, y_test)

print("Mean Squared Error:", mse)
print("Acuracy:", score)
