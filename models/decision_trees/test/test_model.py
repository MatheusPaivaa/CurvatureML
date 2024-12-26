import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
import os

# Add the directory above the current directory to sys.path to import the read_data module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_csv_features, read_csv_output

# Load the model from a file
with open('../trained_models/dt_model_media.pkl', 'rb') as file:
    svr_loaded = pickle.load(file)

# Use the loaded model to make predictions

# Predict values for a test set
x_test = read_csv_features('../../../data/in_out_processed/input/input_face_018.csv')
y_real = read_csv_output('../../../data/in_out_processed/output/output_face_018.csv')

start_time = time.perf_counter()

y_pred = svr_loaded.predict(x_test)

end_time = time.perf_counter()
print("Elapsed Time:", end_time - start_time, "seconds\n")

# Calculate errors
mse = mean_squared_error(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)
mape = np.mean(np.abs((np.array(y_real) - np.array(y_pred)) / np.array(y_real))) * 100

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)
print("Mean Absolute Percentage Error (MAPE):", mape)


# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(y_real, label='Real Values', color='red')
plt.plot(y_pred, label='Predicted Values', linestyle='--', color='blue')
plt.title(f'Comparison of Real and Predicted Values (R² = {r2:.2f}) - Decision Tree')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()
