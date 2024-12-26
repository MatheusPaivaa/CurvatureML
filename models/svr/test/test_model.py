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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_csv_features, read_csv_output

# Load the model from a file
with open('../trained_models/rf_model.pkl', 'rb') as file:
    svr_loaded = pickle.load(file)

# Use the loaded model to make predictions

# Predict values for a test set
x_test = read_csv_features('../../../data/in_out_processed/input/input_face_2373.csv')
y_real = read_csv_output('../../../data/in_out_processed/output/output_face_2373.csv')

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
plt.title('Comparison of Real and Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_real, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--')
plt.title('Scatter Plot of Real vs. Predicted Values')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.show()

# Residual Analysis
residuals = np.array(y_real) - np.array(y_pred)

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='blue', edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Analysis')
plt.xlabel('Index')
plt.ylabel('Residual')
plt.show()

# Plot histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, color='blue', edgecolor='k', alpha=0.7)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Identify potential outliers
outliers = np.abs(residuals) > (np.mean(np.abs(residuals)) + 3 * np.std(np.abs(residuals)))
print("Number of identified outliers:", np.sum(outliers))

# Descriptive Statistics of Errors
residuals_df = pd.DataFrame(residuals, columns=['Residual'])
print(residuals_df.describe())

# Recalculate MAPE excluding outliers
y_real_filtered = np.array(y_real)[~outliers]
y_pred_filtered = np.array(y_pred)[~outliers]

mape_filtered = np.mean(np.abs((y_real_filtered - y_pred_filtered) / y_real_filtered)) * 100
print("Mean Absolute Percentage Error (MAPE) after excluding outliers:", mape_filtered)
