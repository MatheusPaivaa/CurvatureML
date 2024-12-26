import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_obj_vertex, read_csv_output

# Load the model from a file
with open('./trained_models/rf_model_obj.pkl', 'rb') as file:
    svr_loaded = pickle.load(file)

# Use the loaded model to make predictions

# Predict values for a test set
x_test = read_obj_vertex('../../../data/face_processed/face_2373.obj')
y_real = read_csv_output('../../../data/in_out_processed/output/output_face_2373.csv')
y_pred = svr_loaded.predict(x_test)

# Calculate the errors
mse = mean_squared_error(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(y_real, label='Real Values', color='red')
plt.plot(y_pred, label='Predicted Values', linestyle='--', color='blue')
plt.title('Comparison of Real and Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()