import numpy as np
import os
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_obj_vertex, read_csv_output

# Directories
output_dir = '../../../data/in_out_processed/output'
obj_dir = '../../../data/face_processed'

X_all_obj = []
y_all_obj = []

# Reading data from .obj files
for filename in sorted(os.listdir(obj_dir)):
    if filename.endswith('.obj'):
        obj_path = os.path.join(obj_dir, filename)
        vertices = read_obj_vertex(obj_path)
        X_all_obj.extend(vertices)

        # Associate file .obj with a .csv output file
        output_filename = 'output_' + filename.split('.obj')[0] + '.csv'

        output_path = os.path.join(output_dir, output_filename)
        y = read_csv_output(output_path)
        y_all_obj.extend(y)

X_all_obj = np.array(X_all_obj)
y_all_obj = np.array(y_all_obj)

# print(X_all_obj.shape)
# print(y_all_obj.shape)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all_obj, y_all_obj, test_size=0.3, random_state=42)

# Creating and training the SVR model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    oob_score=False,
    random_state=42
)

# Training model
model.fit(X_train, y_train)

# Predicting the results for the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

score = model.score(X_test, y_test)
print("Acuracy:", score)

# Save the trained model, if needed
with open('../trained_models/rf_model_vertex_media.pkl', 'wb') as file:
    pickle.dump(model, file)
