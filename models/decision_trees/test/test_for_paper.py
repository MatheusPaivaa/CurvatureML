import pickle
import time
import numpy as np
import pandas as pd

# Path to the saved model
model_path = '../trained_models/dt_model_gaussiana.pkl'

# Load the model from a file
with open(model_path, 'rb') as file:
    svr_loaded = pickle.load(file)

# List of input files (replace with the correct file names)
input_files = [
    '../../../data/in_out_processed/input/input_face_001.csv',
    '../../../data/in_out_processed/input/input_face_002.csv',
    '../../../data/in_out_processed/input/input_face_003.csv',
    '../../../data/in_out_processed/input/input_face_004.csv',
    '../../../data/in_out_processed/input/input_face_005.csv',
    '../../../data/in_out_processed/input/input_face_006.csv',
    '../../../data/in_out_processed/input/input_face_007.csv',
    '../../../data/in_out_processed/input/input_face_008.csv',
    '../../../data/in_out_processed/input/input_face_009.csv',
    '../../../data/in_out_processed/input/input_face_010.csv'
]

# Initialize variables for time calculation
total_time = 0.0

# Iterate over each file and make predictions
for input_file in input_files:
    # Read the input file
    x_test = pd.read_csv(input_file)
    
    # Measure prediction time
    start_time = time.perf_counter()
    y_pred = svr_loaded.predict(x_test)
    end_time = time.perf_counter()
    
    # Calculate the time spent and accumulate
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    print(f"Time spent to predict {input_file}: {elapsed_time:.4f} seconds")

# Calculate the average time spent
average_time = total_time / len(input_files)
print(f"Average time used for prediction: {average_time:.4f} seconds")