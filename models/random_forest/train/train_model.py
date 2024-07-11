import numpy as np
import os
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_csv_features, read_csv_output

# Diretórios
input_dir = '../../../data/in_out_processed/input'
output_dir = '../../../data/in_out_processed/output'

X_all_csv = []
y_all_csv = []

# Leitura dos dados
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

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all_csv, y_all_csv, test_size=0.3, random_state=42)

# Criando e treinando o modelo RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=16843)
rf_model.fit(X_train, y_train)

# Avaliando o modelo
score = rf_model.score(X_test, y_test)
print("Acurácia:", score)

# Salvar o modelo treinado, se necessário
with open('../trained_models/rf_model_pouro.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
