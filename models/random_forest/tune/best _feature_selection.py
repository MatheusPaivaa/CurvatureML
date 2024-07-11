import numpy as np
import pandas as pd
import os
import csv
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

# Funções para ler arquivos
def read_csv_features(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Pula a primeira linha (header)
        for row in reader:
            data.append([float(val) for val in row])
    return data

def read_csv_output(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(float(row[0]))
    return data

# Diretórios
features_dir = '../resultados/vetor_carac'
output_dir = '../resultados/curvaturas'

X_all_csv = []
y_all_csv = []

# Leitura dos dados
for filename in os.listdir(features_dir):
    if filename.endswith('.csv'):
        features_path = os.path.join(features_dir, filename)
        X = read_csv_features(features_path)
        X_all_csv.extend(X)

        output_filename = 'curvatura_' + filename.split('carac_')[1]
        output_path = os.path.join(output_dir, output_filename)
        y = read_csv_output(output_path)
        y_all_csv.extend(y)

X_all_df = pd.DataFrame(X_all_csv)
y_all_df = pd.Series(y_all_csv)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all_df, y_all_df, test_size=0.3, random_state=42)

# Treinando um modelo Random Forest para avaliar a importância das características
rf_model = RandomForestRegressor(random_state=16843, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Obtendo as importâncias das características
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame(importances, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

print("Importâncias das Características:")
print(feature_importances)

# Seleção de Características com RFE
n_features_to_select = 3  # Número de características desejadas
rfe = RFE(estimator=rf_model, n_features_to_select=n_features_to_select)
X_rfe = rfe.fit_transform(X_train, y_train)

print("Características selecionadas por RFE:")
print(X_train.columns[rfe.support_])

# Treinando o modelo com as características selecionadas
rf_model_rfe = RandomForestRegressor(random_state=16843, n_jobs=-1)
rf_model_rfe.fit(X_rfe, y_train)

# Avaliando o modelo
X_test_rfe = rfe.transform(X_test)
y_pred = rf_model_rfe.predict(X_test_rfe)
mse = mean_squared_error(y_test, y_pred)
score = rf_model_rfe.score(X_test_rfe, y_test)

print("Mean Squared Error:", mse)
print("Acurácia:", score)
