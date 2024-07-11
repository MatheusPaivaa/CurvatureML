import numpy as np
import os
import csv
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

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
            data.append(float(row[1]))
    return data

# Diretórios
features_dir = '../resultado/vetor_carac'
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

X_all_csv = np.array(X_all_csv)
y_all_csv = np.array(y_all_csv)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all_csv, y_all_csv, test_size=0.3, random_state=42)

# Definindo o modelo
rf_model = RandomForestRegressor(random_state=16843)

# Definindo os parâmetros para o GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Configurando o GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Executando o GridSearchCV
grid_search.fit(X_train, y_train)

# Obtendo os melhores parâmetros
best_params = grid_search.best_params_
print("Melhores parâmetros:", best_params)

# Treinando o modelo com os melhores parâmetros
best_rf_model = grid_search.best_estimator_

# Avaliando o modelo
y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
score = best_rf_model.score(X_test, y_test)
print("Acurácia:", score)

# Salvar o modelo treinado, se necessário
with open('rf_model_pouro_tuned.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)
