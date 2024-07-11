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

# Carregar o modelo de um arquivo
with open('../trained_models/rf_model_pouro.pkl', 'rb') as file:
    svr_loaded = pickle.load(file)

# Usar o modelo carregado para fazer previsões

# Prever valores para um conjunto de teste
x_test = read_csv_features('../../../data/in_out_processed/input/input_face_001.csv')
y_real = read_csv_output('../../../data/in_out_processed/output/output_face_001.csv')

start_time = time.perf_counter()

y_pred = svr_loaded.predict(x_test)

end_time = time.perf_counter()
print("Elapsed Time:", end_time - start_time, "seconds\n")

# Calcular os erros
mse = mean_squared_error(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)
mape = np.mean(np.abs((np.array(y_real) - np.array(y_pred)) / np.array(y_real))) * 100

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)
print("Mean Absolute Percentage Error (MAPE):", mape)

# Plotar as curvas
plt.figure(figsize=(10, 6))
plt.plot(y_real, label='Valores Reais', color='red')
plt.plot(y_pred, label='Valores Previstos', linestyle='--', color='blue')
plt.title('Comparação dos Valores Reais e Previstos')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Plotar gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(y_real, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--')
plt.title('Dispersão dos Valores Reais vs. Previstos')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.show()

# Análise dos Resíduos
residuos = np.array(y_real) - np.array(y_pred)

# Plotar os resíduos
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuos)), residuos, color='blue', edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Análise dos Resíduos')
plt.xlabel('Índice')
plt.ylabel('Resíduo')
plt.show()

# Plotar histograma dos resíduos
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=50, color='blue', edgecolor='k', alpha=0.7)
plt.title('Distribuição dos Resíduos')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')
plt.show()

# Identificar possíveis outliers
outliers = np.abs(residuos) > (np.mean(np.abs(residuos)) + 3 * np.std(np.abs(residuos)))
print("Número de outliers identificados:", np.sum(outliers))

# Estatísticas Descritivas dos Erros
residuos_df = pd.DataFrame(residuos, columns=['Resíduo'])
print(residuos_df.describe())

# Recalcular MAPE excluindo outliers
y_real_filtered = np.array(y_real)[~outliers]
y_pred_filtered = np.array(y_pred)[~outliers]

mape_filtered = np.mean(np.abs((y_real_filtered - y_pred_filtered) / y_real_filtered)) * 100
print("Mean Absolute Percentage Error (MAPE) after excluding outliers:", mape_filtered)
