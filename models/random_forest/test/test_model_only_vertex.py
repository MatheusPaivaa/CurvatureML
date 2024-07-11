import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_obj_vertex, read_csv_output

# Carregar o modelo de um arquivo
with open('./trained_models/rf_model_obj.pkl', 'rb') as file:
    svr_loaded = pickle.load(file)

# Usar o modelo carregado para fazer previsões

# Prever valores para um conjunto de teste
x_test = read_obj_vertex('../face_objects_antigo/H00004.obj')
y_real = read_csv_output('../curvaturas_antigo/curvatura_H00004.csv')
y_pred = svr_loaded.predict(x_test)

# Calcular os erros
mse = mean_squared_error(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Plotar as curvas
plt.figure(figsize=(10, 6))
plt.plot(y_real, label='Valores Reais', color='red')
plt.plot(y_pred, label='Valores Previstos', linestyle='--', color='blue')
plt.title('Comparação dos Valores Reais e Previstos')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.show()