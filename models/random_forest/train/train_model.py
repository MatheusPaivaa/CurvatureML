import numpy as np
import os
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts', 'data_clean')))
from data_clean import clean_data

X_all_csv, y_all_csv = clean_data()

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all_csv, y_all_csv, test_size=0.3, random_state=42)

# Criando e treinando o modelo RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=16843, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Avaliando o modelo
# Fazendo previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Avaliando o modelo
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R² Score:", r2)
print("Erro Médio Quadrático (MSE):", mse)
print("Erro Médio Absoluto (MAE):", mae)

# Salvar o modelo treinado, se necessário
with open('../trained_models/rf_model_pouro_media.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
