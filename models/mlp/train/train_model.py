import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts', 'data_clean')))
from data_clean import clean_data

n_iter = 700

X_scaled, y_all_truncated = clean_data()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all_truncated, test_size=0.2, random_state=42)

print("\n### Treinando modelo ###")
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                   max_iter=1, warm_start=True, random_state=42)
train_losses, test_losses = [], []

for i in tqdm(range(n_iter), desc="Treinamento do MLP"):
    mlp.fit(X_train, y_train)
    train_losses.append(mean_squared_error(y_train, mlp.predict(X_train)))
    test_losses.append(mean_squared_error(y_test, mlp.predict(X_test)))

# Gráfico da curva de aprendizado
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss', color='red')
plt.xlabel('Iterações')
plt.ylabel('Mean Squared Error')
plt.title('Curva de Aprendizado')
plt.legend()
plt.savefig("curva_aprendizado.png")
plt.close()

# Avaliação final
y_test_pred = mlp.predict(X_test)
results = {
    "MAE": mean_absolute_error(y_test, y_test_pred),
    "MSE": mean_squared_error(y_test, y_test_pred),
    "R2": r2_score(y_test, y_test_pred)
}

print(results)

dump(mlp, "modelo_final_mlp.pkl")

print("### Treinamento finalizado ###")
print("Resultados e imagens salvas.")
