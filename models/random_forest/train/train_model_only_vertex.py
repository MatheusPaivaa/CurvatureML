import numpy as np
import os
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'read_data')))
from read_data import read_obj_vertex, read_csv_output

# Diretórios
output_dir = '../../../data/in_out_processed/output'
obj_dir = '../../../data/face_processed'

X_all_obj = []
y_all_obj = []

# Leitura de características de arquivos .obj
for filename in sorted(os.listdir(obj_dir)):
    if filename.endswith('.obj'):
        obj_path = os.path.join(obj_dir, filename)
        vertices = read_obj_vertex(obj_path)
        X_all_obj.extend(vertices)

        # Associa cada arquivo .obj a um arquivo de saída .csv
        output_filename = 'pOuro_' + filename.split('.obj')[0] + '.csv'

        output_path = os.path.join(output_dir, output_filename)
        y = read_csv_output(output_path)
        y_all_obj.extend(y)

X_all_obj = np.array(X_all_obj)
y_all_obj = np.array(y_all_obj)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all_obj, y_all_obj, test_size=0.3, random_state=42)

# Criar o modelo de Random Forest
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

# Treinar o modelo
model.fit(X_train, y_train)

# Prever os resultados para o conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

score = model.score(X_test, y_test)
print("Acurácia:", score)

# Salvar o modelo treinado
with open('../trained_models/rf_model_vertex.pkl', 'wb') as file:
    pickle.dump(model, file)
