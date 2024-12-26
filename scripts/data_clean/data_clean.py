import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import vizualize_clean as vc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from joblib import dump


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/', 'read_data')))
from read_data import read_csv_features, read_csv_output

input_dir = '../../../data_raw/in_out_processed/input'
output_dir = '../../../data_raw/in_out_processed/output'

X_all_csv = []
y_all_csv = []

"""
Process:

1. Read data from input and output directories
2. Clean data
    2.1. Remove missing values (Impute)
    2.2. Remove outliers (LOF)
    2.3. Remove duplicates
    2.4. Normalize data

3. Save cleaned data

"""

def read_data():
    global X_all_csv, y_all_csv
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

def remove_missing_values():
    global X_all_csv, y_all_csv
    
    # Identificar valores faltantes no conjunto de features (X_all_csv)
    missing_X = np.isnan(X_all_csv)
    missing_X_count = np.sum(missing_X)

    # Identificar valores faltantes no conjunto de saídas (y_all_csv)
    missing_y = np.isnan(y_all_csv)
    missing_y_count = np.sum(missing_y)

    # Imprimir informações sobre os valores faltantes
    print(f"Valores faltantes em X_all_csv: {missing_X_count}")
    if missing_X_count > 0:
        print("Índices de valores faltantes em X_all_csv:")
        print(np.where(missing_X))

    print(f"Valores faltantes em y_all_csv: {missing_y_count}")
    if missing_y_count > 0:
        print("Índices de valores faltantes em y_all_csv:")
        print(np.where(missing_y))

    # Remover linhas com valores faltantes
    valid_indices = ~np.any(missing_X, axis=1) & ~missing_y.flatten()
    X_all_csv = X_all_csv[valid_indices]
    y_all_csv = y_all_csv[valid_indices]

    print(f"\n📋 Dados restantes após remoção de valores faltantes: {len(X_all_csv)} entradas")

def remove_outliers():
    global X_all_csv, y_all_csv

    # Calcular Z-score
    mean = np.mean(y_all_csv)
    std = np.std(y_all_csv)

    # Limites para os valores aceitáveis
    lower, upper = mean - 3 * std, mean + 3 * std

    # Identificar índices dentro dos limites
    valid_indices = (y_all_csv >= lower) & (y_all_csv <= upper)

    #vc.plot_outliers(y_all_csv, y_all_csv[valid_indices])

    # Filtrar os dados para remover os outliers
    X_all_csv = X_all_csv[valid_indices]
    y_all_csv = y_all_csv[valid_indices]

    print(f"📋 Dados restantes após remoção de outliers nos labels: {len(y_all_csv)} entradas")

def remove_duplicates():
    global X_all_csv, y_all_csv

    # Combinar X e y para identificar duplicatas completas
    data_combined = np.hstack((X_all_csv, y_all_csv.reshape(-1, 1)))

    # Identificar duplicatas
    _, unique_indices = np.unique(data_combined, axis=0, return_index=True)

    # Ordenar os índices únicos para manter a ordem original
    unique_indices = np.sort(unique_indices)

    # Filtrar os dados para remover duplicatas
    X_all_csv = X_all_csv[unique_indices]
    y_all_csv = y_all_csv[unique_indices]

    print(f"📋 Dados restantes após remoção de duplicatas: {len(y_all_csv)} entradas")

def normalize_data():
    global X_all_csv

    # Normalizar as features
    scaler = StandardScaler()
    X_all_csv = scaler.fit_transform(X_all_csv)

    # Salvar o scaler para uso futuro em testes
    scaler_path = "./scaler.joblib"
    dump(scaler, scaler_path)
    print(f"Scaler salvo em: {scaler_path}")

    # Visualizar estatísticas das features normalizadas
    # df = pd.DataFrame(X_all_csv)
    # print(df.describe())

def save_cleaned_data():
    global X_all_csv, y_all_csv

    # Salvar features em um arquivo CSV
    features_df = pd.DataFrame(X_all_csv)
    features_file = os.path.join(output_dir, "cleaned_features.csv")
    features_df.to_csv(features_file, index=False)

    # Salvar labels em um arquivo CSV
    labels_df = pd.DataFrame(y_all_csv, columns=['Label'])
    labels_file = os.path.join(output_dir, "cleaned_labels.csv")
    labels_df.to_csv(labels_file, index=False)

    print(f"Features limpas salvas em: {features_file}")
    print(f"Labels limpas salvas em: {labels_file}")

def clean_data():

    print("Reading data...\n")
    read_data()

    print(f"📋 Dados antes da limpeza: {len(X_all_csv)} entradas\n")

    print("Cleaning data:\n")

    print("- Removing missing values...\n")
    remove_missing_values()

    print("\n- Removing outliers...\n")
    remove_outliers()

    print("\n- Removing duplicates...\n")
    remove_duplicates()

    print("\n- Normalizing data...\n")
    normalize_data()

    # Save cleaned data
    # save_cleaned_data()

    return X_all_csv, y_all_csv