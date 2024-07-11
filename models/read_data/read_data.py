import csv

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
            data.append(float(row[0]))  # Assumindo que a curvatura está na segunda coluna
    return data

def read_obj_vertex(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return vertices
