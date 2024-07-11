#ifndef LEITURA_OBJ_HPP
#define LEITURA_OBJ_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream> 
#include <utility>
#include <numeric>
#include <fstream>
#include <sstream>
#include <complex>

struct Vertex {
    double x, y, z;

    // Operator to add two vertices
    Vertex operator+(const Vertex& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    // Operator to scale a vertex by a factor
    Vertex operator/(double factor) const {
        return {x / factor, y / factor, z / factor};
    }

    // Operator to subtract two vertices
    Vertex operator-(const Vertex& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
};


struct Face {
    int v1, v2, v3;
};

struct Edge {
    int v1, v2;

    // Construtor que ordena os vértices para garantir unicidade.
    Edge(int a, int b) : v1(std::min(a, b)), v2(std::max(a, b)) {}

    // Operadores necessários para usar a estrutura como chave em um std::unordered_map.
    bool operator==(const Edge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }
};

struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
        return std::hash<int>()(edge.v1) ^ std::hash<int>()(edge.v2);
    }
};

void read_obj(const std::string& filename, std::vector<Vertex>& V, std::vector<Face>& F);

#endif // LEITURA_OBJ_HPP
