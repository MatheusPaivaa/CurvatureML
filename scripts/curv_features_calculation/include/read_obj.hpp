#ifndef READ_OBJ_HPP
#define READ_OBJ_HPP

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


// Definition of a Vertex struct representing a point in 3D space
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

// Definition of a Face struct representing a triangular face in a 3D model
struct Face {
    int v1, v2, v3;
};

// Definition of an Edge struct representing an edge in a 3D model
struct Edge {
    int v1, v2;

    // Constructor that orders vertices to ensure uniqueness
    Edge(int a, int b) : v1(std::min(a, b)), v2(std::max(a, b)) {}

    // Operators necessary for using the structure as a key in a std::unordered_map
    bool operator==(const Edge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }
};

// Definition of a hash function for Edge to enable its use as a key in std::unordered_map
struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
        return std::hash<int>()(edge.v1) ^ std::hash<int>()(edge.v2);
    }
};

// Declaration of a function to read a .obj file and populate vectors of Vertex and Face structs
void read_obj(const std::string& filename, std::vector<Vertex>& V, std::vector<Face>& F);

#endif // READ_OBJ_HPP
