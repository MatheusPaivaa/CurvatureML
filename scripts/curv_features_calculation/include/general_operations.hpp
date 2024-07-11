#ifndef GENERAL_OPERATIONS_HPP
#define GENERAL_OPERATIONS_HPP

#include "read_obj.hpp"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <vector>

struct TriangleSides {
    double a, b, c; // Lengths of the sides of the triangle
};

double calculateAngleInRadians(double a, double b, double c);
double distanceBetweenVertices(const Vertex& v1, const Vertex& v2);
double calculateFaceArea(const TriangleSides& sides);
std::vector<double> calculateMixedArea(const std::vector<Vertex>& V, const std::vector<Face>& F);
TriangleSides calculateSides(const std::vector<Vertex>& V, const Face& f);
std::vector<double> calculateAngleSum(const std::vector<Vertex>& V, const std::vector<Face>& F, const std::unordered_map<int, std::set<int>>& vertexToFacesMap);
std::unordered_map<int, std::set<int>> mapVerticesToFaces(const std::vector<Vertex>& V, const std::vector<Face>& F);
void printVertexConnections(const std::vector<Vertex>& V, const std::vector<Face>& F);
void mapAdjacentEdges(const std::vector<Face>& faces, std::unordered_map<Edge, std::vector<int>, EdgeHash>& edgeMap);
Vertex calculateNormal(const Vertex& v1, const Vertex& v2, const Vertex& v3);
double calculateDistance(const Vertex& v1, const Vertex& v2);
double norm(const Vertex& v);
Vertex cross(const Vertex& v1, const Vertex& v2);
double dot(const Vertex& v1, const Vertex& v2);
Vertex vecnorm(const Vertex& v1, const Vertex& v2, const Vertex& v3);
Vertex normalize(const Vertex& v);

#endif // GENERAL_OPERATIONS_HPP
