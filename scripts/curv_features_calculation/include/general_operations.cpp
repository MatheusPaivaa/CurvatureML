#include "general_operations.hpp"

// Function to calculate the angle in radians given the lengths of three sides of a triangle
double calculateAngleInRadians(double a, double b, double c) {
    return std::acos((a * a + b * b - c * c) / (2 * a * b));
}

// Function to calculate the distance between two vertices
double distanceBetweenVertices(const Vertex& v1, const Vertex& v2) {
    return std::sqrt(std::pow(v2.x - v1.x, 2) + std::pow(v2.y - v1.y, 2) + std::pow(v2.z - v1.z, 2));
}

// Function to calculate the area of a face using Heron's formula
double calculateFaceArea(const TriangleSides& sides) {
    double s = (sides.a + sides.b + sides.c) / 2;
    return std::sqrt(s * (s - sides.a) * (s - sides.b) * (s - sides.c));
}

// Function to calculate the lengths of the sides of a triangle face
TriangleSides calculateSides(const std::vector<Vertex>& V, const Face& f) {
    TriangleSides sides;

    sides.a = distanceBetweenVertices(V[f.v2], V[f.v3]);
    sides.b = distanceBetweenVertices(V[f.v1], V[f.v3]);
    sides.c = distanceBetweenVertices(V[f.v1], V[f.v2]);

    return sides;
}

// Function to map vertices to the faces they belong to
std::unordered_map<int, std::set<int>> mapVerticesToFaces(const std::vector<Vertex>& V, const std::vector<Face>& F) {
    std::unordered_map<int, std::set<int>> vertexToFacesMap;

    for (int i = 0; i < F.size(); i++) {
        const auto& face = F[i];
        vertexToFacesMap[face.v1].insert(i);
        vertexToFacesMap[face.v2].insert(i);
        vertexToFacesMap[face.v3].insert(i);
    }

    return vertexToFacesMap;
}

// Function to calculate the sum of angles at each vertex
std::vector<double> calculateAngleSum(const std::vector<Vertex>& V, const std::vector<Face>& F, const std::unordered_map<int, std::set<int>>& vertexToFacesMap) {
    std::vector<double> angleSum(V.size(), 0.0); // Vector to store the sum of angles for each vertex

    for (const auto& item : vertexToFacesMap) {
        int vertexIndex = item.first;
        const auto& faces = item.second;

        for (int faceIndex : faces) {
            const Face& face = F[faceIndex];
            int vIndices[3] = {face.v1, face.v2, face.v3};
            std::vector<int> vIndexVec(vIndices, vIndices + 3);

            auto it = std::find(vIndexVec.begin(), vIndexVec.end(), vertexIndex);
            int pos = std::distance(vIndexVec.begin(), it);
            Vertex v1 = V[vertexIndex];
            Vertex v2 = V[vIndexVec[(pos + 1) % 3]];
            Vertex v3 = V[vIndexVec[(pos + 2) % 3]];

            double a = distanceBetweenVertices(v2, v3);
            double b = distanceBetweenVertices(v1, v3);
            double c = distanceBetweenVertices(v1, v2);

            angleSum[vertexIndex] += calculateAngleInRadians(b, c, a);
        }
    }

    return angleSum;
}

// Function to calculate the mixed area for each vertex
std::vector<double> calculateMixedArea(const std::vector<Vertex>& V, const std::vector<Face>& F) {
    std::vector<TriangleSides> faceSides(F.size());
    std::vector<double> faceAreas(F.size());
    std::vector<double> mixedArea(V.size(), 0.0);

    for (size_t i = 0; i < F.size(); ++i) {
        faceSides[i] = calculateSides(V, F[i]);
        faceAreas[i] = calculateFaceArea(faceSides[i]);

        mixedArea[F[i].v1] += faceAreas[i] / 3.0;
        mixedArea[F[i].v2] += faceAreas[i] / 3.0;
        mixedArea[F[i].v3] += faceAreas[i] / 3.0;
    }

    return mixedArea;
}

// Function to map adjacent edges to their faces
void mapAdjacentEdges(const std::vector<Face>& faces, std::unordered_map<Edge, std::vector<int>, EdgeHash>& edgeMap) {
    for (size_t i = 0; i < faces.size(); ++i) {
        const Face& face = faces[i];

        // Create the edges for the current face, ensuring smaller indices come first
        std::vector<Edge> edges = {
            {std::min(face.v1, face.v2), std::max(face.v1, face.v2)},
            {std::min(face.v2, face.v3), std::max(face.v2, face.v3)},
            {std::min(face.v3, face.v1), std::max(face.v3, face.v1)}
        };

        // Add the current face index to the lists of the corresponding edges
        for (const Edge& edge : edges) {
            edgeMap[edge].push_back(static_cast<int>(i));
        }
    }
}

// Function to print vertex connections
void printVertexConnections(const std::vector<Vertex>& V, const std::vector<Face>& F) {
    std::unordered_map<int, std::unordered_set<int>> vertexToFaces;

    // Map each vertex to the faces it belongs to
    for (int i = 0; i < F.size(); i++) {
        const Face& face = F[i];
        vertexToFaces[face.v1].insert(i);
        vertexToFaces[face.v2].insert(i);
        vertexToFaces[face.v3].insert(i);
    }

    // For each vertex, print its coordinates and the other vertices of the shared faces
    for (int i = 0; i < V.size(); i++) {
        const Vertex& v = V[i];
        std::cout << "Vertex " << i + 1 << " (" << v.x << ", " << v.y << ", " << v.z << "): Faces with ";

        std::unordered_set<int> adjacentVertices;
        for (int faceIndex : vertexToFaces[i]) {
            const Face& face = F[faceIndex];
            if (face.v1 != i) adjacentVertices.insert(face.v1);
            if (face.v2 != i) adjacentVertices.insert(face.v2);
            if (face.v3 != i) adjacentVertices.insert(face.v3);
        }

        for (int adjV : adjacentVertices) {
            const Vertex& va = V[adjV];
            std::cout << "Vertex " << adjV + 1 << " (" << va.x << ", " << va.y << ", " << va.z << "), ";
        }
        std::cout << std::endl;
    }
}

// Function to calculate the normal of a face using the cross product
Vertex calculateNormal(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
    Vertex u = {v2.x - v1.x, v2.y - v1.y, v2.z - v1.z};
    Vertex v = {v3.x - v1.x, v3.y - v1.y, v3.z - v1.z};
    return {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
}

// Function to calculate the norm (magnitude) of a vector
double norm(const Vertex& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Function to normalize a vector
Vertex normalize(const Vertex& v) {
    double length = norm(v);
    return {v.x / length, v.y / length, v.z / length};
}

// Function to calculate the distance between two vertices
double calculateDistance(const Vertex& v1, const Vertex& v2) {
    return sqrt(pow(v1.x - v2.x, 2) + pow(v1.y - v2.y, 2) + pow(v1.z - v2.z, 2));
}

// Function to calculate the cross product
Vertex cross(const Vertex& v1, const Vertex& v2) {
    return {v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x};
}

// Function to calculate the dot product
double dot(const Vertex& v1, const Vertex& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// Function to calculate the normal vector given three points (vertex of a face)
Vertex vecnorm(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
    return normalize(cross(v2 - v1, v3 - v1));
}
