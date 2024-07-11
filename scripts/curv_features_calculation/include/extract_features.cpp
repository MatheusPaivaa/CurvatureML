#include "extract_features.hpp"

void extractFeatures(const std::vector<Vertex>& vertex, const std::vector<Face>& faces, std::vector<double> starArea, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    // CSV header
    outFile << "X, Y, Z, MeanAdjX, MeanAdjY, MeanAdjZ, VertexDegree, starArea, NumAdjFaces, NormalX, NormalY, NormalZ, MeanNormalX, MeanNormalY, MeanNormalZ\n";

    std::unordered_map<int, std::vector<Vertex>> vertexToNormals;
    std::unordered_map<int, int> vertexToFaceCount;

    // Calculate normals for each face and map them to vertex
    for (const Face& face : faces) {
        Vertex normal = calculateNormal(vertex[face.v1], vertex[face.v2], vertex[face.v3]);
        normal = normalize(normal);
        vertexToNormals[face.v1].push_back(normal);
        vertexToNormals[face.v2].push_back(normal);
        vertexToNormals[face.v3].push_back(normal);
        vertexToFaceCount[face.v1]++;
        vertexToFaceCount[face.v2]++;
        vertexToFaceCount[face.v3]++;
    }

    std::unordered_map<int, std::unordered_set<int>> vertexTovertex;

    // Map each vertex to its adjacent vertex
    for (const Face& face : faces) {
        vertexTovertex[face.v1].insert(face.v2);
        vertexTovertex[face.v1].insert(face.v3);
        vertexTovertex[face.v2].insert(face.v1);
        vertexTovertex[face.v2].insert(face.v3);
        vertexTovertex[face.v3].insert(face.v1);
        vertexTovertex[face.v3].insert(face.v2);
    }

    // Process each vertex
    for (int i = 0; i < vertex.size(); i++) {
        const Vertex& v = vertex[i];
        double totalDistance = 0;
        std::vector<Vertex> adjvertex;

        // Calculate the mean position of adjacent vertex
        for (int adjIndex : vertexTovertex[i]) {
            adjvertex.push_back(vertex[adjIndex]);
            totalDistance += calculateDistance(v, vertex[adjIndex]);
        }

        Vertex meanAdj = std::accumulate(adjvertex.begin(), adjvertex.end(), Vertex{0, 0, 0},
                                         [](Vertex a, Vertex b) {
                                             return Vertex{a.x + b.x, a.y + b.y, a.z + b.z};
                                         });
        if (!adjvertex.empty()) {
            meanAdj.x /= adjvertex.size();
            meanAdj.y /= adjvertex.size();
            meanAdj.z /= adjvertex.size();
        }

        int vertexDegree = vertexTovertex[i].size();
        int numAdjFaces = vertexToFaceCount[i];

        // Calculate the normal vector of the vertex as the mean of the normals of adjacent faces
        Vertex normal = std::accumulate(vertexToNormals[i].begin(), vertexToNormals[i].end(), Vertex{0, 0, 0},
                                        [](Vertex a, Vertex b) {
                                            return Vertex{a.x + b.x, a.y + b.y, a.z + b.z};
                                        });
        if (!vertexToNormals[i].empty()) {
            normal.x /= vertexToNormals[i].size();
            normal.y /= vertexToNormals[i].size();
            normal.z /= vertexToNormals[i].size();
            normal = normalize(normal);
        }

        // Calculate the mean normal vectors of adjacent vertex
        std::vector<Vertex> adjNormals;
        for (int adjIndex : vertexTovertex[i]) {
            adjNormals.push_back(std::accumulate(vertexToNormals[adjIndex].begin(), vertexToNormals[adjIndex].end(), Vertex{0, 0, 0},
                                                 [](Vertex a, Vertex b) {
                                                     return Vertex{a.x + b.x, a.y + b.y, a.z + b.z};
                                                 }));
        }

        Vertex meanNormal = std::accumulate(adjNormals.begin(), adjNormals.end(), Vertex{0, 0, 0},
                                            [](Vertex a, Vertex b) {
                                                return Vertex{a.x + b.x, a.y + b.y, a.z + b.z};
                                            });
        if (!adjNormals.empty()) {
            meanNormal.x /= adjNormals.size();
            meanNormal.y /= adjNormals.size();
            meanNormal.z /= adjNormals.size();
            meanNormal = normalize(meanNormal);
        }

        // Write to CSV file
        outFile << v.x << ", " << v.y << ", " << v.z << ", ";
        outFile << meanAdj.x << ", " << meanAdj.y << ", " << meanAdj.z << ", ";
        outFile << vertexDegree << ", " << starArea[i] << ", ";
        outFile << numAdjFaces << ", ";
        outFile << normal.x << ", " << normal.y << ", " << normal.z << ", ";
        outFile << meanNormal.x << ", " << meanNormal.y << ", " << meanNormal.z << "\n";
    }

    outFile.close();
}
