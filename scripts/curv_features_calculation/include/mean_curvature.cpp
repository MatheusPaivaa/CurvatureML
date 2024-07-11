#include "mean_curvature.hpp"

// Function to calculate the dihedral angle and its sum for a given vertex
double calculatePhi(const std::vector<Vertex>& V, int i, int j, int l, int k) {
    Vertex eij = V[j] - V[i];
    Vertex eij2 = normalize(eij);

    Vertex nijk = vecnorm(V[i], V[j], V[l]);
    Vertex njil = vecnorm(V[j], V[i], V[k]);

    double phi = atan2(dot(eij2, cross(nijk, njil)), dot(nijk, njil)) * norm(eij);

    return phi;
}

// Main function to calculate the mean curvature at all given vertex
std::vector<double> calcMean(const std::vector<Vertex>& V, const std::vector<Face>& F, std::vector<double> areaStar) {
    std::vector<double> meanCurvature(V.size(), 0.0);
    std::vector<double> dihedralAngles(V.size(), 0.0);

    std::unordered_map<Edge, std::vector<int>, EdgeHash> edgeMap;
    mapAdjacentEdges(F, edgeMap);

    // Loop through all edges and calculate dihedral angles
    for (const auto& pair : edgeMap) {
        const Edge& edge = pair.first;
        const std::vector<int>& sharingFaces = pair.second;

        if (sharingFaces.size() < 2) continue; // Need at least two faces to calculate the dihedral angle

        const Face& face1 = F[sharingFaces[0]];
        const Face& face2 = F[sharingFaces[1]];

        std::unordered_set<int> vertexFace1 = {face1.v1, face1.v2, face1.v3};
        std::unordered_set<int> vertexFace2 = {face2.v1, face2.v2, face2.v3};

        vertexFace1.erase(edge.v1);
        vertexFace1.erase(edge.v2);
        vertexFace2.erase(edge.v1);
        vertexFace2.erase(edge.v2);

        if (vertexFace1.empty() || vertexFace2.empty()) continue;

        int l = *vertexFace1.begin(); // Remaining vertex in the first face
        int k = *vertexFace2.begin(); // Remaining vertex in the second face

        bool sameOrientation = (F[sharingFaces[0]].v1 == edge.v1 && F[sharingFaces[0]].v2 == edge.v2) || 
                               (F[sharingFaces[0]].v2 == edge.v1 && F[sharingFaces[0]].v3 == edge.v2) || 
                               (F[sharingFaces[0]].v3 == edge.v1 && F[sharingFaces[0]].v1 == edge.v2);

        if (sameOrientation) {
            std::swap(l, k);
        }

        double dihedralAngle = calculatePhi(V, edge.v1, edge.v2, l, k);

        dihedralAngles[edge.v2] += dihedralAngle;
        dihedralAngles[edge.v1] += dihedralAngle;
    }

    // Calculate mean curvature for each vertex
    for (size_t i = 0; i < V.size(); ++i) {
        if (areaStar[i] != 0) {
            meanCurvature[i] = (0.25 * dihedralAngles[i]) / areaStar[i];
        }
    }

    return meanCurvature;
}
