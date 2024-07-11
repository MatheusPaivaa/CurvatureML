#include "gaussian_curvature.hpp"

std::vector<double> calcGaussian(const std::vector<Vertex>& V, const std::vector<Face>& F, std::vector<double> areaStar) {
    // Initialize vectors to store the sum of angles and Gaussian curvature
    std::vector<double> anglesSum(V.size(), 0.0);
    std::vector<double> gaussianCurvature(V.size(), 0.0);
    std::unordered_map<int, std::set<int>> mapping;

    // Map vertices to faces
    mapping = mapVerticesToFaces(V, F);

    // Calculate the sum of angles for each vertex
    anglesSum = calculateAngleSum(V, F, mapping);

    // Calculate Gaussian curvature for each vertex
    for (size_t i = 0; i < V.size(); ++i) {
        if (areaStar[i] != 0) {
            gaussianCurvature[i] = (2 * M_PI - anglesSum[i]) / areaStar[i];
        }

        // Uncomment to print Gaussian curvature at each vertex
        // std::cout << "Gaussian Curvature at vertex " << i << ": " << curvaGaussiana[i] << std::endl;
    }

    return gaussianCurvature;
}
