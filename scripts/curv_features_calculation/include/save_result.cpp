#include "save_result.hpp"

void save_result(const std::vector<double>& gaussianCurvature, const std::vector<double>& meanCurvature, const std::string& filename) {
    std::ofstream file(filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening the file for writing.\n";
        return;
    }

    // Write the header to the file
    file << "GaussianCurvature,MeanCurvature\n";

    // Set the precision for the output stream
    file << std::fixed << std::setprecision(14); // Adjust precision as needed

    // Write the curvature data to the file
    for (size_t i = 0; i < gaussianCurvature.size() && i < meanCurvature.size(); ++i) {
        file << gaussianCurvature[i] << ',' << meanCurvature[i] << '\n';
    }

    file.close(); // Close the file
}
