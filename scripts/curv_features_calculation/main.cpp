#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include "include/read_obj.hpp"
#include "include/extract_features.hpp"
#include "include/gaussian_curvature.hpp"
#include "include/mean_curvature.hpp"
#include "include/general_operations.hpp"
#include "include/save_result.hpp"

int main() {
    namespace fs = std::filesystem;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string input_dir = "../../data/face_processed";
    std::string output_dir_pouro = "../../data/in_out_processed/output";
    std::string output_dir_vcarac = "../../data/in_out_processed/input";

    // Check and create output directory if necessary
    if (!fs::exists(output_dir_pouro)) {
        fs::create_directories(output_dir_pouro);
    }

    if (!fs::exists(output_dir_vcarac)) {
        fs::create_directories(output_dir_vcarac);
    }

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".obj") {
            std::vector<Vertex> vertices;
            std::vector<Face> faces;

            std::string input_path = entry.path().string();
            read_obj(input_path, vertices, faces);

            std::vector<double> Amixed(vertices.size(), 0.0);
            Amixed = calculateMixedArea(vertices, faces);

            std::vector<double> curvaGaussiana = calcGaussian(vertices, faces, Amixed);
            std::vector<double> curvaCurvMedia = calcMean(vertices, faces, Amixed);

            // Construct the output file name based on the input file name
            std::string base_name = entry.path().stem().string(); // "face_1b"
            std::string output_filename_pouro = output_dir_pouro + "/output_" + base_name + ".csv";
            std::string output_filename_carac = output_dir_vcarac + "/input_" + base_name + ".csv";

            save_result(curvaGaussiana, curvaCurvMedia, output_filename_pouro);
            extractFeatures(vertices, faces, Amixed, output_filename_carac);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execution_time = end - start;
    std::cout << "\nExecution time: " << execution_time.count() << " ms\n" << std::endl;

    return 0;
}
