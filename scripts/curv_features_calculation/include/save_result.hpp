#ifndef SAVE_RESULT_HPP
#define SAVE_RESULT_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

/**
 * @brief Saves the Gaussian and Mean Curvature data to a file.
 * 
 * This function takes two vectors containing Gaussian and Mean Curvature values
 * and writes them to a specified file in CSV format. Each line of the file will
 * contain a Gaussian Curvature value and a Mean Curvature value, separated by a comma.
 *
 * @param gaussianCurvature A vector containing the Gaussian Curvature values.
 * @param meanCurvature A vector containing the Mean Curvature values.
 * @param filename The name of the file to write the data to.
 * 
**/

void save_result(const std::vector<double>& gaussianCurvature, const std::vector<double>& meanCurvature, const std::string& filename);

#endif // SAVE_RESULT_HPP
