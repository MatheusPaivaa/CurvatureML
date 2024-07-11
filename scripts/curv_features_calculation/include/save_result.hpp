#ifndef SAVE_RESULT_HPP
#define SAVE_RESULT_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

// Declaração da função que salva curvas em um arquivo CSV
void save_result(const std::vector<double>& gaussianCurvature, const std::vector<double>& meanCurvature, const std::string& filename);

#endif // SAVE_RESULT_HPP
