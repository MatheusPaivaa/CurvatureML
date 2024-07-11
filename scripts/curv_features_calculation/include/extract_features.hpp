#ifndef EXTRACT_FEATURES_HPP
#define EXTRACT_FEATURES_HPP

#include "general_operations.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

void extractFeatures(const std::vector<Vertex>& vertex, const std::vector<Face>& faces, std::vector<double> starArea, const std::string& filename);

#endif // EXTRACT_FEATURES_HPP
