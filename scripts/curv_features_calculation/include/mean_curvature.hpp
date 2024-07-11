#ifndef MEAN_CURVATURE_HPP
#define MEAN_CURVATURE_HPP

#include "read_obj.hpp"
#include "general_operations.hpp"

std::vector<double> calcMean(const std::vector<Vertex>& V, const std::vector<Face>& F, std::vector<double> areaStar);

#endif // MEAN_CURVATURE_HPP
