#ifndef GAUSSIAN_CURVATURE_HPP
#define GAUSSIAN_CURVATURE_HPP

#include "read_obj.hpp"
#include "general_operations.hpp"

std::vector<double> calcGaussian(const std::vector<Vertex>& V, const std::vector<Face>& F, std::vector<double> areaStar);

#endif // GAUSSIAN_CURVATURE_HPP
