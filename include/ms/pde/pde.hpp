#pragma once

#include <vector>

namespace ms {

struct Heat1DResult {
    std::vector<double> x;
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

Heat1DResult pde_heat_1d(
    const std::vector<double>& x0,
    double alpha,
    double dx,
    double dt,
    size_t steps);

} // namespace ms
