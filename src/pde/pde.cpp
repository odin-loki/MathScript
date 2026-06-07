#include "ms/pde/pde.hpp"

namespace ms {

Heat1DResult pde_heat_1d(
    const std::vector<double>& x0,
    double alpha,
    double dx,
    double dt,
    size_t steps) {
    Heat1DResult result;
    if (x0.size() < 3 || steps == 0) {
        return result;
    }

    const size_t n = x0.size();
    const double r = alpha * dt / (dx * dx);
    if (r > 0.5) {
        return result;
    }

    result.x.resize(n);
    for (size_t i = 0; i < n; ++i) {
        result.x[i] = static_cast<double>(i) * dx;
    }

    std::vector<double> u = x0;
    std::vector<double> next(n);
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (size_t step = 1; step <= steps; ++step) {
        for (size_t i = 1; i + 1 < n; ++i) {
            next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        }
        next[0] = u[0];
        next[n - 1] = u[n - 1];
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

} // namespace ms
