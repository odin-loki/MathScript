#include "ms/ode/ode.hpp"

namespace ms {

OdeResult ode_euler(OdeFunc f, double t0, double y0, double t_end, size_t steps) {
    OdeResult result;
    if (steps == 0) {
        return result;
    }
    const double h = (t_end - t0) / static_cast<double>(steps);
    result.t.reserve(steps + 1);
    result.y.reserve(steps + 1);
    double t = t0;
    double y = y0;
    for (size_t i = 0; i <= steps; ++i) {
        result.t.push_back(t);
        result.y.push_back(y);
        if (i < steps) {
            y += h * f(t, y);
            t += h;
        }
    }
    return result;
}

OdeResult ode_rk4(OdeFunc f, double t0, double y0, double t_end, size_t steps) {
    OdeResult result;
    if (steps == 0) {
        return result;
    }
    const double h = (t_end - t0) / static_cast<double>(steps);
    result.t.reserve(steps + 1);
    result.y.reserve(steps + 1);
    double t = t0;
    double y = y0;
    for (size_t i = 0; i <= steps; ++i) {
        result.t.push_back(t);
        result.y.push_back(y);
        if (i < steps) {
            const double k1 = f(t, y);
            const double k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
            const double k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
            const double k4 = f(t + h, y + h * k3);
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            t += h;
        }
    }
    return result;
}

} // namespace ms
