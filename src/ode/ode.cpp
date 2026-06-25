#include "ms/ode/ode.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

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

OdeResult ode_midpoint(OdeFunc f, double t0, double y0, double t_end, size_t steps) {
    OdeResult result;
    if (steps == 0) return result;
    const double h = (t_end - t0) / static_cast<double>(steps);
    result.t.reserve(steps + 1);
    result.y.reserve(steps + 1);
    double t = t0, y = y0;
    for (size_t i = 0; i <= steps; ++i) {
        result.t.push_back(t);
        result.y.push_back(y);
        if (i < steps) {
            double k1 = f(t, y);
            double k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
            y += h * k2;
            t += h;
        }
    }
    return result;
}

OdeResult ode_rk2(OdeFunc f, double t0, double y0, double t_end, size_t steps) {
    // Heun's method
    OdeResult result;
    if (steps == 0) return result;
    const double h = (t_end - t0) / static_cast<double>(steps);
    result.t.reserve(steps + 1);
    result.y.reserve(steps + 1);
    double t = t0, y = y0;
    for (size_t i = 0; i <= steps; ++i) {
        result.t.push_back(t);
        result.y.push_back(y);
        if (i < steps) {
            double k1 = f(t, y);
            double k2 = f(t + h, y + h * k1);
            y += 0.5 * h * (k1 + k2);
            t += h;
        }
    }
    return result;
}

OdeResult ode_rk45(OdeFunc f, double t0, double y0, double t_end,
                   double rtol, double atol) {
    // Dormand-Prince RK45 adaptive step
    static const double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0,
                        c5 = 8.0/9.0;
    static const double a21 = 1.0/5.0;
    static const double a31 = 3.0/40.0,   a32 = 9.0/40.0;
    static const double a41 = 44.0/45.0,  a42 = -56.0/15.0,   a43 = 32.0/9.0;
    static const double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0,
                        a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    static const double a61 = 9017.0/3168.0,  a62 = -355.0/33.0,
                        a63 = 46732.0/5247.0, a64 = 49.0/176.0,
                        a65 = -5103.0/18656.0;
    // 5th-order weights
    static const double b1 = 35.0/384.0, b3 = 500.0/1113.0,
                        b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
    // Error coefficients (e = b - b*)
    static const double e1 = 71.0/57600.0,  e3 = -71.0/16695.0,
                        e4 = 71.0/1920.0,   e5 = -17253.0/339200.0,
                        e6 = 22.0/525.0,    e7 = -1.0/40.0;

    OdeResult result;
    double h = (t_end - t0) / 100.0;
    double t = t0, y = y0;
    result.t.push_back(t);
    result.y.push_back(y);

    const int max_steps = 50000;
    for (int step = 0; step < max_steps && t < t_end; ++step) {
        if (t + h > t_end) h = t_end - t;
        double k1 = f(t,              y);
        double k2 = f(t + c2*h,       y + h*a21*k1);
        double k3 = f(t + c3*h,       y + h*(a31*k1 + a32*k2));
        double k4 = f(t + c4*h,       y + h*(a41*k1 + a42*k2 + a43*k3));
        double k5 = f(t + c5*h,       y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4));
        double k6 = f(t + h,          y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5));
        double y5 = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6);
        double k7 = f(t + h, y5);
        double err = std::abs(h * (e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7));
        double sc = atol + rtol * std::max(std::abs(y), std::abs(y5));
        double err_norm = (sc > 0.0) ? err / sc : err;

        if (err_norm <= 1.0) {
            t += h;
            y  = y5;
            result.t.push_back(t);
            result.y.push_back(y);
        }
        double factor = (err_norm > 0.0) ?
            std::min(5.0, std::max(0.2, 0.9 * std::pow(err_norm, -0.2))) : 5.0;
        h *= factor;
        h = std::max(h, 1e-12 * std::abs(t_end - t0));
    }
    return result;
}

OdeResult ode_rk23(OdeFunc f, double t0, double y0, double t_end,
                   double rtol, double atol) {
    // Bogacki-Shampine RK23
    OdeResult result;
    double h = (t_end - t0) / 50.0;
    double t = t0, y = y0;
    result.t.push_back(t);
    result.y.push_back(y);
    double k1 = f(t, y);

    const int max_steps = 50000;
    for (int step = 0; step < max_steps && t < t_end; ++step) {
        if (t + h > t_end) h = t_end - t;
        double k2 = f(t + 0.5*h,    y + 0.5*h*k1);
        double k3 = f(t + 0.75*h,   y + 0.75*h*k2);
        double y3 = y + h*(2.0/9.0*k1 + 1.0/3.0*k2 + 4.0/9.0*k3);
        double k4 = f(t + h, y3);
        double err = std::abs(h * (5.0/72.0*k1 - 1.0/12.0*k2 - 1.0/9.0*k3 + 1.0/8.0*k4));
        double sc = atol + rtol * std::max(std::abs(y), std::abs(y3));
        double err_norm = (sc > 0.0) ? err / sc : err;

        if (err_norm <= 1.0) {
            t += h;
            y  = y3;
            k1 = k4;
            result.t.push_back(t);
            result.y.push_back(y);
        }
        double factor = (err_norm > 0.0) ?
            std::min(5.0, std::max(0.2, 0.9 * std::pow(err_norm, -1.0/3.0))) : 5.0;
        h *= factor;
    }
    return result;
}

OdeResultVec ode_euler_vec(OdeFuncVec f, double t0,
                            const std::vector<double>& y0,
                            double t_end, size_t steps) {
    OdeResultVec result;
    if (steps == 0) return result;
    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0;
    auto y = y0;
    result.t.push_back(t);
    result.y.push_back(y);
    for (size_t i = 0; i < steps; ++i) {
        auto dy = f(t, y);
        for (size_t j = 0; j < y.size(); ++j) y[j] += h * dy[j];
        t += h;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    return result;
}

OdeResultVec ode_rk4_vec(OdeFuncVec f, double t0,
                          const std::vector<double>& y0,
                          double t_end, size_t steps) {
    OdeResultVec result;
    if (steps == 0) return result;
    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0;
    auto y = y0;
    result.t.push_back(t);
    result.y.push_back(y);
    const auto vadd = [](const std::vector<double>& a, const std::vector<double>& b,
                         double s, const std::vector<double>& base) {
        std::vector<double> out(base.size());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = base[i] + s * a[i];
        (void)b;
        return out;
    };
    (void)vadd;
    for (size_t step = 0; step < steps; ++step) {
        auto k1 = f(t, y);
        std::vector<double> y2(y.size()), y3(y.size()), y4(y.size());
        for (size_t j = 0; j < y.size(); ++j) y2[j] = y[j] + 0.5*h*k1[j];
        auto k2 = f(t + 0.5*h, y2);
        for (size_t j = 0; j < y.size(); ++j) y3[j] = y[j] + 0.5*h*k2[j];
        auto k3 = f(t + 0.5*h, y3);
        for (size_t j = 0; j < y.size(); ++j) y4[j] = y[j] + h*k3[j];
        auto k4 = f(t + h, y4);
        for (size_t j = 0; j < y.size(); ++j)
            y[j] += (h/6.0)*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
        t += h;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    return result;
}

OdeResult ode_adams_bashforth2(OdeFunc f, double t0, double y0,
                                double t_end, size_t steps) {
    OdeResult result;
    if (steps == 0) return result;
    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0, y = y0;
    result.t.push_back(t);
    result.y.push_back(y);
    // bootstrap with one Euler step
    double f_prev = f(t, y);
    if (steps >= 1) {
        y += h * f_prev;
        t += h;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    double f_curr = f(t, y);
    for (size_t i = 2; i <= steps; ++i) {
        double y_new = y + h * (1.5 * f_curr - 0.5 * f_prev);
        t += h;
        f_prev = f_curr;
        y = y_new;
        f_curr = f(t, y);
        result.t.push_back(t);
        result.y.push_back(y);
    }
    return result;
}

} // namespace ms
