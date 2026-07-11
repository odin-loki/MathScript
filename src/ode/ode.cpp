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

namespace {

constexpr double kNewtonTol = 1e-10;
constexpr int kNewtonMaxIter = 50;
constexpr double kNewtonEps = 1e-8;
constexpr double kPicardTol = 1e-10;
constexpr int kPicardMaxIter = 50;

double vec_max_norm(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

std::vector<double> gauss_solve(std::vector<std::vector<double>> A,
                                 std::vector<double> b) {
    const size_t n = b.size();
    if (A.size() != n) {
        return {};
    }
    for (size_t i = 0; i < n; ++i) {
        if (A[i].size() != n) {
            return {};
        }
    }

    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        double max_abs = std::abs(A[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            const double v = std::abs(A[row][col]);
            if (v > max_abs) {
                max_abs = v;
                pivot = row;
            }
        }
        if (max_abs < 1e-14) {
            return {};
        }
        if (pivot != col) {
            std::swap(A[pivot], A[col]);
            std::swap(b[pivot], b[col]);
        }
        const double diag = A[col][col];
        for (size_t row = col + 1; row < n; ++row) {
            const double factor = A[row][col] / diag;
            for (size_t j = col; j < n; ++j) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    std::vector<double> x(n, 0.0);
    for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
        double sum = b[static_cast<size_t>(row)];
        for (size_t j = static_cast<size_t>(row) + 1; j < n; ++j) {
            sum -= A[static_cast<size_t>(row)][j] * x[j];
        }
        const double diag = A[static_cast<size_t>(row)][static_cast<size_t>(row)];
        if (std::abs(diag) < 1e-14) {
            return {};
        }
        x[static_cast<size_t>(row)] = sum / diag;
    }
    return x;
}
constexpr int kShootMaxIter = 100;
constexpr double kShootTol = 1e-8;
constexpr double kEventTol = 1e-9;
constexpr int kEventMaxBisect = 60;

double central_df(OdeFunc f, double t, double y) {
    const double fp = f(t, y + kNewtonEps);
    const double fm = f(t, y - kNewtonEps);
    return (fp - fm) / (2.0 * kNewtonEps);
}

double interpolate_linear(const std::vector<double>& xs,
                          const std::vector<double>& ys, double x) {
    if (xs.empty()) {
        return 0.0;
    }
    if (x <= xs.front()) {
        return ys.front();
    }
    if (x >= xs.back()) {
        return ys.back();
    }
    const auto it = std::upper_bound(xs.begin(), xs.end(), x);
    const size_t i = static_cast<size_t>(it - xs.begin());
    const double x0 = xs[i - 1];
    const double x1 = xs[i];
    const double y0 = ys[i - 1];
    const double y1 = ys[i];
    const double w = (x - x0) / (x1 - x0);
    return y0 + w * (y1 - y0);
}

double delayed_value(double td, double t0,
                     const std::function<double(double)>& history,
                     const std::vector<double>& t_grid,
                     const std::vector<double>& y_grid) {
    if (td <= t0) {
        return history(td);
    }
    return interpolate_linear(t_grid, y_grid, td);
}

} // namespace

OdeResult ode_backward_euler(OdeFunc f, double t0, double y0,
                              double t_end, size_t steps) {
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
            const double t_next = t + h;
            double y_next = y;
            for (int iter = 0; iter < kNewtonMaxIter; ++iter) {
                const double fy = f(t_next, y_next);
                const double g = y_next - y - h * fy;
                if (std::abs(g) < kNewtonTol) {
                    break;
                }
                const double dfdy = central_df(f, t_next, y_next);
                const double dg = 1.0 - h * dfdy;
                if (std::abs(dg) < 1e-14) {
                    break;
                }
                y_next -= g / dg;
            }
            y = y_next;
            t = t_next;
        }
    }
    return result;
}

OdeBvpResult ode_bvp_shooting(OdeBvpFunc f, double t0, double y_a,
                               double t_end, double y_b, size_t steps) {
    OdeBvpResult result;
    if (steps == 0) {
        return result;
    }

    const auto integrate = [&](double slope) -> double {
        OdeFuncVec sys = [&](double t, const std::vector<double>& state) {
            std::vector<double> d(2);
            d[0] = state[1];
            d[1] = f(t, state[0], state[1]);
            return d;
        };
        const std::vector<double> y0 = {y_a, slope};
        const OdeResultVec ivp = ode_rk4_vec(sys, t0, y0, t_end, steps);
        if (ivp.y.empty()) {
            return slope;
        }
        return ivp.y.back()[0];
    };

    double s_lo = -10.0;
    double s_hi = 10.0;
    double r_lo = integrate(s_lo) - y_b;
    double r_hi = integrate(s_hi) - y_b;

    int expand = 0;
    while (r_lo * r_hi > 0.0 && expand < 20) {
        s_lo *= 2.0;
        s_hi *= 2.0;
        r_lo = integrate(s_lo) - y_b;
        r_hi = integrate(s_hi) - y_b;
        ++expand;
    }

    if (r_lo * r_hi > 0.0) {
        return result;
    }

    double s_mid = 0.5 * (s_lo + s_hi);
    size_t iter = 0;
    for (; iter < static_cast<size_t>(kShootMaxIter); ++iter) {
        s_mid = 0.5 * (s_lo + s_hi);
        const double r_mid = integrate(s_mid) - y_b;
        if (std::abs(r_mid) < kShootTol) {
            result.converged = true;
            break;
        }
        if (r_lo * r_mid <= 0.0) {
            s_hi = s_mid;
            r_hi = r_mid;
        } else {
            s_lo = s_mid;
            r_lo = r_mid;
        }
    }

    if (!result.converged) {
        const double r_mid = integrate(s_mid) - y_b;
        if (std::abs(r_mid) < kShootTol) {
            result.converged = true;
        }
    }
    result.iterations = iter + (result.converged ? 1u : 0u);

    OdeFuncVec sys = [&](double t, const std::vector<double>& state) {
        std::vector<double> d(2);
        d[0] = state[1];
        d[1] = f(t, state[0], state[1]);
        return d;
    };
    const std::vector<double> y0 = {y_a, s_mid};
    const OdeResultVec ivp = ode_rk4_vec(sys, t0, y0, t_end, steps);
    result.t = ivp.t;
    result.y.reserve(ivp.y.size());
    result.yp.reserve(ivp.y.size());
    for (const auto& state : ivp.y) {
        result.y.push_back(state[0]);
        result.yp.push_back(state[1]);
    }
    if (result.converged && !result.y.empty()) {
        const double err = std::abs(result.y.back() - y_b);
        if (err > 10.0 * kShootTol) {
            result.converged = false;
        }
    }
    return result;
}

OdeResult ode_dde_fixed_step(std::function<double(double, double, double)> f,
                              std::function<double(double)> history,
                              double t0, double t_end, double tau,
                              size_t steps) {
    OdeResult result;
    if (steps == 0 || tau <= 0.0) {
        return result;
    }
    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0;
    double y = history(t0);
    result.t.push_back(t);
    result.y.push_back(y);

    for (size_t step = 0; step < steps; ++step) {
        const auto delayed_at = [&](double ts) {
            return delayed_value(ts - tau, t0, history, result.t, result.y);
        };

        const double k1 = f(t, y, delayed_at(t));
        const double y2 = y + 0.5 * h * k1;
        const double k2 = f(t + 0.5 * h, y2, delayed_at(t + 0.5 * h));
        const double y3 = y + 0.5 * h * k2;
        const double k3 = f(t + 0.5 * h, y3, delayed_at(t + 0.5 * h));
        const double y4 = y + h * k3;
        const double k4 = f(t + h, y4, delayed_at(t + h));
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    return result;
}

OdeEventResult ode_event_detect(OdeFunc f, OdeFunc event_g,
                                 double t0, double y0, double t_end,
                                 size_t steps) {
    OdeEventResult result;
    if (steps == 0) {
        return result;
    }
    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0;
    double y = y0;
    double g_prev = event_g(t, y);
    result.t.push_back(t);
    result.y.push_back(y);

    for (size_t step = 0; step < steps; ++step) {
        const double k1 = f(t, y);
        const double k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
        const double k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
        const double k4 = f(t + h, y + h * k3);
        const double y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        const double t_next = t + h;
        const double g_next = event_g(t_next, y_next);

        if (g_prev * g_next < 0.0) {
            double lo_t = t;
            double hi_t = t_next;
            double lo_g = g_prev;
            double hi_g = g_next;
            double lo_y = y;
            double hi_y = y_next;
            double event_t = hi_t;
            double event_y = hi_y;

            for (int bisect = 0; bisect < kEventMaxBisect; ++bisect) {
                const double mid_t = 0.5 * (lo_t + hi_t);
                const double frac = (hi_t > lo_t) ? (mid_t - lo_t) / (hi_t - lo_t) : 0.0;
                const double mid_y = lo_y + frac * (hi_y - lo_y);
                const double mid_g = event_g(mid_t, mid_y);
                event_t = mid_t;
                event_y = mid_y;
                if (std::abs(mid_g) < kEventTol || (hi_t - lo_t) < kEventTol) {
                    break;
                }
                if (lo_g * mid_g <= 0.0) {
                    hi_t = mid_t;
                    hi_y = mid_y;
                    hi_g = mid_g;
                } else {
                    lo_t = mid_t;
                    lo_y = mid_y;
                    lo_g = mid_g;
                }
            }
            result.event_times.push_back(event_t);
            result.event_values.push_back(event_y);
        }

        y = y_next;
        t = t_next;
        g_prev = g_next;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    return result;
}

OdeResultVec ode_backward_euler_vec(OdeFuncVec f, double t0,
                                     const std::vector<double>& y0,
                                     double t_end, size_t steps) {
    OdeResultVec result;
    if (steps == 0 || y0.empty()) {
        return result;
    }
    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0;
    auto y = y0;
    result.t.push_back(t);
    result.y.push_back(y);

    for (size_t step = 0; step < steps; ++step) {
        const double t_next = t + h;
        auto y_next = y;
        for (int iter = 0; iter < kPicardMaxIter; ++iter) {
            const auto fy = f(t_next, y_next);
            if (fy.size() != y.size()) {
                break;
            }
            std::vector<double> y_new(y.size());
            for (size_t j = 0; j < y.size(); ++j) {
                y_new[j] = y[j] + h * fy[j];
            }
            const double change = vec_max_norm(y_new, y_next);
            y_next = std::move(y_new);
            if (!std::isfinite(change)) {
                break;
            }
            if (change < kPicardTol) {
                break;
            }
        }
        y = std::move(y_next);
        t = t_next;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    return result;
}

// Index-1 DAE: semi-explicit form dy/dt = f(t,y,z), 0 = g(t,y,z).
// Index-1 means the algebraic Jacobian dg/dz is nonsingular so z can be
// locally determined from y. This solver is a deliberate simplification:
// explicit RK4 predictor for y using the current z, then Newton correction
// for z at each step. Adequate for well-conditioned index-1 systems; not a
// substitute for a fully implicit DAE integrator.
DaeResult ode_dae_index1(DaeDiffFunc f, DaeAlgFunc g, double t0,
                          const std::vector<double>& y0,
                          const std::vector<double>& z0,
                          double t_end, size_t steps) {
    DaeResult result;
    if (steps == 0 || y0.empty() || z0.empty()) {
        return result;
    }

    const double h = (t_end - t0) / static_cast<double>(steps);
    double t = t0;
    auto y = y0;
    auto z = z0;
    result.converged = true;

    const auto solve_algebraic = [&](double t_new, const std::vector<double>& y_new,
                                     std::vector<double>& z_state) -> bool {
        const size_t n = z_state.size();
        for (int iter = 0; iter < kNewtonMaxIter; ++iter) {
            const auto gv = g(t_new, y_new, z_state);
            if (gv.size() != n) {
                return false;
            }
            double max_res = 0.0;
            for (double v : gv) {
                max_res = std::max(max_res, std::abs(v));
            }
            if (max_res < kNewtonTol) {
                return true;
            }

            std::vector<std::vector<double>> J(n, std::vector<double>(n, 0.0));
            for (size_t j = 0; j < n; ++j) {
                auto zp = z_state;
                auto zm = z_state;
                zp[j] += kNewtonEps;
                zm[j] -= kNewtonEps;
                const auto gp = g(t_new, y_new, zp);
                const auto gm = g(t_new, y_new, zm);
                if (gp.size() != n || gm.size() != n) {
                    return false;
                }
                for (size_t i = 0; i < n; ++i) {
                    J[i][j] = (gp[i] - gm[i]) / (2.0 * kNewtonEps);
                }
            }

            std::vector<double> rhs(n);
            for (size_t i = 0; i < n; ++i) {
                rhs[i] = -gv[i];
            }
            const auto dz = gauss_solve(J, rhs);
            if (dz.size() != n) {
                return false;
            }
            for (size_t j = 0; j < n; ++j) {
                z_state[j] += dz[j];
            }
        }
        const auto gv = g(t_new, y_new, z_state);
        if (gv.size() != n) {
            return false;
        }
        for (double v : gv) {
            if (std::abs(v) >= kNewtonTol) {
                return false;
            }
        }
        return true;
    };

    result.t.push_back(t);
    result.y.push_back(y);
    result.z.push_back(z);

    for (size_t step = 0; step < steps; ++step) {
        const auto k1 = f(t, y, z);
        if (k1.size() != y.size()) {
            result.converged = false;
            break;
        }

        std::vector<double> y2(y.size());
        for (size_t j = 0; j < y.size(); ++j) {
            y2[j] = y[j] + 0.5 * h * k1[j];
        }
        const auto k2 = f(t + 0.5 * h, y2, z);
        if (k2.size() != y.size()) {
            result.converged = false;
            break;
        }

        std::vector<double> y3(y.size());
        for (size_t j = 0; j < y.size(); ++j) {
            y3[j] = y[j] + 0.5 * h * k2[j];
        }
        const auto k3 = f(t + 0.5 * h, y3, z);
        if (k3.size() != y.size()) {
            result.converged = false;
            break;
        }

        std::vector<double> y4(y.size());
        for (size_t j = 0; j < y.size(); ++j) {
            y4[j] = y[j] + h * k3[j];
        }
        const auto k4 = f(t + h, y4, z);
        if (k4.size() != y.size()) {
            result.converged = false;
            break;
        }

        std::vector<double> y_new(y.size());
        for (size_t j = 0; j < y.size(); ++j) {
            y_new[j] = y[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }

        const double t_next = t + h;
        if (!solve_algebraic(t_next, y_new, z)) {
            result.converged = false;
        }

        y = std::move(y_new);
        t = t_next;
        result.t.push_back(t);
        result.y.push_back(y);
        result.z.push_back(z);
    }

    return result;
}

} // namespace ms
