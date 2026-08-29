// Wave 56: Global optimization and new scalar root finder tests
#define _USE_MATH_DEFINES
#include "ms/optim/optim.hpp"
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Rosenbrock function (classic test)
// -----------------------------------------------------------------------
static double rosenbrock(const std::vector<double>& x) {
    return 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) +
           (1.0 - x[0]) * (1.0 - x[0]);
}

// Simple sphere
static double sphere(const std::vector<double>& x) {
    double s = 0.0;
    for (auto v : x) s += v * v;
    return s;
}

// -----------------------------------------------------------------------
// Nelder-Mead
// -----------------------------------------------------------------------
TEST(OptimNelderMead, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::nelder_mead(sphere, {2.0, 3.0});
    EXPECT_LT(r.f_val, 0.01);
}

TEST(OptimNelderMead, Rosenbrock_FvalNearZero) {
    ms::OptimResult r = ms::nelder_mead(rosenbrock, {0.0, 0.0}, 1e-8, 5000);
    EXPECT_LT(r.f_val, 1.0);
}

TEST(OptimNelderMead, ResultIsFinite) {
    ms::OptimResult r = ms::nelder_mead(sphere, {1.0, 1.0, 1.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    for (auto v : r.x) EXPECT_TRUE(std::isfinite(v));
}

TEST(OptimNelderMead, CorrectDimension) {
    ms::OptimResult r = ms::nelder_mead(sphere, {1.0, 2.0, 3.0});
    EXPECT_EQ(r.x.size(), 3u);
}

// -----------------------------------------------------------------------
// BFGS
// -----------------------------------------------------------------------
TEST(OptimBFGS, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::bfgs(sphere, {2.0, 3.0});
    EXPECT_LT(r.f_val, 1e-6);
}

TEST(OptimBFGS, Sphere5D_Converges) {
    ms::OptimResult r = ms::bfgs(sphere, {1.0, 1.0, 1.0, 1.0, 1.0});
    EXPECT_LT(r.f_val, 1e-4);
}

TEST(OptimBFGS, ResultIsFinite) {
    ms::OptimResult r = ms::bfgs(sphere, {3.0, -2.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
}

// -----------------------------------------------------------------------
// L-BFGS
// -----------------------------------------------------------------------
TEST(OptimLBFGS, Sphere2D_Converges) {
    ms::OptimResult r = ms::lbfgs(sphere, {5.0, 5.0});
    EXPECT_LT(r.f_val, 1e-4);
}

TEST(OptimLBFGS, ResultIsFinite) {
    ms::OptimResult r = ms::lbfgs(sphere, {1.0, 2.0, 3.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    EXPECT_EQ(r.x.size(), 3u);
}

// -----------------------------------------------------------------------
// Conjugate Gradient (Polak-Ribière+)
// -----------------------------------------------------------------------
namespace {

ms::GradND sphere_grad = [](const std::vector<double>& x) {
    return x;
};

ms::GradND rosenbrock_grad = [](const std::vector<double>& x) {
    const double dx =
        -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    const double dy = 200.0 * (x[1] - x[0] * x[0]);
    return std::vector<double>{dx, dy};
};

ms::GradND make_quadratic_grad(const std::vector<double>& diag) {
    return [diag](const std::vector<double>& x) {
        std::vector<double> g(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            g[i] = diag[i] * x[i];
        }
        return g;
    };
}

} // namespace

TEST(OptimConjugateGradient, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::conjugate_gradient(sphere, sphere_grad, {2.0, 3.0});
    EXPECT_LT(r.f_val, 1e-6);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.x[0], 0.0, 1e-4);
    EXPECT_NEAR(r.x[1], 0.0, 1e-4);
}

TEST(OptimConjugateGradient, Rosenbrock_ConvergesToKnownMinimum) {
    ms::OptimResult r = ms::conjugate_gradient(
        rosenbrock, rosenbrock_grad, {-1.0, 1.0}, 1e-6, 2000);
    EXPECT_LT(r.f_val, 1e-3);
    EXPECT_NEAR(r.x[0], 1.0, 1e-2);
    EXPECT_NEAR(r.x[1], 1.0, 1e-2);
}

TEST(OptimConjugateGradient, HighDimQuadratic_ConvergesWithinNSteps) {
    const int n = 10;
    std::vector<double> x0(static_cast<size_t>(n), 1.0);
    ms::OptimResult r = ms::conjugate_gradient(
        sphere, sphere_grad, x0, 1e-8, n + 20);
    EXPECT_LT(r.f_val, 1e-6);
    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.iterations, static_cast<size_t>(n + 15));
}

TEST(OptimConjugateGradient, AgreesWithBFGS_OnSphere) {
    ms::OptimResult cg = ms::conjugate_gradient(sphere, sphere_grad, {3.0, -2.0});
    ms::OptimResult bf = ms::bfgs(sphere, {3.0, -2.0});
    EXPECT_LT(cg.f_val, 1e-6);
    EXPECT_LT(bf.f_val, 1e-6);
    EXPECT_NEAR(cg.x[0], bf.x[0], 1e-3);
    EXPECT_NEAR(cg.x[1], bf.x[1], 1e-3);
}

TEST(OptimConjugateGradient, AlreadyAtMinimum_TerminatesImmediately) {
    ms::OptimResult r = ms::conjugate_gradient(sphere, sphere_grad, {0.0, 0.0});
    EXPECT_LT(r.f_val, 1e-12);
    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.iterations, 2u);
}

TEST(OptimConjugateGradient, IllConditionedQuadratic_StillConverges) {
    const std::vector<double> diag = {1.0, 100.0, 10000.0};
    ms::FuncND f = [&diag](const std::vector<double>& x) {
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            s += 0.5 * diag[i] * x[i] * x[i];
        }
        return s;
    };
    ms::GradND g = make_quadratic_grad(diag);
    ms::OptimResult r = ms::conjugate_gradient(
        f, g, {10.0, -5.0, 2.0}, 1e-6, 5000);
    EXPECT_LT(r.f_val, 1e-4);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.x[0], 0.0, 1e-3);
    EXPECT_NEAR(r.x[1], 0.0, 1e-3);
    EXPECT_NEAR(r.x[2], 0.0, 1e-3);
}

// -----------------------------------------------------------------------
// Adam
// -----------------------------------------------------------------------
TEST(OptimAdam, Sphere2D_ReducesObjective) {
    ms::OptimResult r = ms::adam(sphere, {2.0, 2.0});
    EXPECT_LT(r.f_val, 4.0);  // Should reduce from initial 8.0
}

TEST(OptimAdam, ResultIsFinite) {
    ms::OptimResult r = ms::adam(sphere, {1.0, 1.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
}

// -----------------------------------------------------------------------
// RMSprop (adaptive gradient)
// -----------------------------------------------------------------------
TEST(OptimRMSprop, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, {2.0, 3.0}, 0.01, 0.9, 1e-8, 3000);
    EXPECT_LT(r.f_val, 1e-4);
    EXPECT_NEAR(r.x[0], 0.0, 1e-2);
    EXPECT_NEAR(r.x[1], 0.0, 1e-2);
}

TEST(OptimRMSprop, Rosenbrock_ConvergesToKnownMinimum) {
    ms::OptimResult r = ms::rmsprop(
        rosenbrock, rosenbrock_grad, {-1.0, 1.0}, 0.01, 0.9, 1e-8, 8000);
    EXPECT_LT(r.f_val, 0.03);
    EXPECT_NEAR(r.x[0], 1.0, 5e-2);
    EXPECT_NEAR(r.x[1], 1.0, 5e-2);
}

TEST(OptimRMSprop, QuadraticMinimum_Found) {
    const std::vector<double> diag = {2.0, 4.0, 8.0};
    ms::FuncND f = [&diag](const std::vector<double>& x) {
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            s += 0.5 * diag[i] * x[i] * x[i];
        }
        return s;
    };
    ms::GradND g = make_quadratic_grad(diag);
    ms::OptimResult r = ms::rmsprop(f, g, {5.0, -3.0, 2.0}, 0.1, 0.9, 1e-8, 8000);
    EXPECT_LT(r.f_val, 0.02);
    EXPECT_NEAR(r.x[0], 0.0, 5e-2);
    EXPECT_NEAR(r.x[1], 0.0, 5e-2);
    EXPECT_NEAR(r.x[2], 0.0, 5e-2);
}

TEST(OptimRMSprop, ResultIsFinite) {
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, {3.0, -2.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    for (auto v : r.x) EXPECT_TRUE(std::isfinite(v));
}

TEST(OptimRMSprop, CorrectDimension) {
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, {1.0, 2.0, 3.0});
    EXPECT_EQ(r.x.size(), 3u);
}

TEST(OptimRMSprop, AlreadyAtMinimum_TerminatesImmediately) {
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, {0.0, 0.0});
    EXPECT_LT(r.f_val, 1e-12);
    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.iterations, 2u);
}

TEST(OptimRMSprop, Sphere2D_ReducesObjective) {
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, {2.0, 2.0});
    EXPECT_LT(r.f_val, 4.0);
}

TEST(OptimRMSprop, IterationsWithinMaxIter) {
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, {1.0, 1.0}, 0.001, 0.9, 1e-8, 50);
    EXPECT_LE(r.iterations, 50u);
}

TEST(OptimRMSprop, HighDimQuadratic_Converges) {
    const int n = 8;
    std::vector<double> x0(static_cast<size_t>(n), 1.0);
    ms::OptimResult r = ms::rmsprop(sphere, sphere_grad, x0, 0.01, 0.9, 1e-8, 2000);
    EXPECT_LT(r.f_val, 1e-3);
    EXPECT_TRUE(r.converged);
}

TEST(OptimRMSprop, IllConditionedQuadratic_StillConverges) {
    const std::vector<double> diag = {1.0, 10.0, 100.0};
    ms::FuncND f = [&diag](const std::vector<double>& x) {
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            s += 0.5 * diag[i] * x[i] * x[i];
        }
        return s;
    };
    ms::GradND g = make_quadratic_grad(diag);
    const double f0 = f({10.0, -5.0, 2.0});
    ms::OptimResult r = ms::rmsprop(
        f, g, {10.0, -5.0, 2.0}, 0.05, 0.9, 1e-8, 8000);
    EXPECT_LT(r.f_val, 0.5 * f0);
    EXPECT_LT(r.f_val, 1.0);
}

// -----------------------------------------------------------------------
// Adadelta (adaptive learning rate)
// -----------------------------------------------------------------------
TEST(OptimAdadelta, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, {2.0, 3.0}, 1.0, 0.95, 1e-6, 3000);
    EXPECT_LT(r.f_val, 1e-4);
    EXPECT_NEAR(r.x[0], 0.0, 1e-2);
    EXPECT_NEAR(r.x[1], 0.0, 1e-2);
}

TEST(OptimAdadelta, Rosenbrock_ImprovesObjective) {
    const double f0 = rosenbrock({-1.0, 1.0});
    ms::OptimResult r = ms::adadelta(
        rosenbrock, rosenbrock_grad, {-1.0, 1.0}, 1.0, 0.95, 1e-6, 8000);
    EXPECT_LT(r.f_val, f0);
    EXPECT_LT(r.f_val, 10.0);
}

TEST(OptimAdadelta, QuadraticMinimum_Found) {
    const std::vector<double> diag = {2.0, 4.0, 8.0};
    ms::FuncND f = [&diag](const std::vector<double>& x) {
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            s += 0.5 * diag[i] * x[i] * x[i];
        }
        return s;
    };
    ms::GradND g = make_quadratic_grad(diag);
    ms::OptimResult r = ms::adadelta(f, g, {5.0, -3.0, 2.0}, 1.0, 0.95, 1e-6, 8000);
    EXPECT_LT(r.f_val, 0.02);
    EXPECT_NEAR(r.x[0], 0.0, 5e-2);
    EXPECT_NEAR(r.x[1], 0.0, 5e-2);
    EXPECT_NEAR(r.x[2], 0.0, 5e-2);
}

TEST(OptimAdadelta, ResultIsFinite) {
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, {3.0, -2.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    for (auto v : r.x) EXPECT_TRUE(std::isfinite(v));
}

TEST(OptimAdadelta, CorrectDimension) {
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, {1.0, 2.0, 3.0});
    EXPECT_EQ(r.x.size(), 3u);
}

TEST(OptimAdadelta, AlreadyAtMinimum_TerminatesImmediately) {
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, {0.0, 0.0});
    EXPECT_LT(r.f_val, 1e-12);
    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.iterations, 2u);
}

TEST(OptimAdadelta, Sphere2D_ReducesObjective) {
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, {2.0, 2.0});
    EXPECT_LT(r.f_val, 4.0);
}

TEST(OptimAdadelta, IterationsWithinMaxIter) {
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, {1.0, 1.0}, 1.0, 0.95, 1e-6, 50);
    EXPECT_LE(r.iterations, 50u);
}

TEST(OptimAdadelta, HighDimQuadratic_Converges) {
    const int n = 8;
    std::vector<double> x0(static_cast<size_t>(n), 1.0);
    ms::OptimResult r = ms::adadelta(sphere, sphere_grad, x0, 1.0, 0.95, 1e-6, 2000);
    EXPECT_LT(r.f_val, 1e-3);
    EXPECT_TRUE(r.converged);
}

TEST(OptimAdadelta, IllConditionedQuadratic_StillConverges) {
    const std::vector<double> diag = {1.0, 10.0, 100.0};
    ms::FuncND f = [&diag](const std::vector<double>& x) {
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            s += 0.5 * diag[i] * x[i] * x[i];
        }
        return s;
    };
    ms::GradND g = make_quadratic_grad(diag);
    const double f0 = f({10.0, -5.0, 2.0});
    ms::OptimResult r = ms::adadelta(
        f, g, {10.0, -5.0, 2.0}, 1.0, 0.95, 1e-6, 8000);
    EXPECT_LT(r.f_val, 0.5 * f0);
    EXPECT_LT(r.f_val, 1.0);
}

// -----------------------------------------------------------------------
// Levenberg-Marquardt (nonlinear least squares)
// -----------------------------------------------------------------------
namespace {

// Exponential decay model: y = A*exp(-k*t)
double exp_decay_model(double A, double k, double t) {
    return A * std::exp(-k * t);
}

// Gaussian peak model: y = A*exp(-(t-mu)^2 / (2*sigma^2))
double gaussian_model(double A, double mu, double sigma, double t) {
    double d = (t - mu) / sigma;
    return A * std::exp(-0.5 * d * d);
}

// Power law model: y = A*t^p
double power_law_model(double A, double p, double t) {
    return A * std::pow(t, p);
}

} // namespace

TEST(OptimLM, ExponentialDecay_RecoversTrueParameters) {
    // True model: y = 5.0 * exp(-0.5*t), noise-free synthetic data
    const double A_true = 5.0, k_true = 0.5;
    std::vector<double> ts = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(exp_decay_model(A_true, k_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = exp_decay_model(p[0], p[1], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {1.0, 1.0});
    EXPECT_NEAR(r.x[0], A_true, 1e-4);
    EXPECT_NEAR(r.x[1], k_true, 1e-4);
    EXPECT_LT(r.f_val, 1e-8);
}

TEST(OptimLM, ExponentialDecay_FValNearZero) {
    const double A_true = 2.5, k_true = 1.2;
    std::vector<double> ts = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(exp_decay_model(A_true, k_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = exp_decay_model(p[0], p[1], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {1.0, 0.1});
    EXPECT_LT(r.f_val, 1e-10);
}

TEST(OptimLM, ExponentialDecay_ConvergesFromPoorInitialGuess) {
    // Initial guess far from the truth: A off by 10x, k off by 20x.
    const double A_true = 4.0, k_true = 0.3;
    std::vector<double> ts = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(exp_decay_model(A_true, k_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = exp_decay_model(p[0], p[1], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {40.0, 6.0}, 500);
    ASSERT_TRUE(std::isfinite(r.x[0]));
    ASSERT_TRUE(std::isfinite(r.x[1]));
    EXPECT_NEAR(r.x[0], A_true, 1e-3);
    EXPECT_NEAR(r.x[1], k_true, 1e-3);
    EXPECT_LT(r.f_val, 1e-6);
}

TEST(OptimLM, GaussianPeak_RecoversTrueParameters) {
    // True model: y = 3.0*exp(-(t-2.0)^2 / (2*1.5^2))
    const double A_true = 3.0, mu_true = 2.0, sigma_true = 1.5;
    std::vector<double> ts = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(gaussian_model(A_true, mu_true, sigma_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = gaussian_model(p[0], p[1], p[2], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {1.0, 1.0, 1.0});
    EXPECT_NEAR(r.x[0], A_true, 1e-3);
    EXPECT_NEAR(r.x[1], mu_true, 1e-3);
    EXPECT_NEAR(r.x[2], sigma_true, 1e-3);
    EXPECT_LT(r.f_val, 1e-6);
}

TEST(OptimLM, GaussianPeak_FValNearZero) {
    const double A_true = 1.5, mu_true = 0.5, sigma_true = 0.8;
    std::vector<double> ts = {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(gaussian_model(A_true, mu_true, sigma_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = gaussian_model(p[0], p[1], p[2], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {1.0, 0.0, 1.0});
    EXPECT_LT(r.f_val, 1e-8);
}

TEST(OptimLM, PowerLaw_RecoversTrueParameters) {
    // True model: y = 2.0*t^1.5
    const double A_true = 2.0, p_true = 1.5;
    std::vector<double> ts = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(power_law_model(A_true, p_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = power_law_model(p[0], p[1], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {1.0, 1.0});
    EXPECT_NEAR(r.x[0], A_true, 1e-3);
    EXPECT_NEAR(r.x[1], p_true, 1e-3);
    EXPECT_LT(r.f_val, 1e-6);
}

TEST(OptimLM, LinearRegression_MatchesClosedFormOLS) {
    // y = m*t + b, true m=2.0, b=1.0, with residuals as a nonlinear-least-
    // squares call (even though the model itself is linear).
    const double m_true = 2.0, b_true = 1.0;
    std::vector<double> ts = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(m_true * t + b_true);

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = p[0] * ts[i] + p[1] - ys[i];
        }
        return r;
    };

    // Closed-form OLS via normal equations for y = m*t + b.
    const size_t N = ts.size();
    double sum_t = 0.0, sum_y = 0.0, sum_tt = 0.0, sum_ty = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum_t += ts[i];
        sum_y += ys[i];
        sum_tt += ts[i] * ts[i];
        sum_ty += ts[i] * ys[i];
    }
    const double n = static_cast<double>(N);
    const double m_ols = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t);
    const double b_ols = (sum_y - m_ols * sum_t) / n;

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.5, 0.5});
    EXPECT_NEAR(r.x[0], m_ols, 1e-6);
    EXPECT_NEAR(r.x[1], b_ols, 1e-6);
    EXPECT_NEAR(r.x[0], m_true, 1e-6);
    EXPECT_NEAR(r.x[1], b_true, 1e-6);
}

TEST(OptimLM, LinearRegression_FValNearZero) {
    std::vector<double> ts = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> ys = {1.0, 3.0, 5.0, 7.0};  // y = 2t + 1 exactly

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = p[0] * ts[i] + p[1] - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.0, 0.0});
    EXPECT_LT(r.f_val, 1e-12);
}

TEST(OptimLM, TrivialSingleParameter_RecoversScaleFactor) {
    // y = c*t, true c = 3.0
    const double c_true = 3.0;
    std::vector<double> ts = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(c_true * t);

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) r[i] = p[0] * ts[i] - ys[i];
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {1.0});
    ASSERT_EQ(r.x.size(), 1u);
    EXPECT_NEAR(r.x[0], c_true, 1e-6);
    EXPECT_LT(r.f_val, 1e-10);
}

TEST(OptimLM, TrivialSingleParameter_NegativeScaleFactor) {
    const double c_true = -2.5;
    std::vector<double> ts = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(c_true * t);

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) r[i] = p[0] * ts[i] - ys[i];
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.0});
    EXPECT_NEAR(r.x[0], c_true, 1e-6);
}

TEST(OptimLM, AlreadyAtOptimum_StaysNearGoodStart) {
    // Start exactly at the true parameters: should stay there (or improve
    // negligibly further), not diverge away from a near-zero-cost point.
    const double A_true = 5.0, k_true = 0.5;
    std::vector<double> ts = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(exp_decay_model(A_true, k_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = exp_decay_model(p[0], p[1], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {A_true, k_true});
    EXPECT_NEAR(r.x[0], A_true, 1e-6);
    EXPECT_NEAR(r.x[1], k_true, 1e-6);
    EXPECT_LT(r.f_val, 1e-12);
    EXPECT_LE(r.iterations, 10u) << "Should converge almost immediately from the optimum";
}

TEST(OptimLM, AlreadyAtOptimum_ConvergedFlagTrue) {
    const double c_true = 4.0;
    std::vector<double> ts = {1.0, 2.0, 3.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(c_true * t);

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) r[i] = p[0] * ts[i] - ys[i];
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {c_true});
    EXPECT_TRUE(r.converged);
}

TEST(OptimLM, ResultIsFinite) {
    ms::ResidualFunc residuals = [](const std::vector<double>& p) {
        return std::vector<double>{p[0] - 2.0, p[1] + 3.0};
    };
    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.0, 0.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    for (double v : r.x) EXPECT_TRUE(std::isfinite(v));
}

TEST(OptimLM, CorrectDimensionPreserved) {
    ms::ResidualFunc residuals = [](const std::vector<double>& p) {
        return std::vector<double>{p[0] - 1.0, p[1] - 2.0, p[2] - 3.0};
    };
    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.0, 0.0, 0.0});
    EXPECT_EQ(r.x.size(), 3u);
    EXPECT_NEAR(r.x[0], 1.0, 1e-6);
    EXPECT_NEAR(r.x[1], 2.0, 1e-6);
    EXPECT_NEAR(r.x[2], 3.0, 1e-6);
}

TEST(OptimLM, OverdeterminedSystem_MoreResidualsThanParameters) {
    // 10 residuals, 2 parameters: exercises the m > n case throughout.
    const double A_true = 1.8, k_true = 0.7;
    std::vector<double> ts;
    for (int i = 0; i < 10; ++i) ts.push_back(0.1 * i);
    std::vector<double> ys;
    for (double t : ts) ys.push_back(exp_decay_model(A_true, k_true, t));

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            r[i] = exp_decay_model(p[0], p[1], ts[i]) - ys[i];
        }
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.5, 0.1});
    EXPECT_NEAR(r.x[0], A_true, 1e-4);
    EXPECT_NEAR(r.x[1], k_true, 1e-4);
}

TEST(OptimLM, IterationsWithinMaxIter) {
    ms::ResidualFunc residuals = [](const std::vector<double>& p) {
        return std::vector<double>{p[0] - 7.0};
    };
    ms::OptimResult r = ms::levenberg_marquardt(residuals, {0.0}, 50);
    EXPECT_LE(r.iterations, 50u);
    EXPECT_NEAR(r.x[0], 7.0, 1e-6);
}

TEST(OptimLM, CustomLambda0_StillConverges) {
    const double c_true = 6.0;
    std::vector<double> ts = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> ys;
    for (double t : ts) ys.push_back(c_true * t);

    ms::ResidualFunc residuals = [&](const std::vector<double>& p) {
        std::vector<double> r(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) r[i] = p[0] * ts[i] - ys[i];
        return r;
    };

    ms::OptimResult r = ms::levenberg_marquardt(residuals, {20.0}, 200, 1e-10, 1.0);
    EXPECT_NEAR(r.x[0], c_true, 1e-4);
}

// -----------------------------------------------------------------------
// Simulated Annealing
// -----------------------------------------------------------------------
TEST(OptimSimulatedAnnealing, Sphere2D_FindsNearOptimum) {
    ms::OptimResult r = ms::simulated_annealing(
        sphere, {2.0, 2.0}, 1.0, 0.99, 5000, 42);
    EXPECT_LT(r.f_val, 2.0);
}

TEST(OptimSimulatedAnnealing, ResultIsFinite) {
    ms::OptimResult r = ms::simulated_annealing(
        sphere, {3.0, -3.0}, 2.0, 0.995, 3000, 99);
    EXPECT_TRUE(std::isfinite(r.f_val));
}

TEST(OptimSimulatedAnnealing, CorrectDimension) {
    ms::OptimResult r = ms::simulated_annealing(
        sphere, {1.0, 2.0, 3.0}, 1.0, 0.99, 1000, 1);
    EXPECT_EQ(r.x.size(), 3u);
}

// -----------------------------------------------------------------------
// Differential Evolution
// -----------------------------------------------------------------------
TEST(OptimDifferentialEvolution, Sphere2D_FindsNearOptimum) {
    std::vector<std::pair<double,double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    ms::OptimResult r = ms::differential_evolution(sphere, bounds, 20, 0.8, 0.9, 500, 42);
    EXPECT_LT(r.f_val, 0.5);
}

TEST(OptimDifferentialEvolution, ResultIsFinite) {
    std::vector<std::pair<double,double>> bounds = {{-10.0, 10.0}, {-10.0, 10.0}};
    ms::OptimResult r = ms::differential_evolution(sphere, bounds, 10, 0.8, 0.9, 200, 7);
    EXPECT_TRUE(std::isfinite(r.f_val));
    EXPECT_EQ(r.x.size(), 2u);
}

TEST(OptimDifferentialEvolution, RespectsRespectsBounds) {
    std::vector<std::pair<double,double>> bounds = {{-2.0, 2.0}, {-2.0, 2.0}};
    ms::OptimResult r = ms::differential_evolution(sphere, bounds, 15, 0.8, 0.9, 300, 3);
    EXPECT_GE(r.x[0], -2.0);
    EXPECT_LE(r.x[0],  2.0);
    EXPECT_GE(r.x[1], -2.0);
    EXPECT_LE(r.x[1],  2.0);
}

// -----------------------------------------------------------------------
// Particle Swarm
// -----------------------------------------------------------------------
TEST(OptimParticleSwarm, Sphere2D_ReducesObjective) {
    std::vector<std::pair<double,double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    ms::OptimResult r = ms::particle_swarm(sphere, bounds, 20, 200, 42);
    EXPECT_LT(r.f_val, 1.0);
}

TEST(OptimParticleSwarm, ResultIsFinite) {
    std::vector<std::pair<double,double>> bounds = {{-10.0, 10.0}, {-10.0, 10.0}};
    ms::OptimResult r = ms::particle_swarm(sphere, bounds, 15, 100, 99);
    EXPECT_TRUE(std::isfinite(r.f_val));
    EXPECT_EQ(r.x.size(), 2u);
}

TEST(OptimParticleSwarm, BestNotWorse) {
    std::vector<std::pair<double,double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    ms::OptimResult r = ms::particle_swarm(sphere, bounds, 20, 300, 42);
    EXPECT_LE(r.f_val, sphere(r.x));  // should be the best value
}

// -----------------------------------------------------------------------
// CMA-ES
// -----------------------------------------------------------------------
TEST(OptimCMAES, Sphere2D_ConvergesFromOriginOffset) {
    ms::OptimResult r = ms::cmaes(sphere, {2.0, 3.0}, 0.5, 500, 42);
    EXPECT_LT(r.f_val, 1e-4);
    EXPECT_EQ(r.x.size(), 2u);
}

TEST(OptimCMAES, Sphere2D_ConvergesFromNegativeStart) {
    ms::OptimResult r = ms::cmaes(sphere, {-3.0, 4.0}, 0.5, 500, 7);
    EXPECT_LT(r.f_val, 1e-4);
}

TEST(OptimCMAES, Sphere2D_ConvergesFromFarStart) {
    ms::OptimResult r = ms::cmaes(sphere, {5.0, -5.0}, 1.0, 800, 99);
    EXPECT_LT(r.f_val, 1e-4);
}

TEST(OptimCMAES, Rosenbrock2D_Converges) {
    ms::OptimResult r = ms::cmaes(rosenbrock, {0.0, 0.0}, 0.3, 1500, 42);
    EXPECT_LT(r.f_val, 1.0);
}

TEST(OptimCMAES, Rosenbrock2D_ConvergesFromAlternateStart) {
    ms::OptimResult r = ms::cmaes(rosenbrock, {-1.0, 1.0}, 0.5, 2000, 123);
    EXPECT_LT(r.f_val, 1.0);
}

TEST(OptimCMAES, Sphere5D_Converges) {
    ms::OptimResult r = ms::cmaes(
        sphere, {1.0, 1.0, 1.0, 1.0, 1.0}, 0.5, 800, 42);
    EXPECT_LT(r.f_val, 1e-3);
    EXPECT_EQ(r.x.size(), 5u);
}

TEST(OptimCMAES, Sphere10D_Converges) {
    ms::OptimResult r = ms::cmaes(
        sphere,
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        0.5, 1000, 42);
    EXPECT_LT(r.f_val, 1e-2);
    EXPECT_EQ(r.x.size(), 10u);
}

TEST(OptimCMAES, CorrectDimension) {
    ms::OptimResult r = ms::cmaes(sphere, {1.0, 2.0, 3.0}, 0.5, 300, 1);
    EXPECT_EQ(r.x.size(), 3u);
    EXPECT_TRUE(std::isfinite(r.f_val));
    for (auto v : r.x) EXPECT_TRUE(std::isfinite(v));
}

TEST(OptimCMAES, DeterministicWithSameSeed) {
    ms::OptimResult r1 = ms::cmaes(sphere, {2.0, -1.0}, 0.5, 400, 42);
    ms::OptimResult r2 = ms::cmaes(sphere, {2.0, -1.0}, 0.5, 400, 42);
    EXPECT_DOUBLE_EQ(r1.f_val, r2.f_val);
    ASSERT_EQ(r1.x.size(), r2.x.size());
    for (size_t i = 0; i < r1.x.size(); ++i) {
        EXPECT_DOUBLE_EQ(r1.x[i], r2.x[i]);
    }
}

TEST(OptimCMAES, Sphere1D_EdgeCase) {
    ms::FuncND f1d = [](const std::vector<double>& x) { return x[0] * x[0]; };
    ms::OptimResult r = ms::cmaes(f1d, {3.0}, 0.5, 300, 42);
    EXPECT_EQ(r.x.size(), 1u);
    EXPECT_LT(r.f_val, 1e-4);
    EXPECT_TRUE(std::isfinite(r.f_val));
}

// -----------------------------------------------------------------------
// Bisection
// -----------------------------------------------------------------------
TEST(OptimBisection, FindsRootOfLinear) {
    // f(x) = x - 3  => root at 3
    ms::Func1D f = [](double x) { return x - 3.0; };
    double r = ms::bisection(f, 0.0, 10.0);
    EXPECT_NEAR(r, 3.0, 1e-8);
}

TEST(OptimBisection, FindsRootOfSine) {
    // sin(x) = 0 in [1, 4] => root at pi
    double r = ms::bisection([](double x){ return std::sin(x); }, 1.0, 4.0);
    EXPECT_NEAR(r, M_PI, 1e-8);
}

// -----------------------------------------------------------------------
// Brent's method
// -----------------------------------------------------------------------
TEST(OptimBrentq, FindsRootOfCubic) {
    // f(x) = x^3 - 2x - 5 => root near 2.0946
    ms::Func1D f = [](double x) { return x * x * x - 2.0 * x - 5.0; };
    double r = ms::brentq(f, 1.0, 3.0);
    EXPECT_NEAR(f(r), 0.0, 1e-8);
}

TEST(OptimBrentq, FindsRootOfCos) {
    double r = ms::brentq([](double x){ return std::cos(x); }, 0.0, 2.0);
    EXPECT_NEAR(r, M_PI / 2.0, 1e-8);
}

// -----------------------------------------------------------------------
// Illinois method (anti-stagnation regula falsi)
// -----------------------------------------------------------------------
TEST(OptimIllinois, FindsRootOfSquareMinusTwo) {
    // f(x) = x^2 - 2 => root at sqrt(2)
    ms::Func1D f = [](double x) { return x * x - 2.0; };
    double r = ms::illinois(f, 0.0, 2.0);
    EXPECT_NEAR(r, std::sqrt(2.0), 1e-8);
}

TEST(OptimIllinois, FindsRootOfCosMinusX) {
    // f(x) = cos(x) - x, root near 0.7390851332
    ms::Func1D f = [](double x) { return std::cos(x) - x; };
    double r = ms::illinois(f, 0.0, 1.0);
    EXPECT_NEAR(f(r), 0.0, 1e-8);
    EXPECT_NEAR(r, 0.7390851332, 1e-6);
}

TEST(OptimIllinois, ConvergesOnStagnationProneBracket) {
    // f(x) = x^10 - 1 on [0, 1.5]: extremely convex near the left endpoint,
    // a classic case where plain regula falsi keeps the left endpoint fixed
    // for many iterations and converges very slowly. Illinois should still
    // converge to the root at x = 1 well within max_iter.
    ms::Func1D f = [](double x) { return std::pow(x, 10) - 1.0; };
    double r = ms::illinois(f, 0.0, 1.5, 1e-9, 200);
    EXPECT_NEAR(r, 1.0, 1e-6);
    EXPECT_NEAR(f(r), 0.0, 1e-6);
}

TEST(OptimIllinois, ConvergesFasterThanPlainRegulaFalsiOnStagnationCase) {
    // Instrumented copies of plain regula falsi and Illinois on the same
    // stagnation-prone bracket, counting iterations to reach the same
    // tolerance. Illinois should need substantially fewer iterations since
    // regula falsi keeps `a` pinned at 0 for a very long time.
    ms::Func1D f = [](double x) { return std::pow(x, 10) - 1.0; };
    const double a0 = 0.0, b0 = 1.5, tol = 1e-9;
    const int cap = 500;

    auto regula_falsi_iters = [&]() {
        double a = a0, b = b0, fa = f(a), fb = f(b);
        for (int i = 0; i < cap; ++i) {
            double c = b - fb * (b - a) / (fb - fa);
            double fc = f(c);
            if (std::abs(fc) < tol) return i + 1;
            if (fa * fc < 0.0) { b = c; fb = fc; }
            else { a = c; fa = fc; }
        }
        return cap;
    };

    auto illinois_iters = [&]() {
        double a = a0, b = b0, fa = f(a), fb = f(b);
        int retained = 0;
        for (int i = 0; i < cap; ++i) {
            double c = b - fb * (b - a) / (fb - fa);
            double fc = f(c);
            if (std::abs(fc) < tol) return i + 1;
            if (fa * fc < 0.0) {
                b = c; fb = fc;
                if (retained == -1) fa *= 0.5;
                retained = -1;
            } else {
                a = c; fa = fc;
                if (retained == 1) fb *= 0.5;
                retained = 1;
            }
        }
        return cap;
    };

    int rf_iters = regula_falsi_iters();
    int il_iters = illinois_iters();
    EXPECT_LT(il_iters, rf_iters);
    EXPECT_LE(il_iters, 200);
}

TEST(OptimIllinois, AgreesWithBrentqAndBisectionOnSmoothFunctions) {
    ms::Func1D f1 = [](double x) { return x * x * x - 2.0 * x - 5.0; };
    double r_illinois1 = ms::illinois(f1, 1.0, 3.0);
    double r_brentq1 = ms::brentq(f1, 1.0, 3.0);
    EXPECT_NEAR(r_illinois1, r_brentq1, 1e-6);

    ms::Func1D f2 = [](double x) { return std::sin(x); };
    double r_illinois2 = ms::illinois(f2, 1.0, 4.0);
    double r_bisection2 = ms::bisection(f2, 1.0, 4.0);
    EXPECT_NEAR(r_illinois2, r_bisection2, 1e-6);
    EXPECT_NEAR(r_illinois2, M_PI, 1e-6);
}

TEST(OptimIllinois, NonBracketingIntervalReturnsNaN) {
    // f(x) = x^2 + 1 has no real root, and f(a), f(b) have the same sign
    // for any a, b => not a valid bracket.
    ms::Func1D f = [](double x) { return x * x + 1.0; };
    double r = ms::illinois(f, 0.0, 2.0);
    EXPECT_TRUE(std::isnan(r));
}

TEST(OptimIllinois, RootAtLeftEndpointHandledGracefully) {
    // f(x) = x - 1, f(a) == 0 exactly at a = 1.
    ms::Func1D f = [](double x) { return x - 1.0; };
    double r = ms::illinois(f, 1.0, 2.0);
    EXPECT_NEAR(r, 1.0, 1e-9);
    EXPECT_TRUE(std::isfinite(r));
}

TEST(OptimIllinois, RootAtRightEndpointHandledGracefully) {
    // f(x) = x - 2, f(b) == 0 exactly at b = 2.
    ms::Func1D f = [](double x) { return x - 2.0; };
    double r = ms::illinois(f, 1.0, 2.0);
    EXPECT_NEAR(r, 2.0, 1e-9);
    EXPECT_TRUE(std::isfinite(r));
}

// -----------------------------------------------------------------------
// Secant method
// -----------------------------------------------------------------------
TEST(OptimSecant, FindsRootOfQuadratic) {
    ms::Func1D f = [](double x) { return x * x - 4.0; };
    double r = ms::secant(f, 1.0, 3.0);
    EXPECT_NEAR(std::abs(r), 2.0, 1e-6);
}

// -----------------------------------------------------------------------
// Halley's method
// -----------------------------------------------------------------------
TEST(OptimHalley, FindsRootOfCubic) {
    ms::Func1D f  = [](double x) { return x * x * x - x - 1.0; };
    ms::Func1D df = [](double x) { return 3.0 * x * x - 1.0; };
    ms::Func1D d2f = [](double x) { return 6.0 * x; };
    double r = ms::halley(f, df, d2f, 1.5);
    EXPECT_NEAR(f(r), 0.0, 1e-10);
}

// -----------------------------------------------------------------------
// Fixed point
// -----------------------------------------------------------------------
TEST(OptimFixedPoint, SqrtViaFixedPoint) {
    // g(x) = (x + 2/x)/2 => fixed point is sqrt(2)
    ms::Func1D g = [](double x) { return (x + 2.0 / x) / 2.0; };
    double r = ms::fixed_point(g, 1.5);
    EXPECT_NEAR(r, std::sqrt(2.0), 1e-8);
}
