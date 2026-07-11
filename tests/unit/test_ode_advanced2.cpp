// MathScript Advanced ODE Tests (Wave): backward Euler, BVP shooting, DDE, events

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ms/ode/ode.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// ode_backward_euler
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, BackwardEuler_ZeroSteps_ReturnsEmpty) {
    const auto f = [](double, double y) { return -y; };
    const auto result = ode_backward_euler(f, 0.0, 1.0, 1.0, 0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.y.empty());
}

TEST(OdeAdvanced2, BackwardEuler_StiffDecay_RemainsStable) {
    const double k = 50.0;
    const auto f = [k](double, double y) { return -k * y; };
    const double h_unstable = 0.1; // h*k = 5 > 2, forward Euler unstable
    const size_t steps = static_cast<size_t>((1.0 - 0.0) / h_unstable);
    const auto fwd = ode_euler(f, 0.0, 1.0, 1.0, steps);
    const auto bwd = ode_backward_euler(f, 0.0, 1.0, 1.0, steps);
    ASSERT_FALSE(fwd.y.empty());
    ASSERT_FALSE(bwd.y.empty());
    const double fwd_abs = std::abs(fwd.y.back());
    EXPECT_TRUE(!std::isfinite(fwd.y.back()) || fwd_abs > 1e4)
        << "forward Euler should blow up for stiff step";
    EXPECT_TRUE(std::isfinite(bwd.y.back()));
    EXPECT_LT(std::abs(bwd.y.back()), 10.0);
}

TEST(OdeAdvanced2, BackwardEuler_StiffDecay_MatchesAnalytic) {
    const double k = 20.0;
    const auto f = [k](double, double y) { return -k * y; };
    const auto result = ode_backward_euler(f, 0.0, 1.0, 1.0, 500);
    ASSERT_FALSE(result.y.empty());
    const double exact = std::exp(-k * 1.0);
    EXPECT_NEAR(result.y.back(), exact, 0.05);
}

TEST(OdeAdvanced2, BackwardEuler_StiffDecay_ConvergesWithMoreSteps) {
    const double k = 10.0;
    const auto f = [k](double, double y) { return -k * y; };
    const double exact = std::exp(-k * 1.0);
    const auto coarse = ode_backward_euler(f, 0.0, 1.0, 1.0, 20);
    const auto fine = ode_backward_euler(f, 0.0, 1.0, 1.0, 2000);
    const double err_coarse = std::abs(coarse.y.back() - exact);
    const double err_fine = std::abs(fine.y.back() - exact);
    EXPECT_LT(err_coarse, 0.5);
    EXPECT_LT(err_fine, err_coarse);
}

TEST(OdeAdvanced2, BackwardEuler_LinearGrowth_Reasonable) {
    const auto f = [](double, double) { return 1.0; };
    const auto result = ode_backward_euler(f, 0.0, 0.0, 2.0, 200);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 2.0, 0.1);
}

TEST(OdeAdvanced2, BackwardEuler_ProducesMonotoneGrid) {
    const auto f = [](double, double y) { return -y; };
    const auto result = ode_backward_euler(f, 0.0, 1.0, 1.0, 10);
    ASSERT_EQ(result.t.size(), 11u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// ode_bvp_shooting
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, BvpShooting_SinSolution_Converges) {
    const auto f = [](double, double y, double) { return -y; }; // y'' = -y
    const auto result = ode_bvp_shooting(f, 0.0, 0.0, M_PI / 2.0, 1.0, 200);
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0u);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 1.0, 1e-3);
    EXPECT_NEAR(result.y.front(), 0.0, 1e-9);
}

TEST(OdeAdvanced2, BvpShooting_SinSolution_MatchesAnalytic) {
    const auto f = [](double, double y, double) { return -y; };
    const auto result = ode_bvp_shooting(f, 0.0, 0.0, M_PI / 2.0, 1.0, 400);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.t.size(), result.y.size());
    ASSERT_EQ(result.t.size(), result.yp.size());
    for (size_t i = 0; i < result.t.size(); i += result.t.size() / 5) {
        EXPECT_NEAR(result.y[i], std::sin(result.t[i]), 0.02);
    }
    EXPECT_NEAR(result.yp.front(), 1.0, 0.05); // y'(0) = 1 for sin(t)
}

TEST(OdeAdvanced2, BvpShooting_ZeroSteps_ReturnsEmpty) {
    const auto f = [](double, double, double) { return 0.0; };
    const auto result = ode_bvp_shooting(f, 0.0, 0.0, 1.0, 0.0, 0);
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.t.empty());
}

TEST(OdeAdvanced2, BvpShooting_ConstantSolution_Works) {
    // y'' = 0, y(0)=2, y(1)=2 => y=2
    const auto f = [](double, double, double) { return 0.0; };
    const auto result = ode_bvp_shooting(f, 0.0, 2.0, 1.0, 2.0, 50);
    EXPECT_TRUE(result.converged);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 2.0, 1e-4);
    EXPECT_NEAR(result.yp.front(), 0.0, 1e-3);
}

TEST(OdeAdvanced2, BvpShooting_LinearSolution_Works) {
    // y'' = 0, y(0)=0, y(1)=1 => y=t
    const auto f = [](double, double, double) { return 0.0; };
    const auto result = ode_bvp_shooting(f, 0.0, 0.0, 1.0, 1.0, 100);
    EXPECT_TRUE(result.converged);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 1.0, 1e-4);
    EXPECT_NEAR(result.y[result.y.size() / 2], 0.5, 0.01);
}

TEST(OdeAdvanced2, BvpShooting_CosSolution_MatchesBoundary) {
    // y'' = -y, y(0)=1, y(pi)=-1 => y=cos(t)
    const auto f = [](double, double y, double) { return -y; };
    const auto result = ode_bvp_shooting(f, 0.0, 1.0, M_PI, -1.0, 300);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.y.back(), -1.0, 1e-3);
    EXPECT_NEAR(result.y.front(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// ode_dde_fixed_step
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, DdeFixedStep_ZeroSteps_ReturnsEmpty) {
    const auto f = [](double, double, double) { return 0.0; };
    const auto hist = [](double) { return 1.0; };
    const auto result = ode_dde_fixed_step(f, hist, 0.0, 1.0, 0.5, 0);
    EXPECT_TRUE(result.t.empty());
}

TEST(OdeAdvanced2, DdeFixedStep_NonPositiveTau_ReturnsEmpty) {
    const auto f = [](double, double, double) { return 0.0; };
    const auto hist = [](double) { return 1.0; };
    const auto result = ode_dde_fixed_step(f, hist, 0.0, 1.0, 0.0, 10);
    EXPECT_TRUE(result.t.empty());
}

TEST(OdeAdvanced2, DdeFixedStep_LargeTau_ReducesToOde) {
    // tau huge => delayed term always history(0)=3 => dy/dt = -y + 3 => y -> 3
    const auto f = [](double, double y, double yd) { return -y + yd; };
    const auto hist = [](double) { return 3.0; };
    const double tau = 1000.0;
    const auto result = ode_dde_fixed_step(f, hist, 0.0, 5.0, tau, 200);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back(), 3.0, 0.1);
}

TEST(OdeAdvanced2, DdeFixedStep_CoarseVsFine_Agree) {
    const auto f = [](double, double y, double yd) { return -0.5 * y + 0.3 * yd; };
    const auto hist = [](double t) { return 1.0 + 0.1 * t; };
    const double tau = 0.3;
    const auto coarse = ode_dde_fixed_step(f, hist, 0.0, 2.0, tau, 40);
    const auto fine = ode_dde_fixed_step(f, hist, 0.0, 2.0, tau, 400);
    ASSERT_FALSE(coarse.y.empty());
    ASSERT_FALSE(fine.y.empty());
    EXPECT_NEAR(coarse.y.back(), fine.y.back(), 0.05);
}

TEST(OdeAdvanced2, DdeFixedStep_SolutionRemainsBounded) {
    const auto f = [](double, double y, double yd) { return -y + yd; };
    const auto hist = [](double) { return 1.0; };
    const auto result = ode_dde_fixed_step(f, hist, 0.0, 3.0, 0.5, 100);
    ASSERT_FALSE(result.y.empty());
    for (double val : result.y) {
        EXPECT_TRUE(std::isfinite(val));
        EXPECT_LT(std::abs(val), 100.0);
    }
}

TEST(OdeAdvanced2, DdeFixedStep_HistoryUsedAtStart) {
    const auto f = [](double, double, double yd) { return yd; };
    const auto hist = [](double) { return 2.0; };
    const auto result = ode_dde_fixed_step(f, hist, 0.0, 0.5, 0.2, 10);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.front(), 2.0, 1e-12);
}

TEST(OdeAdvanced2, DdeFixedStep_MonotoneTimeGrid) {
    const auto f = [](double, double, double) { return 0.0; };
    const auto hist = [](double) { return 0.0; };
    const auto result = ode_dde_fixed_step(f, hist, 0.0, 1.0, 0.25, 20);
    ASSERT_EQ(result.t.size(), 21u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// ode_event_detect
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, EventDetect_ZeroCrossing_FindsEventNearFive) {
    const auto f = [](double, double) { return 1.0; };
    const auto g = [](double, double y) { return y; };
    const auto result = ode_event_detect(f, g, 0.0, -5.0, 10.0, 100);
    ASSERT_EQ(result.event_times.size(), 1u);
    ASSERT_EQ(result.event_values.size(), 1u);
    EXPECT_NEAR(result.event_times[0], 5.0, 1e-3);
    EXPECT_NEAR(result.event_values[0], 0.0, 1e-3);
}

TEST(OdeAdvanced2, EventDetect_NoEvent_WhenAlwaysPositive) {
    const auto f = [](double, double y) { return y; };
    const auto g = [](double, double y) { return y; };
    const auto result = ode_event_detect(f, g, 0.0, 1.0, 1.0, 50);
    EXPECT_TRUE(result.event_times.empty());
    EXPECT_TRUE(result.event_values.empty());
    ASSERT_FALSE(result.y.empty());
}

TEST(OdeAdvanced2, EventDetect_MultipleEvents_Detected) {
    // dy/dt = 1, g = sin(t) crosses zero multiple times on [0, 2pi]
    const auto f = [](double, double) { return 1.0; };
    const auto g = [](double t, double) { return std::sin(t); };
    const auto result = ode_event_detect(f, g, 0.0, 0.0, 2.0 * M_PI, 500);
    EXPECT_GE(result.event_times.size(), 2u);
    for (double et : result.event_times) {
        EXPECT_NEAR(std::sin(et), 0.0, 1e-2);
    }
}

TEST(OdeAdvanced2, EventDetect_ZeroSteps_ReturnsEmpty) {
    const auto f = [](double, double) { return 1.0; };
    const auto g = [](double, double y) { return y; };
    const auto result = ode_event_detect(f, g, 0.0, -1.0, 1.0, 0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.event_times.empty());
}

TEST(OdeAdvanced2, EventDetect_TrajectoryPreserved) {
    const auto f = [](double, double) { return 1.0; };
    const auto g = [](double, double y) { return y; };
    const auto result = ode_event_detect(f, g, 0.0, -2.0, 4.0, 40);
    ASSERT_EQ(result.t.size(), result.y.size());
    EXPECT_EQ(result.t.size(), 41u);
    EXPECT_NEAR(result.y.front(), -2.0, 1e-12);
}

TEST(OdeAdvanced2, EventDetect_DecayNeverCrosses_NoEvents) {
    const auto f = [](double, double) { return 1.0; };
    const auto g = [](double, double y) { return y - 100.0; }; // stays negative on [0,2]
    const auto result = ode_event_detect(f, g, 0.0, 1.0, 2.0, 100);
    EXPECT_TRUE(result.event_times.empty());
}

TEST(OdeAdvanced2, EventDetect_EventAtStart_NotDuplicated) {
    // y starts at 0, g=y => event at t=0 already; no sign change on first step if stays positive
    const auto f = [](double, double) { return 1.0; };
    const auto g = [](double, double y) { return y; };
    const auto result = ode_event_detect(f, g, 0.0, 0.0, 2.0, 20);
    // No crossing from negative to positive; may have zero or no events
    EXPECT_LE(result.event_times.size(), 1u);
}

// ---------------------------------------------------------------------------
// ode_backward_euler_vec
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, BackwardEulerVec_ZeroSteps_ReturnsEmpty) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{-y[0]};
    };
    const auto result = ode_backward_euler_vec(f, 0.0, {1.0}, 1.0, 0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.y.empty());
}

TEST(OdeAdvanced2, BackwardEulerVec_EmptyY0_ReturnsEmpty) {
    const auto f = [](double, const std::vector<double>&) {
        return std::vector<double>{};
    };
    const auto result = ode_backward_euler_vec(f, 0.0, {}, 1.0, 10);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.y.empty());
}

TEST(OdeAdvanced2, BackwardEulerVec_Stiff2x2_RemainsStable) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{-100.0 * y[0], -y[0] - y[1]};
    };
    const std::vector<double> y0 = {1.0, 1.0};
    const size_t steps = 200; // h = 0.005, within Picard convergence for the fast mode
    const auto bwd = ode_backward_euler_vec(f, 0.0, y0, 1.0, steps);
    ASSERT_FALSE(bwd.y.empty());
    EXPECT_TRUE(std::isfinite(bwd.y.back()[0]));
    EXPECT_TRUE(std::isfinite(bwd.y.back()[1]));
    EXPECT_LT(std::abs(bwd.y.back()[0]), 1.0);
    EXPECT_LT(std::abs(bwd.y.back()[1]), 5.0);
    const double exact_fast = std::exp(-100.0 * 1.0);
    EXPECT_NEAR(std::abs(bwd.y.back()[0]), exact_fast, 0.1);
}

TEST(OdeAdvanced2, BackwardEulerVec_Stiff2x2_ForwardBlowsUp) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{-100.0 * y[0], -2.0 * y[1]};
    };
    const std::vector<double> y0 = {1.0, 1.0};
    const size_t fwd_steps = 40; // h = 0.025, explicit Euler unstable on the -100 mode
    const auto fwd = ode_euler_vec(f, 0.0, y0, 1.0, fwd_steps);
    const auto bwd = ode_backward_euler_vec(f, 0.0, y0, 1.0, 250);
    ASSERT_FALSE(fwd.y.empty());
    ASSERT_FALSE(bwd.y.empty());
    const double fwd_norm = std::abs(fwd.y.back()[0]) + std::abs(fwd.y.back()[1]);
    EXPECT_TRUE(!std::isfinite(fwd.y.back()[0]) || fwd_norm > 1e4)
        << "forward Euler should blow up on stiff 2x2 system";
    EXPECT_TRUE(std::isfinite(bwd.y.back()[0]));
    EXPECT_TRUE(std::isfinite(bwd.y.back()[1]));
    EXPECT_LT(std::abs(bwd.y.back()[0]), 10.0);
}

TEST(OdeAdvanced2, BackwardEulerVec_Stiff2x2_ComponentsDecay) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{-50.0 * y[0], -2.0 * y[1]};
    };
    const auto result = ode_backward_euler_vec(f, 0.0, {1.0, 1.0}, 1.0, 500);
    ASSERT_FALSE(result.y.empty());
    EXPECT_LT(std::abs(result.y.back()[0]), 0.1);
    EXPECT_LT(std::abs(result.y.back()[1]), 0.5);
}

TEST(OdeAdvanced2, BackwardEulerVec_HarmonicOscillator_MatchesAnalytic) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const std::vector<double> y0 = {0.0, 1.0};
    const auto result = ode_backward_euler_vec(f, 0.0, y0, M_PI, 400);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back()[0], std::sin(M_PI), 0.05);
    EXPECT_NEAR(result.y.back()[1], std::cos(M_PI), 0.05);
}

TEST(OdeAdvanced2, BackwardEulerVec_HarmonicOscillator_MidTrajectory) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const std::vector<double> y0 = {0.0, 1.0};
    const auto result = ode_backward_euler_vec(f, 0.0, y0, M_PI / 2.0, 300);
    ASSERT_GE(result.y.size(), 2u);
    const size_t mid = result.y.size() / 2;
    EXPECT_NEAR(result.y[mid][0], std::sin(result.t[mid]), 0.05);
    EXPECT_NEAR(result.y[mid][1], std::cos(result.t[mid]), 0.05);
}

TEST(OdeAdvanced2, BackwardEulerVec_HarmonicOscillator_FineVsCoarse) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const std::vector<double> y0 = {0.0, 1.0};
    const double exact_y = std::sin(1.0);
    const double exact_yp = std::cos(1.0);
    const auto coarse = ode_backward_euler_vec(f, 0.0, y0, 1.0, 50);
    const auto fine = ode_backward_euler_vec(f, 0.0, y0, 1.0, 2000);
    const double err_coarse = std::abs(coarse.y.back()[0] - exact_y)
                            + std::abs(coarse.y.back()[1] - exact_yp);
    const double err_fine = std::abs(fine.y.back()[0] - exact_y)
                          + std::abs(fine.y.back()[1] - exact_yp);
    EXPECT_LT(err_coarse, 0.5);
    EXPECT_LT(err_fine, err_coarse);
}

TEST(OdeAdvanced2, BackwardEulerVec_ProducesMonotoneGrid) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{-y[0]};
    };
    const auto result = ode_backward_euler_vec(f, 0.0, {1.0}, 1.0, 10);
    ASSERT_EQ(result.t.size(), 11u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

TEST(OdeAdvanced2, BackwardEulerVec_LinearGrowth_Reasonable) {
    const auto f = [](double, const std::vector<double>&) {
        return std::vector<double>{1.0, 2.0};
    };
    const auto result = ode_backward_euler_vec(f, 0.0, {0.0, 0.0}, 2.0, 200);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back()[0], 2.0, 0.15);
    EXPECT_NEAR(result.y.back()[1], 4.0, 0.3);
}

TEST(OdeAdvanced2, BackwardEulerVec_SolutionRemainsFinite) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{-10.0 * y[0], -5.0 * y[1]};
    };
    const auto result = ode_backward_euler_vec(f, 0.0, {1.0, 1.0}, 2.0, 100);
    for (const auto& state : result.y) {
        for (double v : state) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

// ---------------------------------------------------------------------------
// ode_dae_index1
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, DaeIndex1_ZeroSteps_ReturnsEmpty) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, 1.0, 0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.y.empty());
    EXPECT_TRUE(result.z.empty());
    EXPECT_FALSE(result.converged);
}

TEST(OdeAdvanced2, DaeIndex1_EmptyInitialState_ReturnsEmpty) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>&) {
        return std::vector<double>{};
    };
    const DaeAlgFunc g = [](double, const std::vector<double>&,
                             const std::vector<double>&) {
        return std::vector<double>{};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {}, {}, 1.0, 10);
    EXPECT_TRUE(result.t.empty());
    EXPECT_FALSE(result.converged);
}

TEST(OdeAdvanced2, DaeIndex1_SinCosSolution_Converged) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, M_PI, 200);
    EXPECT_TRUE(result.converged);
    ASSERT_FALSE(result.y.empty());
    ASSERT_FALSE(result.z.empty());
}

TEST(OdeAdvanced2, DaeIndex1_SinCosSolution_YMatchesAnalytic) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, M_PI, 400);
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.y.back()[0], std::sin(M_PI), 0.05);
}

TEST(OdeAdvanced2, DaeIndex1_SinCosSolution_ZMatchesAnalytic) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, M_PI, 400);
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.z.back()[0], std::cos(M_PI), 0.05);
}

TEST(OdeAdvanced2, DaeIndex1_SinCosSolution_MidTrajectory) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, M_PI / 2.0, 300);
    ASSERT_TRUE(result.converged);
    const size_t mid = result.y.size() / 2;
    EXPECT_NEAR(result.y[mid][0], std::sin(result.t[mid]), 0.05);
    EXPECT_NEAR(result.z[mid][0], std::cos(result.t[mid]), 0.05);
}

TEST(OdeAdvanced2, DaeIndex1_SinCosSolution_FineVsCoarse) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const double exact = std::sin(1.0);
    const auto coarse = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, 1.0, 50);
    const auto fine = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, 1.0, 500);
    EXPECT_TRUE(coarse.converged);
    EXPECT_TRUE(fine.converged);
    const double err_coarse = std::abs(coarse.y.back()[0] - exact);
    const double err_fine = std::abs(fine.y.back()[0] - exact);
    EXPECT_LT(err_coarse, 0.2);
    EXPECT_LT(err_fine, err_coarse);
}

TEST(OdeAdvanced2, DaeIndex1_IllPosed_NotConverged) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] * z[0] + 1.0};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {0.0}, 1.0, 20);
    EXPECT_FALSE(result.converged);
}

TEST(OdeAdvanced2, DaeIndex1_IllPosed_RemainsFinite) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] * z[0] + 1.0};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {0.0}, 1.0, 20);
    for (const auto& y_state : result.y) {
        for (double v : y_state) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
    for (const auto& z_state : result.z) {
        for (double v : z_state) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}

TEST(OdeAdvanced2, DaeIndex1_InconsistentConstraints_NotConverged) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t), z[0] + std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, 1.0, 50);
    EXPECT_FALSE(result.converged);
}

TEST(OdeAdvanced2, DaeIndex1_ProducesMonotoneGrid) {
    const DaeDiffFunc f = [](double, const std::vector<double>&,
                              const std::vector<double>& z) {
        return std::vector<double>{z[0]};
    };
    const DaeAlgFunc g = [](double t, const std::vector<double>&,
                             const std::vector<double>& z) {
        return std::vector<double>{z[0] - std::cos(t)};
    };
    const auto result = ode_dae_index1(f, g, 0.0, {0.0}, {1.0}, 1.0, 20);
    ASSERT_EQ(result.t.size(), 21u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// ode_verlet
// ---------------------------------------------------------------------------

namespace {

double harmonic_energy(double q, double v) {
    return 0.5 * v * v + 0.5 * q * q;
}

double harmonic_energy_vec(const std::vector<double>& q,
                           const std::vector<double>& v) {
    double e = 0.0;
    for (size_t i = 0; i < q.size(); ++i) {
        e += 0.5 * v[i] * v[i] + 0.5 * q[i] * q[i];
    }
    return e;
}

} // namespace

TEST(OdeAdvanced2, Verlet_ZeroSteps_ReturnsEmpty) {
    const auto a = [](double, double q) { return -q; };
    const auto result = ode_verlet(a, 0.0, 1.0, 0.0, 10.0, 0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.q.empty());
    EXPECT_TRUE(result.v.empty());
}

TEST(OdeAdvanced2, Verlet_HarmonicOscillator_EnergyConserved) {
    const auto a = [](double, double q) { return -q; };
    const size_t periods = 20;
    const size_t steps_per_period = 100;
    const double t_end = static_cast<double>(periods) * 2.0 * M_PI;
    const size_t steps = periods * steps_per_period;
    const auto result = ode_verlet(a, 0.0, 1.0, 0.0, t_end, steps);
    ASSERT_FALSE(result.q.empty());
    ASSERT_FALSE(result.v.empty());
    const double e0 = harmonic_energy(result.q.front(), result.v.front());
    const double e1 = harmonic_energy(result.q.back(), result.v.back());
    EXPECT_NEAR(e1, e0, 1e-3 * e0);
}

TEST(OdeAdvanced2, Verlet_HarmonicOscillator_MatchesAnalytic) {
    const auto a = [](double, double q) { return -q; };
    const auto result = ode_verlet(a, 0.0, 1.0, 0.0, M_PI, 400);
    ASSERT_FALSE(result.q.empty());
    EXPECT_NEAR(result.q.back(), std::cos(M_PI), 0.01);
    EXPECT_NEAR(result.v.back(), -std::sin(M_PI), 0.01);
}

TEST(OdeAdvanced2, Verlet_HarmonicOscillator_MidTrajectory) {
    const auto a = [](double, double q) { return -q; };
    const auto result = ode_verlet(a, 0.0, 1.0, 0.0, M_PI / 2.0, 300);
    ASSERT_GE(result.q.size(), 2u);
    const size_t mid = result.q.size() / 2;
    EXPECT_NEAR(result.q[mid], std::cos(result.t[mid]), 0.01);
    EXPECT_NEAR(result.v[mid], -std::sin(result.t[mid]), 0.01);
}

TEST(OdeAdvanced2, Verlet_HarmonicOscillator_BetterEnergyThanRK4) {
    const auto a = [](double, double q) { return -q; };

    const size_t periods = 20;
    const size_t steps_per_period = 50;
    const double t_end = static_cast<double>(periods) * 2.0 * M_PI;
    const size_t steps = periods * steps_per_period;

    const auto verlet = ode_verlet(a, 0.0, 1.0, 0.0, t_end, steps);
    ASSERT_FALSE(verlet.q.empty());

    const OdeFuncVec sys = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto rk4 = ode_rk4_vec(sys, 0.0, {1.0, 0.0}, t_end, steps);
    ASSERT_FALSE(rk4.y.empty());

    const double e0 = harmonic_energy(verlet.q.front(), verlet.v.front());
    const double e_verlet = harmonic_energy(verlet.q.back(), verlet.v.back());
    const double e_rk4 = harmonic_energy(rk4.y.back()[0], rk4.y.back()[1]);

    EXPECT_LT(std::abs(e_verlet - e0), 0.05 * e0);
    EXPECT_GT(std::abs(e_rk4 - e0), std::abs(e_verlet - e0));
}

TEST(OdeAdvanced2, Verlet_ProducesMonotoneGrid) {
    const auto a = [](double, double q) { return -q; };
    const auto result = ode_verlet(a, 0.0, 1.0, 0.0, 1.0, 10);
    ASSERT_EQ(result.t.size(), 11u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

TEST(OdeAdvanced2, Verlet_HarmonicOscillator_InitialConditions) {
    const auto a = [](double, double q) { return -q; };
    const auto result = ode_verlet(a, 0.0, 1.0, 0.0, 0.1, 10);
    ASSERT_FALSE(result.q.empty());
    EXPECT_NEAR(result.q.front(), 1.0, 1e-12);
    EXPECT_NEAR(result.v.front(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// ode_verlet_vec
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, VerletVec_ZeroSteps_ReturnsEmpty) {
    const auto a = [](double, const std::vector<double>& q) {
        return std::vector<double>{-q[0], -q[1]};
    };
    const auto result = ode_verlet_vec(a, 0.0, {1.0, 0.5}, {0.0, 0.0}, 1.0, 0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.q.empty());
    EXPECT_TRUE(result.v.empty());
}

TEST(OdeAdvanced2, VerletVec_EmptyQ0_ReturnsEmpty) {
    const auto a = [](double, const std::vector<double>&) {
        return std::vector<double>{};
    };
    const auto result = ode_verlet_vec(a, 0.0, {}, {}, 1.0, 10);
    EXPECT_TRUE(result.t.empty());
}

TEST(OdeAdvanced2, VerletVec_MismatchedQ0V0_ReturnsEmpty) {
    const auto a = [](double, const std::vector<double>& q) {
        return std::vector<double>{-q[0]};
    };
    const auto result = ode_verlet_vec(a, 0.0, {1.0}, {0.0, 0.0}, 1.0, 10);
    EXPECT_TRUE(result.t.empty());
}

TEST(OdeAdvanced2, VerletVec_HarmonicOscillator2D_EnergyConserved) {
    const auto a = [](double, const std::vector<double>& q) {
        return std::vector<double>{-q[0], -q[1]};
    };
    const size_t periods = 20;
    const size_t steps_per_period = 100;
    const double t_end = static_cast<double>(periods) * 2.0 * M_PI;
    const size_t steps = periods * steps_per_period;
    const auto result = ode_verlet_vec(a, 0.0, {1.0, 0.5}, {0.0, 0.0}, t_end, steps);
    ASSERT_FALSE(result.q.empty());
    const double e0 = harmonic_energy_vec(result.q.front(), result.v.front());
    const double e1 = harmonic_energy_vec(result.q.back(), result.v.back());
    EXPECT_NEAR(e1, e0, 1e-3 * e0);
}

TEST(OdeAdvanced2, VerletVec_HarmonicOscillator2D_MatchesScalarComponents) {
    const auto a_scalar = [](double, double q) { return -q; };
    const auto a_vec = [](double, const std::vector<double>& q) {
        return std::vector<double>{-q[0], -q[1]};
    };
    const double t_end = 4.0 * M_PI;
    const size_t steps = 400;
    const auto scalar = ode_verlet(a_scalar, 0.0, 1.0, 0.0, t_end, steps);
    const auto vec = ode_verlet_vec(a_vec, 0.0, {1.0, 1.0}, {0.0, 0.0}, t_end, steps);
    ASSERT_FALSE(scalar.q.empty());
    ASSERT_FALSE(vec.q.empty());
    EXPECT_NEAR(vec.q.back()[0], scalar.q.back(), 0.02);
    EXPECT_NEAR(vec.q.back()[1], scalar.q.back(), 0.02);
}

TEST(OdeAdvanced2, VerletVec_HarmonicOscillator2D_MatchesAnalytic) {
    const auto a = [](double, const std::vector<double>& q) {
        return std::vector<double>{-q[0], -q[1]};
    };
    const auto result = ode_verlet_vec(a, 0.0, {1.0, 0.5}, {0.0, 0.0}, M_PI, 400);
    ASSERT_FALSE(result.q.empty());
    EXPECT_NEAR(result.q.back()[0], std::cos(M_PI), 0.01);
    EXPECT_NEAR(result.q.back()[1], 0.5 * std::cos(M_PI), 0.01);
    EXPECT_NEAR(result.v.back()[0], -std::sin(M_PI), 0.01);
    EXPECT_NEAR(result.v.back()[1], -0.5 * std::sin(M_PI), 0.01);
}

TEST(OdeAdvanced2, VerletVec_ProducesMonotoneGrid) {
    const auto a = [](double, const std::vector<double>& q) {
        return std::vector<double>{-q[0], -q[1]};
    };
    const auto result = ode_verlet_vec(a, 0.0, {1.0, 0.0}, {0.0, 1.0}, 1.0, 10);
    ASSERT_EQ(result.t.size(), 11u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// ode_rk45_vec
// ---------------------------------------------------------------------------

TEST(OdeAdvanced2, Rk45Vec_EmptyY0_ReturnsEmpty) {
    const auto f = [](double, const std::vector<double>&) {
        return std::vector<double>{};
    };
    const auto result = ode_rk45_vec(f, 0.0, {}, 1.0);
    EXPECT_TRUE(result.t.empty());
    EXPECT_TRUE(result.y.empty());
}

TEST(OdeAdvanced2, Rk45Vec_HarmonicOscillator2D_MatchesAnalytic) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto result = ode_rk45_vec(f, 0.0, {1.0, 0.0}, M_PI);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back()[0], std::cos(M_PI), 1e-4);
    EXPECT_NEAR(result.y.back()[1], -std::sin(M_PI), 1e-4);
}

TEST(OdeAdvanced2, Rk45Vec_HarmonicOscillator2D_MidTrajectory) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto result = ode_rk45_vec(f, 0.0, {1.0, 0.0}, M_PI / 2.0);
    ASSERT_GE(result.y.size(), 2u);
    const size_t mid = result.y.size() / 2;
    EXPECT_NEAR(result.y[mid][0], std::cos(result.t[mid]), 1e-3);
    EXPECT_NEAR(result.y[mid][1], -std::sin(result.t[mid]), 1e-3);
}

TEST(OdeAdvanced2, Rk45Vec_AdaptiveStepCount_Reasonable) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto result = ode_rk45_vec(f, 0.0, {1.0, 0.0}, 2.0 * M_PI);
    ASSERT_GE(result.t.size(), 2u);
    EXPECT_GT(result.t.size(), 5u);
    EXPECT_LT(result.t.size(), 5000u);
}

TEST(OdeAdvanced2, Rk45Vec_TighterTolerance_SmallerError) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const double t_end = 1.0;
    const double exact_y = std::cos(t_end);
    const double exact_yp = -std::sin(t_end);
    const auto loose = ode_rk45_vec(f, 0.0, {1.0, 0.0}, t_end, 1e-3, 1e-6);
    const auto tight = ode_rk45_vec(f, 0.0, {1.0, 0.0}, t_end, 1e-9, 1e-12);
    const double err_loose = std::abs(loose.y.back()[0] - exact_y)
                           + std::abs(loose.y.back()[1] - exact_yp);
    const double err_tight = std::abs(tight.y.back()[0] - exact_y)
                           + std::abs(tight.y.back()[1] - exact_yp);
    EXPECT_LT(err_tight, err_loose);
}

TEST(OdeAdvanced2, Rk45Vec_LooserTolerance_StillReasonable) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto result = ode_rk45_vec(f, 0.0, {1.0, 0.0}, 1.0, 1e-2, 1e-5);
    ASSERT_FALSE(result.y.empty());
    EXPECT_NEAR(result.y.back()[0], std::cos(1.0), 0.01);
    EXPECT_NEAR(result.y.back()[1], -std::sin(1.0), 0.01);
}

TEST(OdeAdvanced2, Rk45Vec_ProducesMonotoneGrid) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto result = ode_rk45_vec(f, 0.0, {1.0, 0.0}, 1.0);
    ASSERT_GE(result.t.size(), 2u);
    for (size_t i = 1; i < result.t.size(); ++i) {
        EXPECT_GT(result.t[i], result.t[i - 1]);
    }
}

TEST(OdeAdvanced2, Rk45Vec_SolutionRemainsFinite) {
    const auto f = [](double, const std::vector<double>& y) {
        return std::vector<double>{y[1], -y[0]};
    };
    const auto result = ode_rk45_vec(f, 0.0, {1.0, 0.0}, 10.0);
    for (const auto& state : result.y) {
        for (double v : state) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }
}
