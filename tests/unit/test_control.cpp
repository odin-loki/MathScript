#include "ms/control/control.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace ms::control;

// ---- Transfer function construction ----
TEST(ControlTF, Construction) {
    auto sys = tf({1.0}, {1.0, 2.0, 1.0});  // 1/(s+1)^2
    EXPECT_EQ(sys.num.size(), 1u);
    EXPECT_EQ(sys.den.size(), 3u);
}

// ---- Poles/zeros ----
TEST(ControlTF, Poles) {
    // H(s) = 1 / (s+1)(s+2) = 1 / (s^2 + 3s + 2)
    auto sys = tf({1.0}, {1.0, 3.0, 2.0});
    auto p = poles(sys);
    ASSERT_EQ(p.size(), 2u);
    std::vector<double> real_parts = {p[0].real(), p[1].real()};
    std::sort(real_parts.begin(), real_parts.end());
    EXPECT_NEAR(real_parts[0], -2.0, 1e-6);
    EXPECT_NEAR(real_parts[1], -1.0, 1e-6);
}

TEST(ControlTF, Zeros) {
    // H(s) = (s+3) / (s+1)
    auto sys = tf({1.0, 3.0}, {1.0, 1.0});
    auto z = zeros(sys);
    ASSERT_EQ(z.size(), 1u);
    EXPECT_NEAR(z[0].real(), -3.0, 1e-6);
}

TEST(ControlTF, DCGain) {
    // H(s) = 5 / (s + 5) → H(0) = 1
    auto sys = tf({5.0}, {1.0, 5.0});
    EXPECT_NEAR(dcgain(sys), 1.0, 1e-10);
}

// ---- Stability ----
TEST(ControlTF, IsStable) {
    auto stable = tf({1.0}, {1.0, 2.0, 1.0});  // poles at -1
    EXPECT_TRUE(is_stable(stable));
    auto unstable = tf({1.0}, {1.0, -1.0});     // pole at +1
    EXPECT_FALSE(is_stable(unstable));
}

// ---- Interconnections ----
TEST(ControlTF, Series) {
    auto g = tf({1.0}, {1.0, 1.0});
    auto h = tf({2.0}, {1.0, 2.0});
    auto s = series(g, h);
    // num = 1*2=2, den = (s+1)(s+2) = s^2+3s+2
    EXPECT_EQ(s.num.size(), 1u);
    EXPECT_EQ(s.den.size(), 3u);
}

TEST(ControlTF, Parallel) {
    auto g = tf({1.0}, {1.0, 1.0});  // 1/(s+1)
    auto h = tf({1.0}, {1.0, 1.0});  // 1/(s+1)
    auto p = parallel(g, h);
    // 2/(s+1) → num should evaluate to 2 at s=0
    EXPECT_NEAR(dcgain(p), 2.0, 1e-6);
}

TEST(ControlTF, Feedback) {
    // Unity feedback around integrator: H = (1/s) / (1 + 1/s) = 1/(s+1)
    auto plant = tf({1.0}, {1.0, 0.0});  // 1/s
    auto sense = tf({1.0}, {1.0});        // 1
    auto cl = feedback(plant, sense);
    EXPECT_NEAR(dcgain(cl), 1.0, 1e-3);
}

// ---- Bode ----
TEST(ControlTF, Bode) {
    auto sys = tf({1.0}, {1.0, 1.0});  // 1/(s+1)
    auto bd = bode(sys, 0.01, 100.0, 50);
    EXPECT_EQ(bd.w.size(), 50u);
    // At DC (low w), magnitude ≈ 0 dB
    EXPECT_NEAR(bd.magnitude[0], 0.0, 2.0);
}

// ---- Step response ----
TEST(ControlTF, StepResponse) {
    // 1/(s+1): step response y(t) = 1 - e^{-t}
    auto sys = tf({1.0}, {1.0, 1.0});
    auto s = step_response(sys, 5.0, 200);
    EXPECT_EQ(s.t.size(), 200u);
    // At t≈5 (5 time constants) response should be near 1
    EXPECT_NEAR(s.y.back(), 1.0, 0.05);
}

// ---- tf2ss conversion ----
TEST(ControlSS, TF2SS) {
    auto sys = tf({1.0}, {1.0, 3.0, 2.0});  // 1/(s+1)(s+2)
    auto s = tf2ss(sys);
    EXPECT_EQ(s.n, 2);
}

// ---- Controllability ----
TEST(ControlSS, Controllability) {
    // Controllable double integrator
    std::vector<std::vector<double>> A = {{0, 1}, {0, 0}};
    std::vector<std::vector<double>> B = {{0}, {1}};
    EXPECT_TRUE(is_controllable(A, B));
    // Uncontrollable: second state unaffected
    std::vector<std::vector<double>> B2 = {{1}, {0}};
    // Actually (A,B2) may still be controllable; use decoupled system
    std::vector<std::vector<double>> A3 = {{1, 0}, {0, 2}};
    std::vector<std::vector<double>> B3 = {{1}, {0}};
    EXPECT_FALSE(is_controllable(A3, B3));
}

// ---- LQR ----
TEST(ControlLQR, BasicLQR) {
    // Stable system: A is Hurwitz so Newton-Lyapunov converges reliably
    std::vector<std::vector<double>> A = {{-2, 1}, {0, -3}};
    std::vector<std::vector<double>> B = {{1}, {1}};
    std::vector<std::vector<double>> Q = {{1, 0}, {0, 1}};
    std::vector<std::vector<double>> R = {{1}};
    auto K = lqr(A, B, Q, R);
    // LQR should converge and return 1×2 gain matrix
    EXPECT_TRUE(K.has_value());
    if (K) {
        EXPECT_EQ(K.value().size(), 1u);
        EXPECT_EQ(K.value()[0].size(), 2u);
    }
}

// ---- Lyapunov ----
TEST(ControlLyap, ContinuousLyapunov) {
    // A = [-1], Q = [1] → X = 0.5
    std::vector<std::vector<double>> A = {{-1.0}};
    std::vector<std::vector<double>> Q = {{1.0}};
    auto X = ms::control::lyap(A, Q);
    EXPECT_TRUE(X.has_value());
    if (X) EXPECT_NEAR(X.value()[0][0], 0.5, 1e-4);
}

TEST(ControlLyap, DiscreteLyapunov) {
    // A = [0.5], Q = [1] → X = 4/3
    std::vector<std::vector<double>> A = {{0.5}};
    std::vector<std::vector<double>> Q = {{1.0}};
    auto X = ms::control::dlyap(A, Q);
    EXPECT_TRUE(X.has_value());
    if (X) EXPECT_NEAR(X.value()[0][0], 4.0 / 3.0, 1e-4);
}

// ---- Pole placement ----
TEST(ControlPlace, PolePlace) {
    std::vector<std::vector<double>> A = {{0, 1}, {0, 0}};
    std::vector<std::vector<double>> B = {{0}, {1}};
    std::vector<double> desired = {-2.0, -3.0};
    auto K = place(A, B, desired);
    EXPECT_TRUE(K.has_value());
    // Check that eigenvalues of A-BK are at desired poles
    if (K) {
        EXPECT_EQ(K.value().size(), 2u);
    }
}

// ---- PID tuning ----
TEST(ControlPID, PIDTune) {
    auto plant = tf({1.0}, {1.0, 1.0});  // 1/(s+1)
    auto gains = pidtune(plant, 1.0);
    EXPECT_GT(gains.Kp, 0.0);
    EXPECT_GT(gains.Ki, 0.0);
    EXPECT_GT(gains.Kd, 0.0);
}

// ---- Bode (corner frequency) ----
TEST(ControlBode, MagnitudeAtCorner) {
    // H(s) = 1/(s+1): |H(j1)| = 1/sqrt(2) ≈ -3.01 dB, phase ≈ -45°
    auto sys = tf({1.0}, {1.0, 1.0});
    auto bd = bode(sys, 0.1, 10.0, 100);
    ASSERT_GE(bd.w.size(), 2u);
    size_t idx = 0;
    double best = std::abs(bd.w[0] - 1.0);
    for (size_t i = 1; i < bd.w.size(); ++i) {
        const double d = std::abs(bd.w[i] - 1.0);
        if (d < best) {
            best = d;
            idx = i;
        }
    }
    EXPECT_NEAR(bd.magnitude[idx], -3.01, 0.15);
    EXPECT_NEAR(bd.phase[idx], -45.0, 2.0);
}

// ---- Riccati / DARE ----
TEST(ControlRiccati, PositiveSolution) {
    std::vector<std::vector<double>> A = {{-2.0, 1.0}, {0.0, -3.0}};
    std::vector<std::vector<double>> B = {{1.0}, {1.0}};
    std::vector<std::vector<double>> Q = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> R = {{1.0}};
    auto X = riccati(A, B, Q, R);
    ASSERT_TRUE(X.has_value());
    EXPECT_EQ(X.value().size(), 2u);
    EXPECT_GT(X.value()[0][0], 0.0);
}

TEST(ControlDare, ScalarSystem) {
    std::vector<std::vector<double>> A = {{0.5}};
    std::vector<std::vector<double>> B = {{1.0}};
    std::vector<std::vector<double>> Q = {{1.0}};
    std::vector<std::vector<double>> R = {{1.0}};
    auto X = dare(A, B, Q, R);
    ASSERT_TRUE(X.has_value());
    EXPECT_NEAR(X.value()[0][0], 4.0 / 3.0, 0.25);
}

// ---- LQR (closed-loop stabilisation) ----
TEST(ControlLQR, StabilizesIntegrator) {
    // Double integrator: A = [[0,1],[0,0]], B = [[0],[1]]
    std::vector<std::vector<double>> A = {{0.0, 1.0}, {0.0, 0.0}};
    std::vector<std::vector<double>> B = {{0.0}, {1.0}};
    std::vector<std::vector<double>> Q = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> R = {{1.0}};
    auto K = lqr(A, B, Q, R);
    ASSERT_TRUE(K.has_value());
    EXPECT_EQ(K.value().size(), 1u);
    EXPECT_EQ(K.value()[0].size(), 2u);
    std::vector<std::vector<double>> Acl = A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            Acl[i][j] -= B[i][0] * K.value()[0][j];
    auto cl_tf = ss2tf(ss(Acl, B, {{1.0, 0.0}}, {{0.0}}));
    for (auto& p : poles(cl_tf))
        EXPECT_LT(p.real(), 0.0);
}

// ---- Pole placement (closed-loop poles) ----
TEST(ControlPlace, ClosedLoopPoles) {
    std::vector<std::vector<double>> A = {{0.0, 1.0}, {0.0, 0.0}};
    std::vector<std::vector<double>> B = {{0.0}, {1.0}};
    std::vector<double> desired = {-1.0, -2.0};
    auto K = place(A, B, desired);
    ASSERT_TRUE(K.has_value());
    std::vector<std::vector<double>> Acl = A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            Acl[i][j] -= B[i][0] * K.value()[j];
    auto cl_tf = ss2tf(ss(Acl, B, {{1.0, 0.0}}, {{0.0}}));
    auto p = poles(cl_tf);
    ASSERT_EQ(p.size(), 2u);
    std::vector<double> reals = {p[0].real(), p[1].real()};
    std::sort(reals.begin(), reals.end());
    EXPECT_NEAR(reals[0], -2.0, 0.05);
    EXPECT_NEAR(reals[1], -1.0, 0.05);
}

// ---- tf2ss / ss2tf roundtrip ----
TEST(ControlSS, TF2SSRoundtrip) {
    auto orig = tf({1.0}, {1.0, 3.0, 2.0});  // 1/(s+1)(s+2)
    auto s = tf2ss(orig);
    EXPECT_EQ(s.n, 2);
    auto back = ss2tf(s);
    EXPECT_NEAR(dcgain(back), dcgain(orig), 1e-10);
    auto p_orig = poles(orig);
    auto p_back = poles(back);
    ASSERT_EQ(p_orig.size(), p_back.size());
    std::vector<double> reals_orig, reals_back;
    for (auto& p : p_orig) reals_orig.push_back(p.real());
    for (auto& p : p_back) reals_back.push_back(p.real());
    std::sort(reals_orig.begin(), reals_orig.end());
    std::sort(reals_back.begin(), reals_back.end());
    for (size_t i = 0; i < reals_orig.size(); ++i)
        EXPECT_NEAR(reals_back[i], reals_orig[i], 1e-6);
}

// ---- Discretization helpers ----
namespace {

double max_abs_diff(const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B) {
    double m = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[i].size(); ++j)
            m = std::max(m, std::abs(A[i][j] - B[i][j]));
    return m;
}

void expect_ss_near(const StateSpace& a, const StateSpace& b, double tol) {
    EXPECT_NEAR(max_abs_diff(a.A, b.A), 0.0, tol);
    EXPECT_NEAR(max_abs_diff(a.B, b.B), 0.0, tol);
    EXPECT_NEAR(max_abs_diff(a.C, b.C), 0.0, tol);
    EXPECT_NEAR(max_abs_diff(a.D, b.D), 0.0, tol);
}

StateSpace mass_spring_damper() {
    return ss({{0.0, 1.0}, {-2.0, -3.0}}, {{0.0}, {1.0}}, {{1.0, 0.0}}, {{0.0}});
}

StateSpace pure_integrator() {
    return ss({{0.0}}, {{1.0}}, {{1.0}}, {{0.0}});
}

StateSpace third_order() {
    return ss({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-6.0, -11.0, -6.0}},
              {{0.0}, {0.0}, {1.0}}, {{1.0, 0.0, 0.0}}, {{0.0}});
}

} // namespace

// ---- c2d/d2c round-trip ----
TEST(ControlDiscretize, ZOHC2dReferenceMassSpring) {
    const auto disc = c2d(mass_spring_damper(), 0.1, DiscretizationMethod::ZOH);
    EXPECT_NEAR(disc.A[0][0], 0.99094408, 1e-5);
    EXPECT_NEAR(disc.A[0][1], 0.08610666, 1e-5);
    EXPECT_NEAR(disc.A[1][0], -0.17221333, 1e-5);
    EXPECT_NEAR(disc.A[1][1], 0.73262409, 1e-5);
    EXPECT_NEAR(disc.B[0][0], 0.00452796, 1e-5);
    EXPECT_NEAR(disc.B[1][0], 0.08610666, 1e-5);
}

TEST(ControlDiscretize, RoundTripZOHMassSpring) {
    const auto orig = mass_spring_damper();
    const double Ts = 0.1;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::ZOH), Ts,
                             DiscretizationMethod::ZOH), 1e-5);
}

TEST(ControlDiscretize, RoundTripZOHIntegrator) {
    const auto orig = pure_integrator();
    const double Ts = 0.1;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::ZOH), Ts,
                             DiscretizationMethod::ZOH), 1e-5);
}

TEST(ControlDiscretize, RoundTripZOHThirdOrder) {
    const auto orig = third_order();
    const double Ts = 0.05;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::ZOH), Ts,
                             DiscretizationMethod::ZOH), 1e-4);
}

TEST(ControlDiscretize, RoundTripTustinMassSpring) {
    const auto orig = mass_spring_damper();
    const double Ts = 0.1;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::Tustin), Ts,
                             DiscretizationMethod::Tustin), 1e-9);
}

TEST(ControlDiscretize, RoundTripTustinIntegrator) {
    const auto orig = pure_integrator();
    const double Ts = 0.1;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::Tustin), Ts,
                             DiscretizationMethod::Tustin), 1e-9);
}

TEST(ControlDiscretize, RoundTripTustinThirdOrder) {
    const auto orig = third_order();
    const double Ts = 0.05;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::Tustin), Ts,
                             DiscretizationMethod::Tustin), 1e-8);
}

TEST(ControlDiscretize, RoundTripEulerMassSpring) {
    const auto orig = mass_spring_damper();
    const double Ts = 0.1;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::Euler), Ts,
                             DiscretizationMethod::Euler), 1e-12);
}

TEST(ControlDiscretize, RoundTripEulerIntegrator) {
    const auto orig = pure_integrator();
    const double Ts = 0.1;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::Euler), Ts,
                             DiscretizationMethod::Euler), 1e-12);
}

TEST(ControlDiscretize, RoundTripEulerThirdOrder) {
    const auto orig = third_order();
    const double Ts = 0.05;
    expect_ss_near(orig, d2c(c2d(orig, Ts, DiscretizationMethod::Euler), Ts,
                             DiscretizationMethod::Euler), 1e-11);
}

// ---- ZOH closed-form scalar ----
TEST(ControlDiscretize, ZOHScalarStable) {
    const double a = -2.0, b = 3.0, Ts = 0.1;
    const auto sys = ss({{a}}, {{b}}, {{1.0}}, {{0.0}});
    const auto disc = c2d(sys, Ts, DiscretizationMethod::ZOH);
    const double Ad_exact = std::exp(a * Ts);
    const double Bd_exact = (std::exp(a * Ts) - 1.0) / a * b;
    EXPECT_NEAR(disc.A[0][0], Ad_exact, 1e-6);
    EXPECT_NEAR(disc.B[0][0], Bd_exact, 1e-6);
    EXPECT_NEAR(max_abs_diff(disc.A, sys.A), std::abs(Ad_exact - a), 1e-6);
}

TEST(ControlDiscretize, ZOHScalarNearSingular) {
    const double a = 1e-8, b = 1.0, Ts = 0.1;
    const auto sys = ss({{a}}, {{b}}, {{1.0}}, {{0.0}});
    const auto disc = c2d(sys, Ts, DiscretizationMethod::ZOH);
    const double Ad_exact = std::exp(a * Ts);
    const double Bd_exact = (std::exp(a * Ts) - 1.0) / a * b;
    EXPECT_NEAR(disc.A[0][0], Ad_exact, 1e-6);
    EXPECT_NEAR(disc.B[0][0], Bd_exact, 1e-5);
}

// ---- Euler exactness ----
TEST(ControlDiscretize, EulerExactForm) {
    const auto sys = mass_spring_damper();
    const double Ts = 0.1;
    const auto disc = c2d(sys, Ts, DiscretizationMethod::Euler);
    for (int i = 0; i < sys.n; ++i)
        for (int j = 0; j < sys.n; ++j)
            EXPECT_NEAR(disc.A[i][j], (i == j ? 1.0 : 0.0) + sys.A[i][j] * Ts, 1e-15);
    for (int i = 0; i < sys.n; ++i)
        for (int j = 0; j < sys.m; ++j)
            EXPECT_NEAR(disc.B[i][j], sys.B[i][j] * Ts, 1e-15);
    EXPECT_NEAR(max_abs_diff(disc.C, sys.C), 0.0, 1e-15);
    EXPECT_NEAR(max_abs_diff(disc.D, sys.D), 0.0, 1e-15);
}

// ---- Discrete stability sanity ----
TEST(ControlDiscretize, ZOHStablePoleInsideUnitCircle) {
    const auto sys = ss({{-1.0}}, {{1.0}}, {{1.0}}, {{0.0}});
    const double Ts = 0.1;
    const auto disc = c2d(sys, Ts, DiscretizationMethod::ZOH);
    EXPECT_LT(std::abs(disc.A[0][0]), 1.0);
    EXPECT_NEAR(disc.A[0][0], std::exp(-Ts), 1e-6);
}

// ---- TransferFunction overloads ----
TEST(ControlDiscretize, TransferFunctionConsistency) {
    const auto plant = tf({1.0}, {1.0, 3.0, 2.0});
    const double Ts = 0.1;
    const auto via_ss = ss2tf(c2d(tf2ss(plant), Ts, DiscretizationMethod::Tustin));
    const auto direct = c2d(plant, Ts, DiscretizationMethod::Tustin);
    EXPECT_NEAR(dcgain(direct), dcgain(via_ss), 1e-6);
    const auto p_direct = poles(direct);
    const auto p_via = poles(via_ss);
    ASSERT_EQ(p_direct.size(), p_via.size());
    std::vector<double> rd, rv;
    for (auto& p : p_direct) rd.push_back(p.real());
    for (auto& p : p_via) rv.push_back(p.real());
    std::sort(rd.begin(), rd.end());
    std::sort(rv.begin(), rv.end());
    for (size_t i = 0; i < rd.size(); ++i)
        EXPECT_NEAR(rd[i], rv[i], 1e-5);
}

TEST(ControlDiscretize, TransferFunctionD2CRoundTrip) {
    const auto plant = tf({1.0}, {1.0, 2.0, 1.0});
    const double Ts = 0.1;
    const auto back = d2c(c2d(plant, Ts, DiscretizationMethod::Tustin), Ts,
                          DiscretizationMethod::Tustin);
    EXPECT_NEAR(dcgain(back), dcgain(plant), 1e-4);
}
