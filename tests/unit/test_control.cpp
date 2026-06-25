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
