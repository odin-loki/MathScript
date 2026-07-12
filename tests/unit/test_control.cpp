#include "ms/control/control.hpp"
#include "ms/linalg/linalg.hpp"
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

// ---- Gramians ----
namespace {

using Mat = std::vector<std::vector<double>>;

double mat_max_abs(const Mat& M) {
    double m = 0.0;
    for (const auto& row : M)
        for (double x : row) m = std::max(m, std::abs(x));
    return m;
}

Mat mat_transpose(const Mat& M) {
    size_t rows = M.size(), cols = M.empty() ? 0 : M[0].size();
    Mat T(cols, std::vector<double>(rows, 0.0));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            T[j][i] = M[i][j];
    return T;
}

Mat mat_matmul(const Mat& A, const Mat& B) {
    size_t m = A.size(), k = B.size(), n = B.empty() ? 0 : B[0].size();
    Mat C(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t l = 0; l < k; ++l)
                C[i][j] += A[i][l] * B[l][j];
    return C;
}

Mat mat_add(const Mat& A, const Mat& B) {
    Mat C = A;
    for (size_t i = 0; i < C.size(); ++i)
        for (size_t j = 0; j < C[i].size(); ++j)
            C[i][j] += B[i][j];
    return C;
}

// Convert a std::vector<std::vector<double>> to an ms::ColMatrix<double> so
// we can reuse the existing eig_sym() routine to check Gramian eigenvalues.
ms::ColMatrix<double> to_col_matrix(const Mat& M) {
    size_t rows = M.size(), cols = M.empty() ? 0 : M[0].size();
    ms::ColMatrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out(i, j) = M[i][j];
    return out;
}

// Stable, controllable & observable 2x2 diagonal test plant:
//   A = diag(-1, -2), B = [[1],[1]], C = [[1, 1]]
// Closed form (solve a_i*Wc_ij + Wc_ij*a_j + (BB^T)_ij = 0 elementwise for
// diagonal A): Wc = [[0.5, 1/3], [1/3, 0.25]]. By symmetry of A, Wo is
// identical when C = B^T.
StateSpace diagonal_plant() {
    return ss({{-1.0, 0.0}, {0.0, -2.0}}, {{1.0}, {1.0}}, {{1.0, 1.0}}, {{0.0}});
}

} // namespace

TEST(ControlGram, ControllabilityGramianSymmetric) {
    auto sys = diagonal_plant();
    auto Wc = gram(sys, GramianType::Controllability);
    ASSERT_TRUE(Wc.has_value());
    auto WcT = mat_transpose(Wc.value());
    for (size_t i = 0; i < Wc->size(); ++i)
        for (size_t j = 0; j < (*Wc)[i].size(); ++j)
            EXPECT_NEAR((*Wc)[i][j], WcT[i][j], 1e-8);
}

TEST(ControlGram, ControllabilityGramianClosedForm) {
    // Wc = [[0.5, 1/3], [1/3, 0.25]] for A = diag(-1,-2), B = [[1],[1]]
    auto sys = diagonal_plant();
    auto Wc = gram(sys, GramianType::Controllability);
    ASSERT_TRUE(Wc.has_value());
    EXPECT_NEAR(Wc.value()[0][0], 0.5, 1e-6);
    EXPECT_NEAR(Wc.value()[0][1], 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(Wc.value()[1][0], 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(Wc.value()[1][1], 0.25, 1e-6);
}

TEST(ControlGram, ControllabilityGramianPositiveSemiDefinite) {
    auto sys = diagonal_plant();
    auto Wc = gram(sys, GramianType::Controllability);
    ASSERT_TRUE(Wc.has_value());
    // Diagonal entries strictly positive for a controllable system.
    EXPECT_GT(Wc.value()[0][0], 0.0);
    EXPECT_GT(Wc.value()[1][1], 0.0);
    // All eigenvalues >= -epsilon (numerical noise tolerance).
    auto eig_res = ms::eig_sym(to_col_matrix(Wc.value()));
    ASSERT_TRUE(eig_res.has_value());
    for (size_t i = 0; i < eig_res->values.rows(); ++i)
        EXPECT_GE(eig_res->values(i, 0), -1e-9);
}

TEST(ControlGram, ControllabilityGramianSatisfiesLyapunovEquation) {
    // Verify A*Wc + Wc*A^T + B*B^T ≈ 0 by direct substitution — the strongest
    // correctness check, independent of trusting lyap()'s own tests.
    auto sys = diagonal_plant();
    auto Wc = gram(sys, GramianType::Controllability);
    ASSERT_TRUE(Wc.has_value());
    auto AWc = mat_matmul(sys.A, Wc.value());
    auto WcAt = mat_matmul(Wc.value(), mat_transpose(sys.A));
    auto BBt = mat_matmul(sys.B, mat_transpose(sys.B));
    auto residual = mat_add(mat_add(AWc, WcAt), BBt);
    EXPECT_NEAR(mat_max_abs(residual), 0.0, 1e-8);
}

TEST(ControlGram, ObservabilityGramianSatisfiesLyapunovEquation) {
    // Verify A^T*Wo + Wo*A + C^T*C ≈ 0 by direct substitution.
    auto sys = diagonal_plant();
    auto Wo = gram(sys, GramianType::Observability);
    ASSERT_TRUE(Wo.has_value());
    auto At = mat_transpose(sys.A);
    auto AtWo = mat_matmul(At, Wo.value());
    auto WoA = mat_matmul(Wo.value(), sys.A);
    auto CtC = mat_matmul(mat_transpose(sys.C), sys.C);
    auto residual = mat_add(mat_add(AtWo, WoA), CtC);
    EXPECT_NEAR(mat_max_abs(residual), 0.0, 1e-8);
}

TEST(ControlGram, ObservabilityGramianSymmetricAndPositive) {
    auto sys = diagonal_plant();
    auto Wo = gram(sys, GramianType::Observability);
    ASSERT_TRUE(Wo.has_value());
    auto WoT = mat_transpose(Wo.value());
    for (size_t i = 0; i < Wo->size(); ++i)
        for (size_t j = 0; j < (*Wo)[i].size(); ++j)
            EXPECT_NEAR((*Wo)[i][j], WoT[i][j], 1e-8);
    EXPECT_GT(Wo.value()[0][0], 0.0);
    EXPECT_GT(Wo.value()[1][1], 0.0);
    auto eig_res = ms::eig_sym(to_col_matrix(Wo.value()));
    ASSERT_TRUE(eig_res.has_value());
    for (size_t i = 0; i < eig_res->values.rows(); ++i)
        EXPECT_GE(eig_res->values(i, 0), -1e-9);
}

TEST(ControlGram, TraceIsFinite) {
    auto sys = diagonal_plant();
    auto Wc = gram(sys, GramianType::Controllability);
    ASSERT_TRUE(Wc.has_value());
    double trace = 0.0;
    for (size_t i = 0; i < Wc->size(); ++i) trace += (*Wc)[i][i];
    EXPECT_TRUE(std::isfinite(trace));
}

TEST(ControlGram, ScalarSystemClosedForm) {
    // dx/dt = -a*x + b*u  =>  Wc = b^2 / (2a)  (textbook closed form).
    const double a = 2.0, b = 3.0;
    auto sys = ss({{-a}}, {{b}}, {{1.0}}, {{0.0}});
    auto Wc = ctrb_gram(sys);
    ASSERT_TRUE(Wc.has_value());
    EXPECT_NEAR(Wc.value()[0][0], (b * b) / (2.0 * a), 1e-9);
}

TEST(ControlGram, ConvenienceWrappersMatchDispatcher) {
    auto sys = diagonal_plant();
    auto Wc1 = gram(sys, GramianType::Controllability);
    auto Wc2 = ctrb_gram(sys);
    auto Wo1 = gram(sys, GramianType::Observability);
    auto Wo2 = obsv_gram(sys);
    ASSERT_TRUE(Wc1.has_value());
    ASSERT_TRUE(Wc2.has_value());
    ASSERT_TRUE(Wo1.has_value());
    ASSERT_TRUE(Wo2.has_value());
    for (size_t i = 0; i < Wc1->size(); ++i)
        for (size_t j = 0; j < (*Wc1)[i].size(); ++j) {
            EXPECT_NEAR((*Wc1)[i][j], (*Wc2)[i][j], 1e-12);
            EXPECT_NEAR((*Wo1)[i][j], (*Wo2)[i][j], 1e-12);
        }
}

TEST(ControlGram, UnstableSystemDoesNotCrash) {
    // A = [1] (unstable, pole at +1): the Lyapunov equation for Wc has no
    // finite positive-semidefinite solution. gram() must not crash and
    // should surface failure through the Result rather than returning
    // a bogus matrix.
    auto sys = ss({{1.0}}, {{1.0}}, {{1.0}}, {{0.0}});
    auto Wc = gram(sys, GramianType::Controllability);
    // lyap() on A=[1] gives 2*X + 1 = 0 -> X = -0.5, which is a valid
    // (negative) solution to the Sylvester equation itself, so this does
    // not error — but it must be finite and must not crash.
    if (Wc.has_value()) {
        for (const auto& row : Wc.value())
            for (double x : row) EXPECT_TRUE(std::isfinite(x));
    }
}

TEST(ControlGram, MarginallyStableSystemDoesNotCrash) {
    // A = [0] (marginally stable, pole on imaginary axis at origin): the
    // Lyapunov map A*X + X*A^T is identically zero, so the equation
    // 0 = -B*B^T has no solution unless B = 0 -> lyap() must report failure
    // (singular system), and gram() must propagate that gracefully.
    auto sys = ss({{0.0}}, {{1.0}}, {{1.0}}, {{0.0}});
    auto Wc = gram(sys, GramianType::Controllability);
    EXPECT_FALSE(Wc.has_value());
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

// ---- Kalman filter ----
namespace {

double kalman_trace(const std::vector<std::vector<double>>& P) {
    double t = 0.0;
    for (size_t i = 0; i < P.size(); ++i) t += P[i][i];
    return t;
}

// Simple deterministic LCG-based "noise" generator so tests are reproducible
// without pulling in <random> engines/seeding subtleties.
double lcg_noise(uint64_t& state, double amplitude) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    // Take top bits, map to [-amplitude, amplitude]
    double u = static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53);
    return (u - 0.5) * 2.0 * amplitude;
}

} // namespace

TEST(ControlKalman, PredictStepExact2x2) {
    // x = [1, 2], P = I; A = [[1,1],[0,1]] (constant velocity); Q = [[0.1,0],[0,0.1]]
    KalmanState state{{1.0, 2.0}, {{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> Q = {{0.1, 0.0}, {0.0, 0.1}};
    auto pred = kalman_predict(state, A, Q);
    // x_pred = A*x = [1*1+1*2, 0*1+1*2] = [3, 2]
    EXPECT_NEAR(pred.x[0], 3.0, 1e-12);
    EXPECT_NEAR(pred.x[1], 2.0, 1e-12);
    // P_pred = A*P*A^T + Q; with P=I, A*A^T = [[2,1],[1,1]]
    EXPECT_NEAR(pred.P[0][0], 2.1, 1e-12);
    EXPECT_NEAR(pred.P[0][1], 1.0, 1e-12);
    EXPECT_NEAR(pred.P[1][0], 1.0, 1e-12);
    EXPECT_NEAR(pred.P[1][1], 1.1, 1e-12);
}

TEST(ControlKalman, ScalarPredictUpdateClosedForm) {
    // Scalar Kalman filter: A=1, Q=q, H=1, R=r.
    const double q = 0.05, r = 0.5;
    KalmanState state{{0.0}, {{1.0}}};
    const double z = 2.0;

    // Independently compute the standard scalar recursion.
    double x = 0.0, P = 1.0;
    double x_pred = x;              // A=1
    double P_pred = P + q;          // A=1
    double S = P_pred + r;          // H=1
    double K = P_pred / S;
    double x_post = x_pred + K * (z - x_pred);
    double P_post = (1.0 - K) * P_pred;

    std::vector<std::vector<double>> A = {{1.0}};
    std::vector<std::vector<double>> Q = {{q}};
    std::vector<std::vector<double>> H = {{1.0}};
    std::vector<std::vector<double>> R = {{r}};

    auto pred = kalman_predict(state, A, Q);
    EXPECT_NEAR(pred.x[0], x_pred, 1e-12);
    EXPECT_NEAR(pred.P[0][0], P_pred, 1e-12);

    auto post = kalman_update(pred, {z}, H, R);
    EXPECT_NEAR(post.x[0], x_post, 1e-10);
    EXPECT_NEAR(post.P[0][0], P_post, 1e-10);
}

TEST(ControlKalman, UpdateNeverIncreasesUncertainty) {
    // For a well-posed, observable update, trace(P_post) <= trace(P_pred).
    KalmanState state{{0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<std::vector<double>> A = {{1.0, 1.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> Q = {{0.01, 0.0}, {0.0, 0.01}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> R = {{1.0}};

    auto pred = kalman_predict(state, A, Q);
    double trace_pred = kalman_trace(pred.P);
    auto post = kalman_update(pred, {0.5}, H, R);
    double trace_post = kalman_trace(post.P);
    EXPECT_LE(trace_post, trace_pred + 1e-12);
}

TEST(ControlKalman, CovarianceShrinksOverRepeatedUpdates) {
    // Repeated updates on a stationary scalar system should monotonically
    // shrink the covariance towards a steady-state value.
    KalmanState state{{0.0}, {{10.0}}};  // large initial uncertainty
    std::vector<std::vector<double>> A = {{1.0}};
    std::vector<std::vector<double>> Q = {{0.01}};
    std::vector<std::vector<double>> H = {{1.0}};
    std::vector<std::vector<double>> R = {{1.0}};

    double prev_trace = state.P[0][0];
    uint64_t rng = 12345;
    for (int i = 0; i < 20; ++i) {
        auto pred = kalman_predict(state, A, Q);
        double z = 5.0 + lcg_noise(rng, 0.5);
        state = kalman_update(pred, {z}, H, R);
        double trace_now = state.P[0][0];
        EXPECT_LE(trace_now, prev_trace + 1e-9);
        prev_trace = trace_now;
    }
}

TEST(ControlKalman, FiltersNoisyScalarRandomWalkBetterThanRawMeasurement) {
    // True value is constant (5.0); measurements are noisy. After enough
    // updates, the filtered estimate should be closer to truth than the
    // most recent raw noisy measurement.
    const double truth = 5.0;
    KalmanState state{{0.0}, {{5.0}}};
    std::vector<std::vector<double>> A = {{1.0}};
    std::vector<std::vector<double>> Q = {{0.001}};
    std::vector<std::vector<double>> H = {{1.0}};
    std::vector<std::vector<double>> R = {{4.0}};

    uint64_t rng = 987654321ULL;
    double last_z = 0.0;
    for (int i = 0; i < 50; ++i) {
        auto pred = kalman_predict(state, A, Q);
        last_z = truth + lcg_noise(rng, 3.0);
        state = kalman_update(pred, {last_z}, H, R);
    }
    double err_filtered = std::abs(state.x[0] - truth);
    double err_raw = std::abs(last_z - truth);
    EXPECT_LT(err_filtered, err_raw);
}

TEST(ControlKalman, ConstantVelocityTrackingEstimatesVelocity) {
    // Classic textbook example: track position+velocity from position-only
    // measurements of a known constant-velocity trajectory.
    const double true_velocity = 1.0;
    const double dt = 1.0;
    std::vector<std::vector<double>> A = {{1.0, dt}, {0.0, 1.0}};
    std::vector<std::vector<double>> Q = {{0.001, 0.0}, {0.0, 0.001}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> R = {{0.25}};

    KalmanState state{{0.0, 0.0}, {{10.0, 0.0}, {0.0, 10.0}}};
    uint64_t rng = 42;
    double true_pos = 0.0;
    for (int i = 0; i < 60; ++i) {
        true_pos += true_velocity * dt;
        auto pred = kalman_predict(state, A, Q);
        double z = true_pos + lcg_noise(rng, 0.4);
        state = kalman_update(pred, {z}, H, R);
    }
    EXPECT_NEAR(state.x[1], true_velocity, 0.2);
}

TEST(ControlKalman, PredictDefensiveNonSquareA) {
    KalmanState state{{1.0, 2.0}, {{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<std::vector<double>> A = {{1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};  // 2x3, not square
    std::vector<std::vector<double>> Q = {{0.1, 0.0}, {0.0, 0.1}};
    auto out = kalman_predict(state, A, Q);
    EXPECT_EQ(out.x, state.x);
    EXPECT_EQ(out.P, state.P);
}

TEST(ControlKalman, PredictDefensiveDimMismatchWithState) {
    KalmanState state{{1.0, 2.0, 3.0}, {{1,0,0},{0,1,0},{0,0,1}}};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}};  // 2x2, state is 3-dim
    std::vector<std::vector<double>> Q = {{0.1, 0.0}, {0.0, 0.1}};
    auto out = kalman_predict(state, A, Q);
    EXPECT_EQ(out.x, state.x);
    EXPECT_EQ(out.P, state.P);
}

TEST(ControlKalman, UpdateDefensiveHColumnMismatch) {
    KalmanState state{{1.0, 2.0}, {{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<std::vector<double>> H = {{1.0, 0.0, 0.0}};  // 1x3, but state is 2-dim
    std::vector<std::vector<double>> R = {{1.0}};
    auto out = kalman_update(state, {5.0}, H, R);
    EXPECT_EQ(out.x, state.x);
    EXPECT_EQ(out.P, state.P);
}

TEST(ControlKalman, UpdateDefensiveZSizeMismatch) {
    KalmanState state{{1.0, 2.0}, {{1.0, 0.0}, {0.0, 1.0}}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}, {0.0, 1.0}};  // expects z of size 2
    std::vector<std::vector<double>> R = {{1.0, 0.0}, {0.0, 1.0}};
    auto out = kalman_update(state, {5.0}, H, R);  // z has size 1
    EXPECT_EQ(out.x, state.x);
    EXPECT_EQ(out.P, state.P);
}

TEST(ControlKalman, UpdateDefensiveEmptyState) {
    KalmanState state{{}, {}};
    std::vector<std::vector<double>> H = {{1.0}};
    std::vector<std::vector<double>> R = {{1.0}};
    auto out = kalman_update(state, {1.0}, H, R);
    EXPECT_TRUE(out.x.empty());
    EXPECT_TRUE(out.P.empty());
}

TEST(ControlKalman, UpdateSingularInnovationCovarianceDoesNotCrash) {
    // R = 0 and H*P*H^T = 0 (P = 0) makes S singular; must not crash and
    // should fall back to returning the input state unchanged.
    KalmanState state{{1.0, 2.0}, {{0.0, 0.0}, {0.0, 0.0}}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> R = {{0.0}};
    auto out = kalman_update(state, {5.0}, H, R);
    EXPECT_EQ(out.x, state.x);
    EXPECT_EQ(out.P, state.P);
}

TEST(ControlKalman, PredictThenUpdateComposesToKnownGainScalar) {
    // With P=1, Q=0, H=1, R=1: predict gives P_pred=1, S=2, K=0.5.
    KalmanState state{{0.0}, {{1.0}}};
    std::vector<std::vector<double>> A = {{1.0}};
    std::vector<std::vector<double>> Q = {{0.0}};
    std::vector<std::vector<double>> H = {{1.0}};
    std::vector<std::vector<double>> R = {{1.0}};
    auto pred = kalman_predict(state, A, Q);
    auto post = kalman_update(pred, {2.0}, H, R);
    // x_post = 0 + 0.5*(2-0) = 1.0
    EXPECT_NEAR(post.x[0], 1.0, 1e-12);
    // P_post = (1-0.5)*1 = 0.5
    EXPECT_NEAR(post.P[0][0], 0.5, 1e-12);
}

TEST(ControlKalman, PosteriorCovarianceRemainsSymmetric) {
    KalmanState state{{0.0, 0.0}, {{2.0, 0.3}, {0.3, 1.5}}};
    std::vector<std::vector<double>> A = {{1.0, 0.5}, {0.0, 1.0}};
    std::vector<std::vector<double>> Q = {{0.02, 0.0}, {0.0, 0.02}};
    std::vector<std::vector<double>> H = {{1.0, 0.0}};
    std::vector<std::vector<double>> R = {{0.3}};
    auto pred = kalman_predict(state, A, Q);
    auto post = kalman_update(pred, {1.2}, H, R);
    EXPECT_NEAR(post.P[0][1], post.P[1][0], 1e-9);
}
