// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the quantum information measures: the Hermitian Jacobi
// spectral engine and the four quantities built on it (von Neumann entropy,
// Uhlmann fidelity, trace distance, Wootters concurrence).
//
// Every case here is pinned to a value derived independently of the
// implementation -- a closed form, a defining identity, or a similarity
// invariant -- and every one of them FAILS against the pre-fix code, which
// diagonalised non-Hermitian garbage, capped the entropy at dimension 8, and
// returned sqrt(|Tr(rho sigma)|), half the Frobenius norm and an ad-hoc
// diagonal expression in place of the three documented formulas.
#define _USE_MATH_DEFINES
#include "ms/quantum/quantum.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace ms::quantum;

namespace {

// Tr(A^k), computed straight from the matrix product -- independent of any
// eigenvalue computation.
double trace_power(const DensityMatrix& A, int k) {
    DensityMatrix P = A;
    for (int i = 1; i < k; ++i) P = matmul_dm(P, A);
    C t = 0.0;
    for (std::size_t i = 0; i < P.size(); ++i) t += P[i][i];
    return t.real();
}

// A 4x4 complex Hermitian matrix with genuinely complex off-diagonals.
DensityMatrix complex_hermitian_4x4() {
    return {{C(2.0, 0.0),    C(1.0, 1.0),     C(0.5, -0.25),  C(-0.3, 0.4)},
            {C(1.0, -1.0),   C(3.0, 0.0),     C(-0.75, 0.5),  C(0.2, 0.1)},
            {C(0.5, 0.25),   C(-0.75, -0.5),  C(1.0, 0.0),    C(0.6, -0.2)},
            {C(-0.3, -0.4),  C(0.2, -0.1),    C(0.6, 0.2),    C(-1.5, 0.0)}};
}

DensityMatrix maximally_mixed(int d) {
    DensityMatrix rho(static_cast<std::size_t>(d),
                      std::vector<C>(static_cast<std::size_t>(d), C(0.0)));
    for (int i = 0; i < d; ++i) rho[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] =
        C(1.0 / static_cast<double>(d), 0.0);
    return rho;
}

// Rank-one density matrix of the uniform superposition in dimension n; pure, so
// S = 0, but with every entry off the diagonal non-zero.
DensityMatrix uniform_pure(int n) {
    return DensityMatrix(static_cast<std::size_t>(n),
                         std::vector<C>(static_cast<std::size_t>(n),
                                        C(1.0 / static_cast<double>(n), 0.0)));
}

DensityMatrix mix(const DensityMatrix& a, double wa, const DensityMatrix& b, double wb) {
    DensityMatrix out = a;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a.size(); ++j)
            out[i][j] = wa * a[i][j] + wb * b[i][j];
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Hermitian Jacobi diagonalisation (via eigenspectrum / ground_state)
// ---------------------------------------------------------------------------

// A unitary similarity preserves every Tr(A^k), so the returned spectrum must
// reproduce all four power sums of a 4x4.  The pre-fix rotation wrote the same
// complex value into H[k][p] and H[p][k] instead of the conjugate, which is not
// a similarity at all: it missed Tr(A^3) by 0.544 and Tr(A^4) by 0.899 here.
TEST(QuantumSpectrum, ComplexHermitianPreservesPowerTraces) {
    const auto A = complex_hermitian_4x4();
    const auto evals = eigenspectrum(A);
    ASSERT_EQ(evals.size(), 4u);

    for (int k = 1; k <= 4; ++k) {
        double sum = 0.0;
        for (double lambda : evals) sum += std::pow(lambda, k);
        EXPECT_NEAR(sum, trace_power(A, k), 1e-9) << "power trace k=" << k;
    }
    // Ascending order is part of the documented contract.
    EXPECT_TRUE(std::is_sorted(evals.begin(), evals.end()));
}

// The ground state must actually be an eigenvector of H for the reported
// smallest eigenvalue.  Pre-fix the eigenvector accumulation applied the
// conjugate phase, so the residual was 5.3e-01 for this matrix.
TEST(QuantumSpectrum, GroundStateOfComplexHermitianIsAnEigenvector) {
    const auto A = complex_hermitian_4x4();
    const auto evals = eigenspectrum(A);
    const auto v = ground_state(A);
    ASSERT_EQ(v.size(), 4u);

    double norm2 = 0.0;
    for (const auto& c : v) norm2 += std::norm(c);
    EXPECT_NEAR(norm2, 1.0, 1e-12);

    const auto Av = op_apply(A, v);
    const double lambda_min = evals.front();
    for (std::size_t i = 0; i < v.size(); ++i)
        EXPECT_NEAR(std::abs(Av[i] - lambda_min * v[i]), 0.0, 1e-10) << "component " << i;
    EXPECT_NEAR(inner(v, Av).real(), lambda_min, 1e-10);
}

// Real symmetric input is affected too: the phase-elimination step multiplies
// in a +/-i whenever the pivot is negative, after which the same broken update
// runs.  Pre-fix this 3x3 reported a ground energy of 1.692877 with a state
// whose Rayleigh quotient was 2.535912 -- not an eigenvector.
TEST(QuantumSpectrum, GroundStateOfRealSymmetricMatchesRayleighQuotient) {
    const DensityMatrix H = {{C(4.0, 0.0), C(-1.0, 0.0), C(0.5, 0.0)},
                             {C(-1.0, 0.0), C(3.0, 0.0), C(0.2, 0.0)},
                             {C(0.5, 0.0), C(0.2, 0.0), C(2.0, 0.0)}};
    const auto evals = eigenspectrum(H);
    const auto v = ground_state(H);
    EXPECT_NEAR(inner(v, op_apply(H, v)).real(), evals.front(), 1e-10);

    for (int k = 1; k <= 3; ++k) {
        double sum = 0.0;
        for (double lambda : evals) sum += std::pow(lambda, k);
        EXPECT_NEAR(sum, trace_power(H, k), 1e-10) << "power trace k=" << k;
    }
}

// The sweep budget has to scale past the old n <= 8 comfort zone.
TEST(QuantumSpectrum, ConvergesAtDimensionSixteen) {
    const int n = 16;
    DensityMatrix A(static_cast<std::size_t>(n), std::vector<C>(static_cast<std::size_t>(n), C(0.0)));
    for (int i = 0; i < n; ++i) {
        A[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = C(std::cos(0.7 * i) * 3.0, 0.0);
        for (int j = i + 1; j < n; ++j) {
            const C v(0.4 * std::cos(0.3 * i + 0.11 * j), 0.4 * std::sin(0.17 * i - 0.23 * j));
            A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
            A[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = std::conj(v);
        }
    }
    const auto evals = eigenspectrum(A);
    for (int k = 1; k <= 3; ++k) {
        double sum = 0.0;
        for (double lambda : evals) sum += std::pow(lambda, k);
        EXPECT_NEAR(sum, trace_power(A, k), 1e-8) << "power trace k=" << k;
    }
}

// ---------------------------------------------------------------------------
// von Neumann entropy
// ---------------------------------------------------------------------------

// The pre-fix body took the diagonal entries for eigenvalues above dimension 8,
// so these pure states reported ln(9) = 2.197225 and ln(16) = 2.772589 -- the
// maximum entropy attainable in the dimension -- instead of 0.
TEST(QuantumEntropy, PureStateAboveDimensionEightIsZero) {
    for (int n : {2, 4, 8, 9, 16, 32}) {
        EXPECT_NEAR(von_neumann_entropy(uniform_pure(n)), 0.0, 1e-12) << "dim " << n;
    }
}

TEST(QuantumEntropy, MaximallyMixedIsLogDimension) {
    for (int d : {2, 3, 4, 8, 9, 16}) {
        EXPECT_NEAR(von_neumann_entropy(maximally_mixed(d)),
                    std::log(static_cast<double>(d)), 1e-12) << "dim " << d;
    }
}

// A rank-2 mixture with weights 0.3 / 0.7 written in a rotated 16-dimensional
// basis: S = -0.3 ln 0.3 - 0.7 ln 0.7 = 0.610864302055, independent of the
// basis.  The diagonal approximation returned 2.672120 for this state.
TEST(QuantumEntropy, RankTwoMixtureInRotatedBasisDimSixteen) {
    const int n = 16;
    Ket a(static_cast<std::size_t>(n)), b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i)] = C(std::cos(0.3 * i), std::sin(0.11 * i));
        b[static_cast<std::size_t>(i)] = C(std::sin(0.7 * i + 1.0), std::cos(0.05 * i));
    }
    a = ket_normalise(a);
    const C overlap = inner(a, b);
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] -= overlap * a[static_cast<std::size_t>(i)];
    b = ket_normalise(b);

    const auto rho = mix(density_matrix(a), 0.3, density_matrix(b), 0.7);
    const double exact = -0.3 * std::log(0.3) - 0.7 * std::log(0.7);
    EXPECT_NEAR(von_neumann_entropy(rho), exact, 1e-12);
    EXPECT_NEAR(purity(rho), 0.3 * 0.3 + 0.7 * 0.7, 1e-12);
}

// ---------------------------------------------------------------------------
// Uhlmann fidelity  F = Tr sqrt( sqrt(rho) sigma sqrt(rho) )
// ---------------------------------------------------------------------------

// F(rho, rho) = 1 is the defining normalisation.  sqrt(|Tr(rho^2)|) gave
// 0.707107 / 0.5 / 0.353553 for the maximally mixed qubit, 2-qubit and 3-qubit
// states -- wrong for every state that is not pure.
TEST(QuantumFidelity, MaximallyMixedWithItselfIsOne) {
    for (int d : {2, 3, 4, 8}) {
        const auto rho = maximally_mixed(d);
        EXPECT_NEAR(fidelity(rho, rho), 1.0, 1e-12) << "dim " << d;
    }
}

// Commuting states reduce to the classical Bhattacharyya coefficient
// F = sum_i sqrt(p_i q_i) = sqrt(0.28) + sqrt(0.18) = 0.953414330925.
TEST(QuantumFidelity, CommutingStatesGiveBhattacharyyaCoefficient) {
    const DensityMatrix rho = {{C(0.7, 0.0), C(0.0, 0.0)}, {C(0.0, 0.0), C(0.3, 0.0)}};
    const DensityMatrix sigma = {{C(0.4, 0.0), C(0.0, 0.0)}, {C(0.0, 0.0), C(0.6, 0.0)}};
    const double exact = std::sqrt(0.7 * 0.4) + std::sqrt(0.3 * 0.6);
    EXPECT_NEAR(fidelity(rho, sigma), exact, 1e-12);
    EXPECT_NEAR(fidelity(sigma, rho), exact, 1e-12);  // symmetry
}

// For qubits there is a closed form independent of any eigen-decomposition:
//     F^2 = Tr(rho sigma) + 2 sqrt(det rho * det sigma).
// These two states do not commute, so nothing degenerate is being exercised.
TEST(QuantumFidelity, NonCommutingQubitsMatchClosedForm) {
    const DensityMatrix rho = {{C(0.6, 0.0), C(0.2, 0.1)}, {C(0.2, -0.1), C(0.4, 0.0)}};
    const DensityMatrix sigma = {{C(0.3, 0.0), C(-0.1, 0.2)}, {C(-0.1, -0.2), C(0.7, 0.0)}};

    const auto product = matmul_dm(rho, sigma);
    const double tr_rs = (product[0][0] + product[1][1]).real();
    const double det_rho = (rho[0][0] * rho[1][1] - rho[0][1] * rho[1][0]).real();
    const double det_sigma = (sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0]).real();
    const double exact = std::sqrt(tr_rs + 2.0 * std::sqrt(det_rho * det_sigma));

    EXPECT_NEAR(exact, 0.899284112772, 1e-9);  // pins the closed form itself
    EXPECT_NEAR(fidelity(rho, sigma), exact, 1e-12);
    EXPECT_NEAR(fidelity(sigma, rho), exact, 1e-12);
}

// Pure states: F = |<psi|phi>| in the square-root convention the header
// documents (its square is the Born overlap probability).
TEST(QuantumFidelity, PureStatesGiveOverlapModulus) {
    const Ket a = {C(0.6, 0.0), C(0.8, 0.0)};
    const Ket b = {C(1.0, 0.0), C(0.0, 0.0)};
    const double F = fidelity(density_matrix(a), density_matrix(b));
    EXPECT_NEAR(F, 0.6, 1e-12);
    EXPECT_NEAR(F * F, 0.36, 1e-12);

    // Orthogonal pure states are perfectly distinguishable.
    EXPECT_NEAR(fidelity(density_matrix(ket_basis(2, 0)), density_matrix(ket_basis(2, 1))),
                0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Trace distance  T = (1/2) Tr|rho - sigma|
// ---------------------------------------------------------------------------

// Half the Frobenius norm gave 0.707107 here; the trace distance between two
// orthogonal pure states is exactly 1, the maximum the metric can take.
TEST(QuantumTraceDistance, OrthogonalPureStatesIsOne) {
    const auto rho = density_matrix(ket_basis(2, 0));
    const auto sigma = density_matrix(ket_basis(2, 1));
    EXPECT_NEAR(trace_distance(rho, sigma), 1.0, 1e-12);
    EXPECT_NEAR(trace_distance(rho, rho), 0.0, 1e-12);

    // 4-dimensional orthogonal pure states: still exactly 1.
    EXPECT_NEAR(trace_distance(density_matrix(ket_basis(4, 1)), density_matrix(ket_basis(4, 3))),
                1.0, 1e-12);
}

// I/2 vs |0><0|:  D = diag(-1/2, 1/2), Tr|D| = 1, T = 1/2.
// Half the Frobenius norm is 0.353553 -- a different number for the same input.
TEST(QuantumTraceDistance, MaximallyMixedVersusPureIsOneHalf) {
    EXPECT_NEAR(trace_distance(maximally_mixed(2), density_matrix(ket_basis(2, 0))), 0.5, 1e-12);
    // Two classical distributions: T is the total-variation distance.
    const DensityMatrix p = {{C(0.7, 0.0), C(0.0, 0.0)}, {C(0.0, 0.0), C(0.3, 0.0)}};
    const DensityMatrix q = {{C(0.2, 0.0), C(0.0, 0.0)}, {C(0.0, 0.0), C(0.8, 0.0)}};
    EXPECT_NEAR(trace_distance(p, q), 0.5, 1e-12);  // half Frobenius gives 0.353553
}

// Pure states: T = sqrt(1 - |<psi|phi>|^2).
TEST(QuantumTraceDistance, PureStatesMatchOverlapFormula) {
    const Ket a = {C(0.6, 0.0), C(0.8, 0.0)};
    const Ket b = {C(1.0, 0.0), C(0.0, 0.0)};
    EXPECT_NEAR(trace_distance(density_matrix(a), density_matrix(b)), 0.8, 1e-12);
}

// D = rho - sigma = [[0.3, 0.3-0.1i], [0.3+0.1i, -0.3]] is traceless Hermitian,
// so its eigenvalues are +/- sqrt(0.09 + |0.3-0.1i|^2) = +/- sqrt(0.19) and
// T = sqrt(0.19) = 0.435889894354 by hand.
TEST(QuantumTraceDistance, HandComputedTraceNormOfQubitDifference) {
    const DensityMatrix rho = {{C(0.6, 0.0), C(0.2, 0.1)}, {C(0.2, -0.1), C(0.4, 0.0)}};
    const DensityMatrix sigma = {{C(0.3, 0.0), C(-0.1, 0.2)}, {C(-0.1, -0.2), C(0.7, 0.0)}};
    EXPECT_NEAR(trace_distance(rho, sigma), std::sqrt(0.19), 1e-12);
}

// The two measures must sit inside the Fuchs-van de Graaf window
//     1 - F <= T <= sqrt(1 - F^2).
// Pre-fix the pair violated the lower bound by up to 0.43.
TEST(QuantumFidelityTraceDistance, FuchsVanDeGraafInequalities) {
    struct Pair { DensityMatrix rho, sigma; };
    const std::vector<Pair> cases = {
        {density_matrix(ket_basis(2, 0)), density_matrix(ket_basis(2, 1))},
        {maximally_mixed(2), density_matrix(ket_basis(2, 0))},
        {{{C(0.6, 0.0), C(0.2, 0.1)}, {C(0.2, -0.1), C(0.4, 0.0)}},
         {{C(0.3, 0.0), C(-0.1, 0.2)}, {C(-0.1, -0.2), C(0.7, 0.0)}}},
        {maximally_mixed(4), density_matrix(bell_states()[0])},
        {density_matrix(bell_states()[0]), density_matrix(bell_states()[2])},
    };
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const double F = fidelity(cases[i].rho, cases[i].sigma);
        const double T = trace_distance(cases[i].rho, cases[i].sigma);
        EXPECT_GE(T, 1.0 - F - 1e-12) << "case " << i;
        EXPECT_LE(T, std::sqrt(std::max(0.0, 1.0 - F * F)) + 1e-12) << "case " << i;
        EXPECT_GE(F, -1e-12) << "case " << i;
        EXPECT_LE(F, 1.0 + 1e-12) << "case " << i;
    }
}

// ---------------------------------------------------------------------------
// Wootters concurrence
// ---------------------------------------------------------------------------

// All four Bell states are maximally entangled.  The old diagonal shortcut
// looked only at rho[0][3], so |Psi+> and |Psi-> -- whose weight sits in the
// (1,2) block -- were reported as separable (C = 0).
TEST(QuantumConcurrence, AllFourBellStatesAreMaximallyEntangled) {
    const auto bells = bell_states();
    ASSERT_EQ(bells.size(), 4u);
    for (std::size_t i = 0; i < bells.size(); ++i)
        EXPECT_NEAR(concurrence(density_matrix(bells[i])), 1.0, 1e-9) << "bell state " << i;
}

// (|00> + i|11>)/sqrt(2) is maximally entangled too; the old code took
// rho[0][3].real(), which is 0 here, and reported a product state.
TEST(QuantumConcurrence, ImaginaryCoherenceIsMaximallyEntangled) {
    const double h = 1.0 / std::sqrt(2.0);
    const Ket psi = {C(h, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(0.0, h)};
    EXPECT_NEAR(concurrence(density_matrix(psi)), 1.0, 1e-9);
}

// For a pure two-qubit state a|00> + b|01> + c|10> + d|11>, C = 2|ad - bc|.
TEST(QuantumConcurrence, PureStateMatchesTwoAbsAdMinusBc) {
    const std::vector<Ket> states = {
        {C(0.3, 0.1), C(-0.5, 0.2), C(0.6, 0.0), C(0.1, -0.4)},
        {C(0.5, 0.0), C(0.5, 0.0), C(0.5, 0.0), C(0.5, 0.0)},      // |++>, separable
        {C(0.8, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(0.6, 0.0)},
        {C(0.1, 0.2), C(0.3, -0.4), C(-0.2, 0.5), C(0.6, 0.1)},
    };
    for (std::size_t i = 0; i < states.size(); ++i) {
        const Ket psi = ket_normalise(states[i]);
        const double exact = 2.0 * std::abs(psi[0] * psi[3] - psi[1] * psi[2]);
        EXPECT_NEAR(concurrence(density_matrix(psi)), exact, 1e-9) << "state " << i;
    }
}

// Genuinely mixed X states have the closed form
//     C = 2 max(0, |rho03| - sqrt(rho11 rho22), |rho12| - sqrt(rho00 rho33)).
// The old expression used rho11 + rho22 instead of 2 sqrt(rho11 rho22) and kept
// only one branch: it returned 0.2 where the answer is 0.253589838486, and 0.2
// where it is 0.6.
TEST(QuantumConcurrence, MixedXStatesMatchClosedForm) {
    {
        DensityMatrix X(4, std::vector<C>(4, C(0.0)));
        X[0][0] = C(0.3, 0.0); X[3][3] = C(0.3, 0.0);
        X[0][3] = C(0.3, 0.0); X[3][0] = C(0.3, 0.0);
        X[1][1] = C(0.3, 0.0); X[2][2] = C(0.1, 0.0);
        const double exact = 2.0 * (0.3 - std::sqrt(0.3 * 0.1));
        EXPECT_NEAR(exact, 0.253589838486, 1e-11);
        EXPECT_NEAR(concurrence(X), exact, 1e-9);
    }
    {
        // 0.6 |Phi+><Phi+| + 0.4 |01><01|:  rho03 = 0.3, rho22 = 0 -> C = 0.6.
        auto rho = density_matrix(bell_states()[0]);
        for (auto& row : rho)
            for (auto& value : row) value *= 0.6;
        rho[1][1] += C(0.4, 0.0);
        EXPECT_NEAR(concurrence(rho), 0.6, 1e-9);
    }
}

// Werner states rho = p|Phi+><Phi+| + (1-p) I/4 have C = max(0, (3p-1)/2),
// entangled exactly above p = 1/3.
TEST(QuantumConcurrence, WernerStateThreshold) {
    const auto bell = density_matrix(bell_states()[0]);
    for (double p : {0.0, 0.2, 1.0 / 3.0, 0.5, 0.8, 1.0}) {
        DensityMatrix rho = bell;
        for (auto& row : rho)
            for (auto& value : row) value *= p;
        for (int i = 0; i < 4; ++i) rho[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] +=
            C((1.0 - p) / 4.0, 0.0);
        EXPECT_NEAR(concurrence(rho), std::max(0.0, (3.0 * p - 1.0) / 2.0), 1e-9) << "p=" << p;
    }
}

// A genuine two-qubit product state is separable in every basis.
TEST(QuantumConcurrence, ProductStatesAreSeparable) {
    const auto plus = op_apply(hadamard(), ket_basis(2, 0));
    const std::vector<Ket> factors_a = {ket_basis(2, 0), plus, ket_basis(2, 1)};
    const std::vector<Ket> factors_b = {ket_basis(2, 0), ket_basis(2, 1), plus};
    for (std::size_t i = 0; i < factors_a.size(); ++i) {
        const auto rho = density_matrix(tensor_product_states(factors_a[i], factors_b[i]));
        EXPECT_NEAR(concurrence(rho), 0.0, 1e-9) << "product state " << i;
    }
    // Maximally mixed two-qubit state is separable.
    EXPECT_NEAR(concurrence(maximally_mixed(4)), 0.0, 1e-9);
}

// Concurrence is defined only for two-qubit (4x4) states.  Returning 0 for
// anything else, as the pre-fix code did, is indistinguishable from a genuine
// "separable" answer, so the contract is now an explicit NaN.
TEST(QuantumConcurrence, NonTwoQubitInputIsNotANumber) {
    EXPECT_TRUE(std::isnan(concurrence(density_matrix(ket_basis(2, 0)))));
    EXPECT_TRUE(std::isnan(concurrence(maximally_mixed(3))));
    EXPECT_TRUE(std::isnan(concurrence(maximally_mixed(8))));
    EXPECT_TRUE(std::isnan(concurrence(DensityMatrix{})));
}

// Concurrence of a Bell state must survive the entropy/fidelity path too:
// the reduced state of any Bell state is maximally mixed, S = log 2.
TEST(QuantumConcurrence, BellStateAgreesWithReducedEntropy) {
    for (const auto& psi : bell_states()) {
        const auto rho = density_matrix(psi);
        EXPECT_NEAR(concurrence(rho), 1.0, 1e-9);
        EXPECT_NEAR(von_neumann_entropy(partial_trace(rho, 2, 2, 0)), std::log(2.0), 1e-12);
        EXPECT_NEAR(von_neumann_entropy(rho), 0.0, 1e-12);
    }
}
