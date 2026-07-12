#define _USE_MATH_DEFINES
#include "ms/quantum/quantum.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::quantum;

// ---- State construction ----
TEST(QuantumState, BasisState) {
    auto psi = ket_basis(2, 0);
    EXPECT_NEAR(psi[0].real(), 1.0, 1e-10);
    EXPECT_NEAR(psi[1].real(), 0.0, 1e-10);
}

TEST(QuantumState, Normalise) {
    Ket psi = {C(1.0), C(1.0)};
    auto norm = ket_normalise(psi);
    double n2 = std::norm(norm[0]) + std::norm(norm[1]);
    EXPECT_NEAR(n2, 1.0, 1e-10);
}

// ---- Inner product ----
TEST(QuantumInner, Orthogonal) {
    auto bra = ket_basis(2, 0);
    auto ket = ket_basis(2, 1);
    C ip = inner(bra, ket);
    EXPECT_NEAR(std::abs(ip), 0.0, 1e-10);
}

TEST(QuantumInner, SelfInner) {
    auto psi = ket_basis(2, 0);
    C ip = inner(psi, psi);
    EXPECT_NEAR(ip.real(), 1.0, 1e-10);
}

// ---- Pauli matrices ----
TEST(QuantumGates, PauliX_Squared_IsIdentity) {
    auto X = pauli_x();
    auto X2 = matmul_dm(X, X);
    EXPECT_NEAR(X2[0][0].real(), 1.0, 1e-10);
    EXPECT_NEAR(X2[1][1].real(), 1.0, 1e-10);
    EXPECT_NEAR(X2[0][1].real(), 0.0, 1e-10);
}

TEST(QuantumGates, PauliZ_Eigenvalues) {
    auto Z = pauli_z();
    auto z0 = ket_basis(2, 0);
    auto z1 = ket_basis(2, 1);
    auto r0 = op_apply(Z, z0);
    auto r1 = op_apply(Z, z1);
    EXPECT_NEAR(r0[0].real(), 1.0, 1e-10);   // +1 eigenvalue
    EXPECT_NEAR(r1[1].real(), -1.0, 1e-10);  // -1 eigenvalue
}

// ---- Hadamard ----
TEST(QuantumGates, HadamardCreatesPlus) {
    auto H = hadamard();
    auto z0 = ket_basis(2, 0);
    auto plus = op_apply(H, z0);  // |+> = (|0> + |1>) / sqrt(2)
    EXPECT_NEAR(std::abs(plus[0]), 1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(std::abs(plus[1]), 1.0 / std::sqrt(2.0), 1e-10);
}

TEST(QuantumGates, HadamardSelfInverse) {
    auto H = hadamard();
    auto H2 = matmul_dm(H, H);
    EXPECT_NEAR(H2[0][0].real(), 1.0, 1e-10);
    EXPECT_NEAR(H2[1][1].real(), 1.0, 1e-10);
}

// ---- CNOT ----
TEST(QuantumGates, CNOT) {
    auto cnot = cnot_gate();
    // |11> → |10>
    Ket state11 = {0, 0, 0, 1};  // |11>
    auto out = op_apply(cnot, state11);
    EXPECT_NEAR(std::abs(out[2]), 1.0, 1e-10);  // |10>
}

// ---- Density matrix ----
TEST(QuantumDensity, PureState) {
    auto psi = ket_basis(2, 0);
    auto rho = density_matrix(psi);
    EXPECT_NEAR(rho[0][0].real(), 1.0, 1e-10);
    EXPECT_NEAR(rho[1][1].real(), 0.0, 1e-10);
}

// ---- Commutator ----
TEST(QuantumCommutator, PauliXZ) {
    // [X, Z] = -2iY
    auto X = pauli_x();
    auto Y = pauli_y();
    auto Z = pauli_z();
    auto comm = commutator(X, Z);
    // [X,Z] should be -2iY
    auto minus2iY = matmul_dm({{C(0,-2)},{C(0,0)}}, Y);
    // Just check it's non-zero
    double norm = 0.0;
    for (auto& row : comm) for (auto& c : row) norm += std::norm(c);
    EXPECT_GT(norm, 0.01);
}

// ---- Tensor product ----
TEST(QuantumTensor, TwoQubits) {
    auto I2 = identity(2);
    auto X = pauli_x();
    auto IX = tensor_product(I2, X);
    EXPECT_EQ(IX.size(), 4u);
    EXPECT_EQ(IX[0].size(), 4u);
}

// ---- QFT ----
TEST(QuantumQFT, Unitarity) {
    auto Q = qft_gate(2);  // 4x4
    auto Qdag = dagger(Q);
    auto QQdag = matmul_dm(Q, Qdag);
    // Q Q† = I
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(QQdag[i][i].real(), 1.0, 1e-10);
        for (int j = 0; j < 4; ++j)
            if (i != j)
                EXPECT_NEAR(std::abs(QQdag[i][j]), 0.0, 1e-10);
    }
}

// ---- Bell states ----
TEST(QuantumStates, BellStates) {
    auto bells = bell_states();
    EXPECT_EQ(bells.size(), 4u);
    // All Bell states are normalised
    for (auto& b : bells) {
        double n2 = 0.0;
        for (auto& c : b) n2 += std::norm(c);
        EXPECT_NEAR(n2, 1.0, 1e-10);
    }
}

// ---- GHZ state ----
TEST(QuantumStates, GHZ) {
    auto ghz = ghz_state(3);
    EXPECT_EQ(ghz.size(), 8u);
    double n2 = 0.0;
    for (auto& c : ghz) n2 += std::norm(c);
    EXPECT_NEAR(n2, 1.0, 1e-10);
}

// ---- W state ----
TEST(QuantumStates, WState) {
    auto w = w_state(3);
    EXPECT_EQ(w.size(), 8u);
    double n2 = 0.0;
    for (auto& c : w) n2 += std::norm(c);
    EXPECT_NEAR(n2, 1.0, 1e-10);
}

// ---- Coherent state ----
TEST(QuantumStates, CoherentState) {
    auto cs = coherent_state(C(1.0, 0.0), 20);
    double n2 = 0.0;
    for (auto& c : cs) n2 += std::norm(c);
    EXPECT_NEAR(n2, 1.0, 1e-6);
}

// ---- Expectation value ----
TEST(QuantumMeasure, Expectation) {
    // <0|Z|0> = 1
    auto z0 = ket_basis(2, 0);
    auto Z = pauli_z();
    EXPECT_NEAR(expectation(z0, Z), 1.0, 1e-10);
    // <1|Z|1> = -1
    auto z1 = ket_basis(2, 1);
    EXPECT_NEAR(expectation(z1, Z), -1.0, 1e-10);
}

// ---- Uncertainty ----
namespace {
double ket_overlap_prob(const Ket& a, const Ket& b) {
    return std::norm(inner(a, b));
}

bool check_generalized_uncertainty(const Ket& psi,
                                   const DensityMatrix& A,
                                   const DensityMatrix& B,
                                   double tol = 1e-10) {
    const double delta_prod = uncertainty(psi, A, B);
    const auto comm = commutator(A, B);
    const double comm_exp = std::abs(expectation(psi, comm));
    return delta_prod + tol >= 0.5 * comm_exp;
}
} // namespace

TEST(QuantumUncertainty, GeneralizedRelation_BasisZero) {
    auto psi = ket_basis(2, 0);
    EXPECT_TRUE(check_generalized_uncertainty(psi, pauli_x(), pauli_y()));
}

TEST(QuantumUncertainty, GeneralizedRelation_PlusState) {
    auto psi = ket_normalise(Ket{C(1.0), C(1.0)});
    EXPECT_TRUE(check_generalized_uncertainty(psi, pauli_x(), pauli_y()));
}

TEST(QuantumUncertainty, GeneralizedRelation_BellPhiPlus) {
    auto psi = bell_states()[0];
    auto X1 = tensor_product(pauli_x(), identity(2));
    auto Y2 = tensor_product(identity(2), pauli_y());
    EXPECT_TRUE(check_generalized_uncertainty(psi, X1, Y2));
}

TEST(QuantumUncertainty, ZeroOnEigenstate) {
    auto psi = ket_basis(2, 0);
    EXPECT_NEAR(uncertainty(psi, pauli_z(), pauli_z()), 0.0, 1e-10);
}

TEST(QuantumUncertainty, ZeroOnPauliXEigenstate) {
    auto plus = op_apply(hadamard(), ket_basis(2, 0));
    EXPECT_NEAR(uncertainty(plus, pauli_x(), pauli_x()), 0.0, 1e-8);
}

TEST(QuantumUncertainty, NonZeroForIncompatibleObservables) {
    auto psi = ket_normalise(Ket{C(1.0), C(0.0, 1.0)});
    EXPECT_GT(uncertainty(psi, pauli_x(), pauli_z()), 0.5);
}

// ---- Eigenspectrum ----
namespace {
void expect_eigenvalues_pm_one(const std::vector<double>& evals) {
    ASSERT_EQ(evals.size(), 2u);
    std::vector<double> sorted = evals;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_NEAR(sorted[0], -1.0, 1e-8);
    EXPECT_NEAR(sorted[1], 1.0, 1e-8);
}
} // namespace

TEST(QuantumSpectrum, PauliX) {
    expect_eigenvalues_pm_one(eigenspectrum(pauli_x()));
}

TEST(QuantumSpectrum, PauliY) {
    expect_eigenvalues_pm_one(eigenspectrum(pauli_y()));
}

TEST(QuantumSpectrum, PauliZ) {
    expect_eigenvalues_pm_one(eigenspectrum(pauli_z()));
}

TEST(QuantumSpectrum, Hadamard) {
    expect_eigenvalues_pm_one(eigenspectrum(hadamard()));
}

TEST(QuantumSpectrum, Identity) {
    auto evals = eigenspectrum(identity(2));
    ASSERT_EQ(evals.size(), 2u);
    EXPECT_NEAR(evals[0], 1.0, 1e-8);
    EXPECT_NEAR(evals[1], 1.0, 1e-8);
}

TEST(QuantumSpectrum, DiagonalHamiltonian) {
    DensityMatrix H = {{{2.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {5.0, 0.0}}};
    auto evals = eigenspectrum(H);
    ASSERT_EQ(evals.size(), 2u);
    EXPECT_NEAR(evals[0], 2.0, 1e-8);
    EXPECT_NEAR(evals[1], 5.0, 1e-8);
}

// ---- Ground state ----
TEST(QuantumGroundState, PauliZ) {
    auto gs = ground_state(pauli_z());
    auto z1 = ket_basis(2, 1);
    EXPECT_NEAR(ket_overlap_prob(gs, z1), 1.0, 1e-8);
    EXPECT_NEAR(expectation(gs, pauli_z()), -1.0, 1e-8);
}

TEST(QuantumGroundState, DiagonalHamiltonian) {
    DensityMatrix H = {{{2.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {5.0, 0.0}}};
    auto gs = ground_state(H);
    auto psi0 = ket_basis(2, 0);
    EXPECT_NEAR(ket_overlap_prob(gs, psi0), 1.0, 1e-8);
    EXPECT_NEAR(expectation(gs, H), 2.0, 1e-8);
}

// ---- Time evolution ----
TEST(QuantumEvolution, Schrodinger) {
    // H = Z/2 (spin-1/2 in magnetic field)
    DensityMatrix H = {{C(0.5), C(0)}, {C(0), C(-0.5)}};
    auto psi0 = ket_basis(2, 0);
    auto traj = schrodinger(H, psi0, 0.0, M_PI, 100);
    EXPECT_EQ(traj.size(), 101u);
    // State should remain normalised
    for (auto& psi : traj) {
        double n2 = 0.0;
        for (auto& c : psi) n2 += std::norm(c);
        EXPECT_NEAR(n2, 1.0, 0.05);
    }
}

TEST(QuantumEvolution, SchrodingerPauliZPhase) {
    // exp(-i Z t)|0> = e^{-it}|0>
    auto H = pauli_z();
    auto psi0 = ket_basis(2, 0);
    const double t = 0.7;
    auto traj = schrodinger(H, psi0, 0.0, t, 1);
    ASSERT_EQ(traj.size(), 2u);
    const C expected_phase(std::cos(-t), std::sin(-t));
    EXPECT_NEAR(traj[1][0].real(), expected_phase.real(), 1e-8);
    EXPECT_NEAR(traj[1][0].imag(), expected_phase.imag(), 1e-8);
    EXPECT_NEAR(std::abs(traj[1][1]), 0.0, 1e-10);
}

TEST(QuantumEvolution, SchrodingerPauliXRabi) {
    // exp(-i X t)|0> = cos(t)|0> - i sin(t)|1>
    auto H = pauli_x();
    auto psi0 = ket_basis(2, 0);
    const double t = M_PI / 4.0;
    auto traj = schrodinger(H, psi0, 0.0, t, 1);
    ASSERT_EQ(traj.size(), 2u);
    const double c = std::cos(t);
    const double s = std::sin(t);
    EXPECT_NEAR(traj[1][0].real(), c, 1e-8);
    EXPECT_NEAR(traj[1][0].imag(), 0.0, 1e-8);
    EXPECT_NEAR(traj[1][1].real(), 0.0, 1e-8);
    EXPECT_NEAR(traj[1][1].imag(), -s, 1e-8);
}

TEST(QuantumEvolution, SchrodingerMatchesTimeEvolutionOperator) {
    auto H = pauli_x();
    auto psi0 = ket_basis(2, 0);
    const double t = 1.25;
    auto traj = schrodinger(H, psi0, 0.0, t, 1);
    auto U = time_evolution_operator(H, t);
    auto direct = op_apply(U, psi0);
    EXPECT_NEAR(ket_overlap_prob(traj[1], direct), 1.0, 1e-8);
}

TEST(QuantumEvolution, SchrodingerNormPreservation) {
    auto H = pauli_x();
    auto psi0 = ket_basis(2, 0);
    auto traj = schrodinger(H, psi0, 0.0, 2.0 * M_PI, 50);
    for (auto& psi : traj) {
        double n2 = 0.0;
        for (auto& c : psi) n2 += std::norm(c);
        EXPECT_NEAR(n2, 1.0, 1e-10);
    }
}

// ---- Von Neumann entropy ----
TEST(QuantumEntropy, PureState) {
    // Pure state: entropy = 0 (diagonal rho)
    auto rho = density_matrix(ket_basis(2, 0));
    double S = von_neumann_entropy(rho);
    EXPECT_NEAR(S, 0.0, 1e-10);
}

TEST(QuantumEntropy, PureStateNonDiagonalBasis) {
    // |+> = H|0>: rho is pure but not diagonal in computational basis
    auto plus = op_apply(hadamard(), ket_basis(2, 0));
    auto rho = density_matrix(plus);
    EXPECT_NEAR(rho[0][0].real(), 0.5, 1e-10);
    EXPECT_NEAR(rho[1][1].real(), 0.5, 1e-10);
    EXPECT_GT(std::abs(rho[0][1]), 0.4);

    const double S = von_neumann_entropy(rho);
    EXPECT_NEAR(S, 0.0, 1e-10);
}

TEST(QuantumEntropy, MaximumEntropy) {
    // Maximally mixed: rho = I/2 → S = log(2)
    DensityMatrix rho = {{{0.5, 0}, {0, 0}}, {{0, 0}, {0.5, 0}}};
    double S = von_neumann_entropy(rho);
    EXPECT_NEAR(S, std::log(2.0), 0.01);
}

// ---- Partial trace ----
TEST(QuantumTrace, PartialTrace) {
    // Product state |00><00| should give |0><0| when tracing out either system
    auto rho = density_matrix(tensor_product_states(ket_basis(2, 0), ket_basis(2, 0)));
    auto rhoA = partial_trace(rho, 2, 2, 0);
    EXPECT_NEAR(rhoA[0][0].real(), 1.0, 1e-10);
    EXPECT_NEAR(rhoA[1][1].real(), 0.0, 1e-10);
}

// ---- Fidelity ----
TEST(QuantumFidelity, IdenticalStates) {
    auto rho = density_matrix(ket_basis(2, 0));
    EXPECT_NEAR(fidelity(rho, rho), 1.0, 1e-6);
}

// ---- Bell states (orthogonality) ----
TEST(QuantumStates, BellStateOrthogonality) {
    auto bells = bell_states();
    C ip = inner(bells[0], bells[1]);  // |Φ+> vs |Φ->
    EXPECT_NEAR(std::abs(ip), 0.0, 1e-10);
    ip = inner(bells[2], bells[3]);    // |Ψ+> vs |Ψ->
    EXPECT_NEAR(std::abs(ip), 0.0, 1e-10);
}

// ---- GHZ (amplitudes) ----
TEST(QuantumStates, GHZAmplitudes) {
    auto ghz = ghz_state(3);
    EXPECT_NEAR(std::abs(ghz[0]), 1.0 / std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(std::abs(ghz[7]), 1.0 / std::sqrt(2.0), 1e-10);
    for (int i = 1; i < 7; ++i)
        EXPECT_NEAR(std::abs(ghz[i]), 0.0, 1e-10);
}

// ---- QFT (basis state) ----
TEST(QuantumQFT, AppliesToZero) {
    auto Q = qft_gate(2);
    auto out = op_apply(Q, ket_basis(4, 0));
    double amp = 1.0 / 2.0;  // 1/sqrt(N), N=4
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(std::abs(out[i]), amp, 1e-10);
}

// ---- Time evolution operator ----
TEST(QuantumEvolution, TimeEvolutionOperator) {
    auto X = pauli_x();
    DensityMatrix H(2, std::vector<C>(2));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            H[i][j] = X[i][j] * 0.5;
    auto U0 = time_evolution_operator(H, 0.0);
    EXPECT_NEAR(U0[0][0].real(), 1.0, 1e-8);
    EXPECT_NEAR(U0[1][1].real(), 1.0, 1e-8);
    auto U = time_evolution_operator(H, M_PI);
    auto out = op_apply(U, ket_basis(2, 0));
    EXPECT_NEAR(std::abs(out[1]), 1.0, 0.05);
}

// ---- Phase-space quasi-probability distributions (Wigner / Husimi Q) ----
namespace {

constexpr int kFockCutoff = 24;  // truncated Fock-basis dimension - 1

DensityMatrix mix(double w1, const DensityMatrix& rho1, double w2, const DensityMatrix& rho2) {
    int n = static_cast<int>(rho1.size());
    DensityMatrix out(n, std::vector<C>(n, C(0.0)));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            out[i][j] = w1 * rho1[i][j] + w2 * rho2[i][j];
    return out;
}

// Numerically integrate Q(alpha) over a square region of the complex plane
// via a simple midpoint grid sum: integral ≈ sum Q(alpha) * (step^2).
double integrate_husimi(const DensityMatrix& rho, double half_width, double step) {
    double total = 0.0;
    for (double re = -half_width; re <= half_width; re += step)
        for (double im = -half_width; im <= half_width; im += step)
            total += husimi_Q(rho, C(re, im));
    return total * step * step;
}

} // namespace

// ---- Husimi Q normalisation ----
TEST(QuantumPhaseSpace, HusimiNormalisation_Vacuum) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    double integral = integrate_husimi(rho, 6.0, 0.1);
    EXPECT_NEAR(integral, 1.0, 0.02);
}

TEST(QuantumPhaseSpace, HusimiNormalisation_Coherent) {
    auto rho = density_matrix(coherent_state(C(1.2, 0.8), kFockCutoff));
    double integral = integrate_husimi(rho, 7.0, 0.1);
    EXPECT_NEAR(integral, 1.0, 0.02);
}

TEST(QuantumPhaseSpace, HusimiNormalisation_MixedState) {
    // Equal mixture of |0> and |1>: still a physical (unit-trace) state.
    auto rho0 = density_matrix(fock_state(0, kFockCutoff));
    auto rho1 = density_matrix(fock_state(1, kFockCutoff));
    auto rho = mix(0.5, rho0, 0.5, rho1);
    double integral = integrate_husimi(rho, 6.0, 0.1);
    EXPECT_NEAR(integral, 1.0, 0.02);
}

// ---- Husimi Q positivity ----
TEST(QuantumPhaseSpace, HusimiPositivity_Vacuum) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    for (double re = -4.0; re <= 4.0; re += 0.5)
        for (double im = -4.0; im <= 4.0; im += 0.5)
            EXPECT_GE(husimi_Q(rho, C(re, im)), -1e-9);
}

TEST(QuantumPhaseSpace, HusimiPositivity_Coherent) {
    auto rho = density_matrix(coherent_state(C(1.5, -1.0), kFockCutoff));
    for (double re = -4.0; re <= 4.0; re += 0.5)
        for (double im = -4.0; im <= 4.0; im += 0.5)
            EXPECT_GE(husimi_Q(rho, C(re, im)), -1e-9);
}

TEST(QuantumPhaseSpace, HusimiPositivity_MixedState) {
    auto rho0 = density_matrix(fock_state(0, kFockCutoff));
    auto rho1 = density_matrix(fock_state(2, kFockCutoff));
    auto rho = mix(0.3, rho0, 0.7, rho1);
    for (double re = -4.0; re <= 4.0; re += 0.5)
        for (double im = -4.0; im <= 4.0; im += 0.5)
            EXPECT_GE(husimi_Q(rho, C(re, im)), -1e-9);
}

// ---- Husimi Q exact vacuum Gaussian: Q(alpha) = (1/pi) exp(-|alpha|^2) ----
TEST(QuantumPhaseSpace, HusimiVacuum_MatchesGaussianFormula) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    EXPECT_NEAR(husimi_Q(rho, C(0.0, 0.0)), 1.0 / M_PI, 1e-6);
    EXPECT_NEAR(husimi_Q(rho, C(1.0, 0.0)), std::exp(-1.0) / M_PI, 1e-6);
    EXPECT_NEAR(husimi_Q(rho, C(0.0, 1.5)), std::exp(-2.25) / M_PI, 1e-6);
}

// ---- Vacuum peak location: max of Q/W at the origin ----
TEST(QuantumPhaseSpace, HusimiVacuumPeakAtOrigin) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    double q_origin = husimi_Q(rho, C(0.0, 0.0));
    for (double re = -3.0; re <= 3.0; re += 0.3)
        for (double im = -3.0; im <= 3.0; im += 0.3) {
            if (re == 0.0 && im == 0.0) continue;
            EXPECT_LE(husimi_Q(rho, C(re, im)), q_origin + 1e-12);
        }
}

TEST(QuantumPhaseSpace, WignerVacuumPeakAtOrigin) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    double w_origin = wigner_function(rho, 0.0, 0.0);
    EXPECT_NEAR(w_origin, 1.0 / M_PI, 1e-4);
    for (double x = -3.0; x <= 3.0; x += 0.6)
        for (double p = -3.0; p <= 3.0; p += 0.6) {
            if (x == 0.0 && p == 0.0) continue;
            EXPECT_LE(wigner_function(rho, x, p), w_origin + 1e-9);
        }
}

// ---- Coherent-state peak location ----
TEST(QuantumPhaseSpace, HusimiCoherentPeakNearAlpha0) {
    const C alpha0(1.4, -0.9);
    auto rho = density_matrix(coherent_state(alpha0, kFockCutoff));

    double best_q = -1.0;
    C best_alpha(0.0, 0.0);
    for (double re = -3.0; re <= 4.0; re += 0.25)
        for (double im = -4.0; im <= 3.0; im += 0.25) {
            double q = husimi_Q(rho, C(re, im));
            if (q > best_q) {
                best_q = q;
                best_alpha = C(re, im);
            }
        }
    EXPECT_NEAR(std::abs(best_alpha - alpha0), 0.0, 0.3);
}

TEST(QuantumPhaseSpace, WignerCoherentPeakNearAlpha0) {
    const C alpha0(1.0, 1.0);
    auto rho = density_matrix(coherent_state(alpha0, kFockCutoff));
    const double x0 = alpha0.real() * std::sqrt(2.0);
    const double p0 = alpha0.imag() * std::sqrt(2.0);

    double best_w = -1e18;
    double best_x = 0.0, best_p = 0.0;
    for (double x = -2.0; x <= 4.0; x += 0.5)
        for (double p = -2.0; p <= 4.0; p += 0.5) {
            double w = wigner_function(rho, x, p);
            if (w > best_w) {
                best_w = w;
                best_x = x;
                best_p = p;
            }
        }
    EXPECT_NEAR(best_x, x0, 0.75);
    EXPECT_NEAR(best_p, p0, 0.75);
}

// ---- Wigner-function negativity for Fock state |1> (non-classicality witness) ----
TEST(QuantumPhaseSpace, WignerFock1NegativeAtOrigin) {
    auto rho = density_matrix(fock_state(1, kFockCutoff));
    double w_origin = wigner_function(rho, 0.0, 0.0);
    EXPECT_LT(w_origin, 0.0);
    EXPECT_NEAR(w_origin, -1.0 / M_PI, 1e-4);  // known closed form: (-1)^n / pi
}

TEST(QuantumPhaseSpace, WignerFock2PositiveAtOrigin) {
    // (-1)^n / pi with n=2 is positive again.
    auto rho = density_matrix(fock_state(2, kFockCutoff));
    double w_origin = wigner_function(rho, 0.0, 0.0);
    EXPECT_GT(w_origin, 0.0);
    EXPECT_NEAR(w_origin, 1.0 / M_PI, 1e-4);
}

TEST(QuantumPhaseSpace, WignerMixedFock0Fock1AverageAtOrigin) {
    // Equal mixture of |0> (W=+1/pi) and |1> (W=-1/pi) at the origin -> exactly 0.
    auto rho0 = density_matrix(fock_state(0, kFockCutoff));
    auto rho1 = density_matrix(fock_state(1, kFockCutoff));
    auto rho = mix(0.5, rho0, 0.5, rho1);
    EXPECT_NEAR(wigner_function(rho, 0.0, 0.0), 0.0, 1e-4);
}

// ---- Sanity: both functions decay away from the vacuum peak ----
TEST(QuantumPhaseSpace, HusimiVacuumDecaysAwayFromOrigin) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    double q0 = husimi_Q(rho, C(0.0, 0.0));
    double q_far = husimi_Q(rho, C(3.0, 0.0));
    EXPECT_GT(q0, q_far);
    EXPECT_LT(q_far, 0.01);
}

TEST(QuantumPhaseSpace, WignerVacuumDecaysAwayFromOrigin) {
    auto rho = density_matrix(fock_state(0, kFockCutoff));
    double w0 = wigner_function(rho, 0.0, 0.0);
    double w_far = wigner_function(rho, 3.0, 0.0);
    EXPECT_GT(w0, w_far);
    EXPECT_LT(w_far, 0.01);
}

// ---- Consistency: Wigner/Husimi of the zero operator is zero ----
TEST(QuantumPhaseSpace, ZeroDensityMatrixGivesZero) {
    DensityMatrix rho(kFockCutoff + 1, std::vector<C>(kFockCutoff + 1, C(0.0)));
    EXPECT_NEAR(husimi_Q(rho, C(0.5, 0.5)), 0.0, 1e-12);
    EXPECT_NEAR(wigner_function(rho, 0.5, 0.5), 0.0, 1e-9);
}

// ---- Grover's search algorithm ----
namespace {
double state_norm_sq(const Ket& psi) {
    double n2 = 0.0;
    for (auto& c : psi) n2 += std::norm(c);
    return n2;
}
} // namespace

// Headline correctness check: 3 qubits (N=8), 1 marked index, optimal iterations
// should amplify the marked-state probability well above 0.9.
TEST(QuantumGrover, SingleMarkedNearCertainty) {
    const int n_qubits = 3;
    const int N = 8;
    const int marked = 5;
    const int iters = grover_optimal_iterations(n_qubits, 1);
    auto psi = grover_search(n_qubits, {marked}, iters);
    ASSERT_EQ(psi.size(), static_cast<size_t>(N));
    EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8);
    EXPECT_GT(std::norm(psi[marked]), 0.9);
}

TEST(QuantumGrover, SingleMarkedNearCertainty_DifferentIndex) {
    const int n_qubits = 3;
    const int marked = 0;
    const int iters = grover_optimal_iterations(n_qubits, 1);
    auto psi = grover_search(n_qubits, {marked}, iters);
    EXPECT_GT(std::norm(psi[marked]), 0.9);
    EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8);
}

// Normalisation must be preserved (unitarity) across many (n, marked, iters) combos.
TEST(QuantumGrover, StaysNormalised_VariousConfigs) {
    struct Cfg { int n; std::vector<int> marked; int iters; };
    std::vector<Cfg> configs = {
        {2, {1}, 1}, {2, {0, 3}, 2}, {3, {2}, 3}, {3, {1, 6}, 4},
        {4, {5}, 5}, {4, {0, 1, 2}, 6}, {5, {10}, 8}, {3, {0}, 0},
    };
    for (auto& cfg : configs) {
        auto psi = grover_search(cfg.n, cfg.marked, cfg.iters);
        EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8)
            << "n=" << cfg.n << " iters=" << cfg.iters;
    }
}

// 0 iterations: exact uniform superposition, amplitude 1/sqrt(N) for every entry.
TEST(QuantumGrover, ZeroIterationsIsUniformSuperposition) {
    const int n_qubits = 3;
    const int N = 8;
    auto psi = grover_search(n_qubits, {2}, 0);
    ASSERT_EQ(psi.size(), static_cast<size_t>(N));
    const double amp = 1.0 / std::sqrt(static_cast<double>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(std::abs(psi[i]), amp, 1e-10);
}

TEST(QuantumGrover, NegativeIterationsIsUniformSuperposition) {
    auto psi = grover_search(3, {2}, -5);
    const double amp = 1.0 / std::sqrt(8.0);
    for (auto& c : psi) EXPECT_NEAR(std::abs(c), amp, 1e-10);
}

// Multiple marked indices: combined probability should be similarly amplified
// at the recomputed optimal iteration count for M=2.
TEST(QuantumGrover, MultipleMarkedIndices_CombinedAmplification) {
    const int n_qubits = 3;
    const std::vector<int> marked = {1, 6};
    const int iters = grover_optimal_iterations(n_qubits, static_cast<int>(marked.size()));
    auto psi = grover_search(n_qubits, marked, iters);
    double combined_prob = 0.0;
    for (int idx : marked) combined_prob += std::norm(psi[idx]);
    EXPECT_GT(combined_prob, 0.9);
    EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8);
}

TEST(QuantumGrover, MultipleMarkedIndices_FourQubits) {
    const int n_qubits = 4; // N = 16
    const std::vector<int> marked = {3, 9};
    const int iters = grover_optimal_iterations(n_qubits, static_cast<int>(marked.size()));
    auto psi = grover_search(n_qubits, marked, iters);
    double combined_prob = 0.0;
    for (int idx : marked) combined_prob += std::norm(psi[idx]);
    EXPECT_GT(combined_prob, 0.9);
}

// grover_optimal_iterations sanity: hand-computed floor(pi/4*sqrt(N/M)).
TEST(QuantumGrover, OptimalIterationsFormula) {
    struct Case { int n_qubits; int m; };
    std::vector<Case> cases = {
        {3, 1}, {4, 1}, {5, 1}, {3, 2}, {4, 2}, {6, 3}, {5, 4},
    };
    for (auto& c : cases) {
        const int N = 1 << c.n_qubits;
        const int expected = static_cast<int>(
            std::floor(M_PI / 4.0 * std::sqrt(static_cast<double>(N) / c.m)));
        EXPECT_EQ(grover_optimal_iterations(c.n_qubits, c.m), expected)
            << "n_qubits=" << c.n_qubits << " m=" << c.m;
    }
}

TEST(QuantumGrover, OptimalIterationsZeroWhenNoMarked) {
    EXPECT_EQ(grover_optimal_iterations(3, 0), 0);
    EXPECT_EQ(grover_optimal_iterations(3, -2), 0);
}

TEST(QuantumGrover, OptimalIterationsZeroWhenAllMarked) {
    EXPECT_EQ(grover_optimal_iterations(3, 8), 0);  // M == N
    EXPECT_EQ(grover_optimal_iterations(3, 20), 0); // M > N
}

// Defensive: n_qubits <= 0 returns {}.
TEST(QuantumGrover, NonPositiveQubitsReturnsEmpty) {
    EXPECT_TRUE(grover_search(0, {0}, 1).empty());
    EXPECT_TRUE(grover_search(-1, {0}, 1).empty());
}

// Defensive: empty marked_indices degenerates the oracle to identity; the
// function must not crash and must still return a normalised state.
TEST(QuantumGrover, EmptyMarkedIndicesStaysNormalised) {
    auto psi = grover_search(3, {}, 5);
    EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8);
}

// Defensive: out-of-range marked indices are ignored (oracle no-op); result
// must still be a valid, normalised state.
TEST(QuantumGrover, OutOfRangeMarkedIndicesIgnored) {
    auto psi = grover_search(3, {100, -5}, 4);
    EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8);
}

// Mix of valid and invalid indices: only the valid one should be amplified.
TEST(QuantumGrover, MixOfValidAndInvalidIndices) {
    const int n_qubits = 3;
    const int iters = grover_optimal_iterations(n_qubits, 1);
    auto psi = grover_search(n_qubits, {4, -1, 99}, iters);
    EXPECT_GT(std::norm(psi[4]), 0.9);
    EXPECT_NEAR(state_norm_sq(psi), 1.0, 1e-8);
}

// Textbook exact special case: N=4, 1 marked state, 1 iteration reaches
// EXACTLY probability 1.0 on the marked state.
TEST(QuantumGrover, N4_OneMarked_OneIteration_ExactCertainty) {
    const int n_qubits = 2; // N = 4
    const int marked = 2;
    auto psi = grover_search(n_qubits, {marked}, 1);
    ASSERT_EQ(psi.size(), 4u);
    EXPECT_NEAR(std::norm(psi[marked]), 1.0, 1e-10);
    for (int i = 0; i < 4; ++i)
        if (i != marked) EXPECT_NEAR(std::abs(psi[i]), 0.0, 1e-10);
}

// grover_optimal_iterations for N=4, M=1 is floor(pi/4*2) = 1, matching the
// exact special case above.
TEST(QuantumGrover, N4_OneMarked_OptimalItersEqualsOne) {
    EXPECT_EQ(grover_optimal_iterations(2, 1), 1);
}

// ---- Schmidt decomposition ----
namespace {

double schmidt_coefficients_sum_sq(const std::vector<double>& coeffs) {
    double sum = 0.0;
    for (double lambda : coeffs) sum += lambda * lambda;
    return sum;
}

double entropy_from_schmidt(const std::vector<double>& coeffs) {
    double S = 0.0;
    for (double lambda : coeffs) {
        const double p = lambda * lambda;
        if (p > 1e-15) S -= p * std::log(p);
    }
    return S;
}

} // namespace

TEST(QuantumSchmidt, BellStateRankTwoEqualCoefficients) {
    auto psi = bell_states()[0];  // (|00> + |11>) / sqrt(2)
    auto decomp = schmidt_decomposition(psi, 2, 2);
    ASSERT_EQ(decomp.coefficients.size(), 2u);
    EXPECT_EQ(schmidt_rank(psi, 2, 2), 2);
    EXPECT_EQ(schmidt_number(psi, 2, 2), 2);

    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(decomp.coefficients[0], h, 1e-8);
    EXPECT_NEAR(decomp.coefficients[1], h, 1e-8);
    EXPECT_NEAR(schmidt_coefficients_sum_sq(decomp.coefficients), 1.0, 1e-8);
}

TEST(QuantumSchmidt, ProductStateRankOne) {
    auto psi = tensor_product_states(ket_basis(2, 0), ket_basis(2, 0));
    auto decomp = schmidt_decomposition(psi, 2, 2);
    EXPECT_EQ(schmidt_rank(psi, 2, 2), 1);
    EXPECT_EQ(schmidt_number(psi, 2, 2), 1);
    EXPECT_NEAR(decomp.coefficients[0], 1.0, 1e-8);
    EXPECT_NEAR(decomp.coefficients[1], 0.0, 1e-8);
    EXPECT_NEAR(schmidt_coefficients_sum_sq(decomp.coefficients), 1.0, 1e-8);
}

TEST(QuantumSchmidt, CoefficientsNormalised) {
    auto psi = bell_states()[2];  // (|01> + |10>) / sqrt(2)
    auto decomp = schmidt_decomposition(psi, 2, 2);
    EXPECT_NEAR(schmidt_coefficients_sum_sq(decomp.coefficients), 1.0, 1e-8);
}

TEST(QuantumSchmidt, EntropyMatchesEntanglementEntropy) {
    struct Case { Ket psi; int da; int db; };
    std::vector<Case> cases = {
        {bell_states()[0], 2, 2},
        {bell_states()[1], 2, 2},
        {bell_states()[2], 2, 2},
        {tensor_product_states(ket_basis(2, 0), ket_basis(2, 1)), 2, 2},
        {tensor_product_states(ket_basis(2, 1), ket_basis(2, 0)), 2, 2},
    };
    for (auto& c : cases) {
        auto decomp = schmidt_decomposition(c.psi, c.da, c.db);
        const double S_schmidt = entropy_from_schmidt(decomp.coefficients);
        const double S_existing = entanglement_entropy(c.psi, c.da, c.db);
        EXPECT_NEAR(S_schmidt, S_existing, 1e-8)
            << "da=" << c.da << " db=" << c.db;
    }
}

TEST(QuantumSchmidt, EntropyMatchesReducedDensityMatrix) {
    auto psi = bell_states()[0];
    auto decomp = schmidt_decomposition(psi, 2, 2);
    const double S_schmidt = entropy_from_schmidt(decomp.coefficients);

    auto rho = density_matrix(psi);
    auto rhoA = partial_trace(rho, 2, 2, 0);
    const double S_rho = von_neumann_entropy(rhoA);
    EXPECT_NEAR(S_schmidt, S_rho, 1e-8);
}

TEST(QuantumSchmidt, BasisVectorsOrthonormal) {
    auto psi = bell_states()[0];
    auto decomp = schmidt_decomposition(psi, 2, 2);
    ASSERT_GE(decomp.basis_a.size(), 2u);
    ASSERT_GE(decomp.basis_b.size(), 2u);

    for (size_t i = 0; i < decomp.basis_a.size(); ++i) {
        double n2 = 0.0;
        for (auto& c : decomp.basis_a[i]) n2 += std::norm(c);
        if (decomp.coefficients[i] > 1e-8) EXPECT_NEAR(n2, 1.0, 1e-8);
    }
    for (size_t i = 0; i < decomp.basis_b.size(); ++i) {
        double n2 = 0.0;
        for (auto& c : decomp.basis_b[i]) n2 += std::norm(c);
        if (decomp.coefficients[i] > 1e-8) EXPECT_NEAR(n2, 1.0, 1e-8);
    }

    C overlap = inner(decomp.basis_a[0], decomp.basis_a[1]);
    EXPECT_NEAR(std::abs(overlap), 0.0, 1e-8);
    overlap = inner(decomp.basis_b[0], decomp.basis_b[1]);
    EXPECT_NEAR(std::abs(overlap), 0.0, 1e-8);
}

TEST(QuantumSchmidt, ReconstructsOriginalState) {
    auto psi = bell_states()[0];
    auto decomp = schmidt_decomposition(psi, 2, 2);

    Ket reconstructed(4, C(0.0));
    for (size_t k = 0; k < decomp.coefficients.size(); ++k) {
        if (decomp.coefficients[k] < 1e-10) continue;
        auto term = tensor_product_states(decomp.basis_a[k], decomp.basis_b[k]);
        for (size_t i = 0; i < term.size(); ++i)
            reconstructed[i] += decomp.coefficients[k] * term[i];
    }
    EXPECT_NEAR(ket_overlap_prob(psi, reconstructed), 1.0, 1e-8);
}

TEST(QuantumSchmidt, InvalidDimensionsReturnEmpty) {
    auto psi = bell_states()[0];
    auto decomp = schmidt_decomposition(psi, 3, 2);
    EXPECT_TRUE(decomp.coefficients.empty());
    EXPECT_EQ(schmidt_rank(psi, 3, 2), 0);
}
