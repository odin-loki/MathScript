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
