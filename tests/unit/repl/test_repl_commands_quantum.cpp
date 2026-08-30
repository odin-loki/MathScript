#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, control_quantum) {
    Interpreter interp;
    expect_contains(interp, "help", "control_step_final(num,den)");
    expect_contains(interp, "help", "quantum_hadamard(psi)");

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "yfinal = control_step_final(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("yfinal"), 1.0, 0.05);
    expect_contains(interp, "control_step_final([1], [1, 1])", "\n");

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_TRUE(interp.state().matrices.count("hpsi") > 0);
    const auto& hpsi = interp.state().matrices.at("hpsi");
    EXPECT_EQ(hpsi.rows(), 2u);
    EXPECT_NEAR(hpsi(0, 0), 1.0 / std::sqrt(2.0), 1e-6);
    EXPECT_NEAR(hpsi(1, 0), 1.0 / std::sqrt(2.0), 1e-6);
    expect_contains(interp, "quantum_hadamard([1; 0])", "state =");
}

TEST(ReplCommandsTest, quantum_von_neumann_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_von_neumann_entropy(rho)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "S = quantum_von_neumann_entropy(rho)");
    EXPECT_NEAR(interp.state().scalars.at("S"), 0.0, 1e-9);

    expect_contains(interp, "quantum_von_neumann_entropy([1, 0; 0, 0])", "0");
}

TEST(ReplCommandsTest, quantum_fidelity) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_fidelity(rho,sigma)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "F = quantum_fidelity(rho, rho)");
    EXPECT_NEAR(interp.state().scalars.at("F"), 1.0, 1e-6);

    expect_contains(interp, "quantum_fidelity([1, 0; 0, 0], [1, 0; 0, 0])", "1");
}

TEST(ReplCommandsTest, quantum_trace_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_trace_distance(rho,sigma)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "T = quantum_trace_distance(rho, rho)");
    EXPECT_NEAR(interp.state().scalars.at("T"), 0.0, 1e-9);

    expect_contains(interp, "quantum_trace_distance([1, 0; 0, 0], [1, 0; 0, 0])", "0");
}

TEST(ReplCommandsTest, quantum_concurrence) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_concurrence(rho)");

    expect_ok(interp, "rho4 = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "Cq = quantum_concurrence(rho4)");
    EXPECT_NEAR(interp.state().scalars.at("Cq"), 0.0, 1e-9);

    expect_contains(interp,
                    "quantum_concurrence([1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0])",
                    "0");
}

TEST(ReplCommandsTest, quantum_entanglement_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_entanglement_entropy(psi,dim_a,dim_b)");

    expect_ok(interp, "psi = [1; 0; 0; 0]");
    expect_ok(interp, "Ee = quantum_entanglement_entropy(psi, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("Ee"), 0.0, 1e-9);

    expect_contains(interp, "quantum_entanglement_entropy([1; 0; 0; 0], 2, 2)", "0");
}

TEST(ReplCommandsTest, quantum_expectation) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_expectation(psi,A)");

    expect_ok(interp, "ex = quantum_expectation([1; 0], [1, 0; 0, -1])");
    EXPECT_NEAR(interp.state().scalars.at("ex"), 1.0, 1e-9);

    expect_contains(interp, "quantum_expectation([1; 0], [1, 0; 0, -1])", "1");
}

TEST(ReplCommandsTest, quantum_expectation_dm) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_expectation_dm(rho,op)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "exdm = quantum_expectation_dm(rho, [1, 0; 0, -1])");
    EXPECT_NEAR(interp.state().scalars.at("exdm"), 1.0, 1e-9);

    expect_contains(interp, "quantum_expectation_dm([1, 0; 0, 0], [1, 0; 0, -1])", "1");
}

TEST(ReplCommandsTest, quantum_inner) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_inner(bra,ket)");

    expect_ok(interp, "inn = quantum_inner([1; 0], [1; 0])");
    EXPECT_NEAR(interp.state().scalars.at("inn"), 1.0, 1e-9);

    expect_contains(interp, "quantum_inner([1; 0], [1; 0])", "1");
}

TEST(ReplCommandsTest, quantum_ket_normalise) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_normalise(psi)");

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_TRUE(interp.state().matrices.count("psi_n") > 0);
    const auto& psi_n = interp.state().matrices.at("psi_n");
    EXPECT_EQ(psi_n.rows(), 2u);
    EXPECT_EQ(psi_n.cols(), 1u);
    EXPECT_NEAR(psi_n(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(psi_n(1, 0), 0.0, 1e-9);
    double psi_norm_sq = 0.0;
    for (size_t i = 0; i < psi_n.rows(); ++i) {
        psi_norm_sq += psi_n(i, 0) * psi_n(i, 0);
    }
    EXPECT_NEAR(std::sqrt(psi_norm_sq), 1.0, 1e-9);

    expect_contains(interp, "quantum_ket_normalise([2; 0])", "state =");
}

TEST(ReplCommandsTest, quantum_partial_trace) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_partial_trace(rho,d1,d2,subsystem)");

    expect_ok(interp, "rho4 = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "rhoA = quantum_partial_trace(rho4, 2, 2, 0)");
    ASSERT_TRUE(interp.state().matrices.count("rhoA") > 0);
    const auto& rhoA = interp.state().matrices.at("rhoA");
    EXPECT_EQ(rhoA.rows(), 2u);
    EXPECT_EQ(rhoA.cols(), 2u);
    EXPECT_NEAR(rhoA(0, 0), 1.0, 1e-9);

    expect_contains(interp,
                    "quantum_partial_trace([1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0], 2, 2, 0)",
                    "rho =");
}

TEST(ReplCommandsTest, quantum_pauli_x) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_x()");

    expect_ok(interp, "X = quantum_pauli_x()");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    const auto& X = interp.state().matrices.at("X");
    EXPECT_EQ(X.rows(), 2u);
    EXPECT_EQ(X.cols(), 2u);
    EXPECT_NEAR(X(0, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_x()", "op =");
}

TEST(ReplCommandsTest, quantum_pauli_z) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_z()");

    expect_ok(interp, "Z = quantum_pauli_z()");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 2u);
    EXPECT_EQ(Z.cols(), 2u);
    EXPECT_NEAR(Z(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(Z(1, 1), -1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_z()", "op =");
}

TEST(ReplCommandsTest, quantum_cnot_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_cnot_gate()");

    expect_ok(interp, "CNOT = quantum_cnot_gate()");
    ASSERT_GT(interp.state().matrices.count("CNOT"), 0u);
    const auto& CNOT = interp.state().matrices.at("CNOT");
    EXPECT_EQ(CNOT.rows(), 4u);
    EXPECT_EQ(CNOT.cols(), 4u);

    expect_contains(interp, "quantum_cnot_gate()", "op =");
}

TEST(ReplCommandsTest, quantum_pauli_plus) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_plus()");

    expect_ok(interp, "Pp = quantum_pauli_plus()");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    const auto& Pp = interp.state().matrices.at("Pp");
    EXPECT_EQ(Pp.rows(), 2u);
    EXPECT_EQ(Pp.cols(), 2u);
    EXPECT_NEAR(Pp(0, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_plus()", "op =");
}

TEST(ReplCommandsTest, quantum_pauli_minus) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_minus()");

    expect_ok(interp, "Pm = quantum_pauli_minus()");
    ASSERT_GT(interp.state().matrices.count("Pm"), 0u);
    const auto& Pm = interp.state().matrices.at("Pm");
    EXPECT_EQ(Pm.rows(), 2u);
    EXPECT_EQ(Pm.cols(), 2u);
    EXPECT_NEAR(Pm(1, 0), 1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_minus()", "op =");
}

TEST(ReplCommandsTest, quantum_toffoli_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_toffoli_gate()");

    expect_ok(interp, "T = quantum_toffoli_gate()");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    const auto& T = interp.state().matrices.at("T");
    EXPECT_EQ(T.rows(), 8u);
    EXPECT_EQ(T.cols(), 8u);
    EXPECT_NEAR(T(6, 7), 1.0, 1e-9);
    EXPECT_NEAR(T(7, 6), 1.0, 1e-9);

    expect_contains(interp, "quantum_toffoli_gate()", "op =");
}

TEST(ReplCommandsTest, quantum_pauli_y) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_y()");

    expect_ok(interp, "Y = quantum_pauli_y()");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    const auto& Y = interp.state().matrices.at("Y");
    EXPECT_EQ(Y.rows(), 2u);
    EXPECT_EQ(Y.cols(), 2u);

    expect_contains(interp, "quantum_pauli_y()", "op =");
}

TEST(ReplCommandsTest, quantum_swap_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_swap_gate()");

    expect_ok(interp, "SWAP = quantum_swap_gate()");
    ASSERT_GT(interp.state().matrices.count("SWAP"), 0u);
    const auto& SWAP = interp.state().matrices.at("SWAP");
    EXPECT_EQ(SWAP.rows(), 4u);
    EXPECT_EQ(SWAP.cols(), 4u);
    EXPECT_NEAR(SWAP(1, 2), 1.0, 1e-9);
    EXPECT_NEAR(SWAP(2, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_swap_gate()", "op =");
}

TEST(ReplCommandsTest, quantum_identity) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_identity()");

    expect_ok(interp, "I2 = quantum_identity()");
    ASSERT_GT(interp.state().matrices.count("I2"), 0u);
    const auto& I2 = interp.state().matrices.at("I2");
    EXPECT_EQ(I2.rows(), 2u);
    EXPECT_EQ(I2.cols(), 2u);
    EXPECT_NEAR(I2(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(I2(1, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_identity()", "op =");
}

TEST(ReplCommandsTest, quantum_hadamard_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_hadamard_gate()");

    expect_ok(interp, "H = quantum_hadamard_gate()");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    const auto& H = interp.state().matrices.at("H");
    EXPECT_EQ(H.rows(), 2u);
    EXPECT_EQ(H.cols(), 2u);
    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(H(0, 0), h, 1e-9);

    expect_contains(interp, "quantum_hadamard_gate()", "op =");
}

TEST(ReplCommandsTest, quantum_rotation_z) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_rotation_z(theta)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Rz = quantum_rotation_z(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Rz"), 0u);
    const auto& Rz = interp.state().matrices.at("Rz");
    EXPECT_EQ(Rz.rows(), 2u);
    EXPECT_EQ(Rz.cols(), 2u);

    expect_contains(interp, "quantum_rotation_z(pi/2)", "op =");
}

TEST(ReplCommandsTest, quantum_rotation_x) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_rotation_x(theta)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Rx = quantum_rotation_x(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Rx"), 0u);
    const auto& Rx = interp.state().matrices.at("Rx");
    EXPECT_EQ(Rx.rows(), 2u);
    EXPECT_EQ(Rx.cols(), 2u);

    expect_contains(interp, "quantum_rotation_x(pi/2)", "op =");
}

TEST(ReplCommandsTest, quantum_rotation_y) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_rotation_y(theta)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Ry = quantum_rotation_y(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Ry"), 0u);
    const auto& Ry = interp.state().matrices.at("Ry");
    EXPECT_EQ(Ry.rows(), 2u);
    EXPECT_EQ(Ry.cols(), 2u);

    expect_contains(interp, "quantum_rotation_y(pi/2)", "op =");
}

TEST(ReplCommandsTest, quantum_phase_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_phase_gate(theta)");

    expect_ok(interp, "P = quantum_phase_gate(1.57)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    const auto& P = interp.state().matrices.at("P");
    EXPECT_EQ(P.rows(), 2u);
    EXPECT_EQ(P.cols(), 2u);
    for (size_t i = 0; i < 2u; ++i) {
        for (size_t j = 0; j < 2u; ++j) {
            EXPECT_TRUE(std::isfinite(P(i, j)));
        }
    }

    expect_contains(interp, "quantum_phase_gate(1.57)", "op =");
}

TEST(ReplCommandsTest, quantum_qft_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_qft_gate(n_qubits)");

    expect_ok(interp, "Q = quantum_qft_gate(2)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    const auto& Q = interp.state().matrices.at("Q");
    EXPECT_EQ(Q.rows(), 4u);
    EXPECT_EQ(Q.cols(), 4u);

    expect_contains(interp, "quantum_qft_gate(2)", "op =");
}

TEST(ReplCommandsTest, quantum_ket_basis) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_basis(dim,index)");

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    const auto& kb = interp.state().matrices.at("kb");
    EXPECT_NEAR(kb(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(kb(1, 0), 0.0, 1e-9);

    expect_contains(interp, "quantum_ket_basis(2, 0)", "state =");
}

TEST(ReplCommandsTest, quantum_fock_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_fock_state(n,n_max)");

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    const auto& fs = interp.state().matrices.at("fs");
    EXPECT_NEAR(fs(1, 0), 1.0, 1e-9);

    expect_contains(interp, "quantum_fock_state(1, 3)", "state =");
}

TEST(ReplCommandsTest, quantum_identity_n) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_identity_n(dim)");

    expect_ok(interp, "I3 = quantum_identity_n(3)");
    ASSERT_GT(interp.state().matrices.count("I3"), 0u);
    const auto& I3 = interp.state().matrices.at("I3");
    EXPECT_EQ(I3.rows(), 3u);
    EXPECT_EQ(I3.cols(), 3u);
    EXPECT_NEAR(I3(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(I3(2, 2), 1.0, 1e-9);

    expect_contains(interp, "quantum_identity_n(3)", "op =");
}

TEST(ReplCommandsTest, quantum_ket_superposition) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_superposition(amps)");

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
    const auto& sup = interp.state().matrices.at("sup");
    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(sup(0, 0), h, 1e-9);
    EXPECT_NEAR(sup(1, 0), h, 1e-9);

    expect_contains(interp, "quantum_ket_superposition([1; 1])", "state =");
}

TEST(ReplCommandsTest, quantum_ghz_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ghz_state(n)");

    expect_ok(interp, "ghz = quantum_ghz_state(3)");
    ASSERT_GT(interp.state().matrices.count("ghz"), 0u);
    const auto& ghz = interp.state().matrices.at("ghz");
    EXPECT_EQ(ghz.rows(), 8u);
    EXPECT_EQ(ghz.cols(), 1u);
    const double amp = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(ghz(0, 0), amp, 1e-9);
    EXPECT_NEAR(ghz(7, 0), amp, 1e-9);

    expect_contains(interp, "quantum_ghz_state(3)", "state =");
}

TEST(ReplCommandsTest, quantum_w_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_w_state(n)");

    expect_ok(interp, "w = quantum_w_state(3)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 8u);
    EXPECT_EQ(w.cols(), 1u);
    const double amp = 1.0 / std::sqrt(3.0);
    EXPECT_NEAR(w(1, 0), amp, 1e-9);
    EXPECT_NEAR(w(2, 0), amp, 1e-9);
    EXPECT_NEAR(w(4, 0), amp, 1e-9);

    expect_contains(interp, "quantum_w_state(3)", "state =");
}

TEST(ReplCommandsTest, quantum_coherent_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_coherent_state(alpha_re,alpha_im,n_max)");

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);

    expect_contains(interp, "quantum_coherent_state(1, 0, 20)", "state =");
}

TEST(ReplCommandsTest, quantum_bell_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_bell_state(index)");

    expect_ok(interp, "s = quantum_bell_state(0)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& bell = interp.state().matrices.at("s");
    EXPECT_EQ(bell.rows(), 4u);
    EXPECT_NEAR(bell(0, 0), 0.707107, 1e-4);
    EXPECT_NEAR(bell(3, 0), 0.707107, 1e-4);

    expect_contains(interp, "quantum_bell_state(0)", "state =");
}

TEST(ReplCommandsTest, quantum_schrodinger) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schrodinger(H,psi0,t0,t1,n_steps)");

    expect_ok(interp, "traj = quantum_schrodinger([0.5, 0; 0, -0.5], [1; 0], 0, 3.141592653589793, 100)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 101u);
    EXPECT_EQ(interp.state().matrices.at("traj").cols(), 2u);
}

TEST(ReplCommandsTest, quantum_time_evolution) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_time_evolution(H,t)");

    expect_ok(interp, "U0 = quantum_time_evolution([0, 0.5; 0.5, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("U0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("U0")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("U0")(1, 1), 1.0, 1e-6);
}

TEST(ReplCommandsTest, quantum_density_matrix) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_density_matrix(psi)");

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rho")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, quantum_op_apply) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_op_apply(op,psi)");

    expect_ok(interp, "psi_h = quantum_op_apply(quantum_hadamard_gate(), [1; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_h"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("psi_h")(0, 0), 0.707, 0.05);
    EXPECT_NEAR(interp.state().matrices.at("psi_h")(1, 0), 0.707, 0.05);
}

TEST(ReplCommandsTest, quantum_schrodinger_final) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schrodinger_final(H,psi0,t0,t1,n_steps)");

    expect_ok(interp,
              "psi = quantum_schrodinger_final([0.5, 0; 0, -0.5], [1; 0], 0, 3.141592653589793, 100)");
    ASSERT_GT(interp.state().matrices.count("psi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("psi").cols(), 1u);
}

TEST(ReplCommandsTest, quantum_commutator) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_commutator(A,B)");

    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "Z = quantum_pauli_z()");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    double comm_norm = 0.0;
    const auto& C = interp.state().matrices.at("C");
    for (size_t i = 0; i < C.rows(); ++i) {
        for (size_t j = 0; j < C.cols(); ++j) {
            comm_norm += C(i, j) * C(i, j);
        }
    }
    EXPECT_GT(std::sqrt(comm_norm), 0.0);
}

TEST(ReplCommandsTest, quantum_tensor_product) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_tensor_product(A,B)");

    expect_ok(interp, "I2 = quantum_identity_n(2)");
    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "tp = quantum_tensor_product(I2, X)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("tp").cols(), 4u);
}

TEST(ReplCommandsTest, quantum_phase) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_wigner(rho,x,p)");
    expect_contains(interp, "help", "quantum_husimi(rho,alpha_re,alpha_im)");
    expect_contains(interp, "help", "quantum_grover_search(n_qubits,marked_indices[,n_iterations])");

    expect_ok(interp, "psi1 = quantum_fock_state(1, 3)");
    expect_ok(interp, "rho1 = quantum_density_matrix(psi1)");

    expect_ok(interp, "w = quantum_wigner(rho1, 0, 0)");
    EXPECT_LT(interp.state().scalars.at("w"), 0.0);
    expect_contains(interp, "quantum_wigner(rho1, 0, 0)", "-");

    expect_ok(interp, "q = quantum_husimi(rho1, 1, 0)");
    EXPECT_GE(interp.state().scalars.at("q"), 0.0);
    expect_contains(interp, "quantum_husimi(rho1, 1, 0)", "0.");

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g").rows(), 4u);
    EXPECT_GT(interp.state().matrices.at("g")(1, 0), interp.state().matrices.at("g")(0, 0));
    expect_contains(interp, "quantum_grover_search(2, [1], 1)", "state =");

    expect_ok(interp, "g2 = quantum_grover_search(2, [1])");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_spectrum) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_eigenspectrum(H)");
    expect_contains(interp, "help", "quantum_ground_state(H)");

    expect_ok(interp, "H = [2, 0; 0, 5]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("evals")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("evals")(1, 0), 5.0, 1e-9);

    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    const auto ref_gs = ms::quantum::ground_state({{{2.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {5.0, 0.0}}});
    EXPECT_NEAR(interp.state().matrices.at("gs")(0, 0), ref_gs[0].real(), 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("gs")(1, 0), ref_gs[1].real(), 1e-9);
}

TEST(ReplCommandsTest, quantum_schmidt_anticommutator) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schmidt_decomposition(psi,dim_a,dim_b)");
    expect_contains(interp, "help", "quantum_anticommutator(A,B)");

    expect_ok(interp, "psi = [0.5; 0.5; 0.5; 0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_schmidt_tensor_outer) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schmidt_number");
    expect_contains(interp, "help", "quantum_ket_tensor_product");
    expect_contains(interp, "help", "quantum_outer");

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sn = quantum_schmidt_number(psi, 2, 2)");
    EXPECT_GE(interp.state().scalars.at("sn"), 1.0);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
    expect_ok(interp, "rho = quantum_outer(a, b)");
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_schmidt) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_dagger");
    expect_contains(interp, "help", "quantum_schmidt_bases");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);

    expect_ok(interp, "le = fem_lagrange_eval(0.25)");
    EXPECT_NEAR(interp.state().matrices.at("le")(0, 0), 0.75, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("le")(0, 1), 0.25, 1e-12);
}

TEST(ReplCommandsTest, quantum_bell_states) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_bell_states");
    expect_ok(interp, "bells = quantum_bell_states()");
    EXPECT_EQ(interp.state().matrices.at("bells").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("bells")(0, 0), 283.0, 1e-12);
}

TEST(ReplCommandsTest, quantum_izaac) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_EQ(interp.state().matrices.at("Ad").rows(), 2u);

    expect_ok(interp, "izaac seed 7");
    expect_ok(interp, "rm = izaac_rand_matrix(2, 3)");
    EXPECT_EQ(interp.state().matrices.at("rm").cols(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_quantum_fem_cfd) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, schrodinger_cellmemory_ptrace) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm330, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm330)");
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_eigen_grover) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, grover_iters_scalar) {
    Interpreter interp;
    expect_contains(interp, "quantum_grover_optimal_iterations(3, 1)", "2");
}

TEST(ReplCommandsTest, matmuldm_izaac) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);
}

TEST(ReplCommandsTest, schmidt_sqrtm) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, diffgeo_ket) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, ket_fock) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, schmidt_mpc) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, eigen_grover) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);

    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, ket_op_comm) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, anti_tensor) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, matmuldm_izaac_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);
}

TEST(ReplCommandsTest, schmidt_sqrtm_2) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, diffgeo_ket_2) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, ket_fock_2) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, schmidt_mpc_2) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_2) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, eigen_grover_2) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);

    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, ket_op_comm_2) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, anti_tensor_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_2) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_3) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_2) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_2) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_3) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_2) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_2) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_2) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_4) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_3) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_3) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_4) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_3) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_3) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_3) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_3) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_5) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_4) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_4) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_5) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_4) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_4) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_4) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_4) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_6) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_5) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_5) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_6) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_5) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_5) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_5) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_5) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_7) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_6) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_6) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_7) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_6) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_6) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_6) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_6) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_8) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_7) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_7) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_8) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_7) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_7) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_7) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_7) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_9) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_8) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_8) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_9) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_8) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_8) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_8) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_8) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_10) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_9) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_9) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_10) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_9) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_9) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_9) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_9) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_11) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_10) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_10) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_11) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_10) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_10) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_10) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_10) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_12) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_11) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_11) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_12) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_11) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_11) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_11) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_11) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_13) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_12) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_12) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_13) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_12) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_12) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_12) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_12) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_14) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_13) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_13) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_14) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_13) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_13) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_13) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_13) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_15) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_14) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_14) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_15) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_14) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_14) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_14) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_14) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_16) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_15) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_15) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_16) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_15) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_15) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_15) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_15) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_17) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_16) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_16) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_17) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_16) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_16) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_16) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_16) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_18) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_17) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_17) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_18) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_17) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_17) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_17) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_17) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_19) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_18) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_18) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_19) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_18) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_18) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_18) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_18) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_20) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_19) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_19) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_20) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_19) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_19) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_19) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_19) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_21) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_20) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_20) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_21) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_20) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_20) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_20) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_20) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_22) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_21) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_21) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_22) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_21) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_21) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_21) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_21) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_23) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_22) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_22) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_23) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_22) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_22) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_22) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_22) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_24) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_23) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_23) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_24) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_23) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_23) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_23) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_23) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_25) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_24) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_24) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_25) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_24) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_24) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_24) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_24) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_26) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_25) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_25) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_26) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_25) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_25) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_25) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_25) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_27) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_26) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_26) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_27) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_26) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_26) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_26) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_26) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_28) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_27) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_27) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_28) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_27) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_27) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_27) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_27) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_29) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_28) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_28) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_29) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_28) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_28) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_28) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_28) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_30) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_29) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_29) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_30) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_29) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_29) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_29) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_29) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_31) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_30) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_30) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_31) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_30) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere_quantum_ket_superposition_30) {
    Interpreter interp;

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_basis_quantum_fock_state_30) {
    Interpreter interp;

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_schmidt_decomposition_mpc_split_30) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5;0.5;0.5;0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_GT(interp.state().matrices.count("sh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(ReplCommandsTest, equity_schrodinger_32) {
    Interpreter interp;

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("eq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_eigen_grover_31) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    ASSERT_GT(interp.state().matrices.count("evals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    ASSERT_GT(interp.state().matrices.count("gs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, quantum_ket_density_op_comm_31) {
    Interpreter interp;

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi_n").rows(), 2u);

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "out = quantum_op_apply(X, psi)");
    ASSERT_GT(interp.state().matrices.count("out"), 0u);
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "Z = [1, 0; 0, -1]");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_anti_tensor_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(ReplCommandsTest, quantum_coherent_32) {
    Interpreter interp;

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(ReplCommandsTest, quantum_dagger_matmul_dm_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(ReplCommandsTest, izaac_rand_matrix_schmidt_bases_31) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(ReplCommandsTest, quantum_density_matrix_noassign) {
    Interpreter interp;
    expect_contains(interp, "quantum_density_matrix([1; 0])", "rho");
    expect_error_contains(interp, "quantum_density_matrix(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_grover_search_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "quantum_grover_search(2, [1], 1)", "state =");
    expect_error_contains(interp, "quantum_grover_search(0, [1])",
                          "positive integer n_qubits");
}

TEST(ReplCommandsTest, quantum_schmidt_rank_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "quantum_schmidt_rank([1; 0; 0; 0], 2, 2)", "1");
    expect_error_contains(interp, "quantum_schmidt_rank([1; 0; 0; 0], 0, 2)",
                          "positive integer dim_a and dim_b");
}

TEST(ReplCommandsTest, quantum_entanglement_entropy_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "quantum_entanglement_entropy([1; 0; 0; 0], 2, 2)", "0");
    expect_error_contains(interp, "quantum_entanglement_entropy([1; 0; 0; 0], 1.5, 2)",
                          "positive integer dim_a and dim_b");
}

TEST(ReplCommandsTest, quantum_schrodinger_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "quantum_schrodinger([0.5, 0; 0, -0.5], [1; 0], 0, 0.1, 5)",
                    "traj =");
    expect_error_contains(interp, "quantum_schrodinger([0.5, 0; 0, -0.5], [1; 0], 0, 0.1, 1.5)",
                          "non-negative integer n_steps");
}

TEST(ReplCommandsTest, quantum_husimi_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "psi1 = quantum_fock_state(1, 3)");
    expect_ok(interp, "rho1 = quantum_density_matrix(psi1)");
    expect_ok(interp, "quantum_husimi(rho1, 1, 0)");
    expect_error_contains(interp, "quantum_husimi(rho1, missing, 0)",
                          "quantum_husimi");
}

TEST(ReplCommandsTest, quantum_hadamard_noassign) {
    Interpreter interp;
    expect_contains(interp, "quantum_hadamard([1; 0])", "state");
    expect_error_contains(interp, "quantum_hadamard([1; 0; 0])", "2x1");
}

TEST(ReplCommandsTest, quantum_ket_normalise_noassign) {
    Interpreter interp;
    expect_contains(interp, "quantum_ket_normalise([2; 0])", "state");
    expect_error_contains(interp, "quantum_ket_normalise([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, quantum_ket_superposition_noassign) {
    Interpreter interp;
    expect_contains(interp, "quantum_ket_superposition([1; 1])", "state");
    expect_error_contains(interp, "quantum_ket_superposition([1, 2; 3, 4])",
                          "coefficient vector");
}

TEST(ReplCommandsTest, quantum_purity_noassign) {
    Interpreter interp;
    expect_contains(interp, "quantum_purity([1, 0; 0, 0])", "1");
    expect_error_contains(interp, "quantum_purity(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_von_neumann_entropy_noassign) {
    Interpreter interp;
    expect_contains(interp, "quantum_von_neumann_entropy([1, 0; 0, 0])", "0");
    expect_error_contains(interp, "quantum_von_neumann_entropy(no_such_matrix)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, quantum_concurrence_noassign) {
    Interpreter interp;
    expect_contains(interp,
                    "quantum_concurrence([1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0])",
                    "0");
    expect_error_contains(interp, "quantum_concurrence(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_tensor_product_noassign) {
    Interpreter interp;
    expect_ok(interp, "I2 = quantum_identity_n(2)");
    expect_ok(interp, "X = quantum_pauli_x()");
    expect_contains(interp, "quantum_tensor_product(I2, X)", "op =");
    expect_error_contains(interp, "quantum_tensor_product(missing, X)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_schmidt_number_noassign) {
    Interpreter interp;
    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_contains(interp, "quantum_schmidt_number(psi, 2, 2)", "2");
    expect_error_contains(interp, "quantum_schmidt_number(missing, 2, 2)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_uncertainty_noassign) {
    Interpreter interp;
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "Z = quantum_pauli_z()");
    expect_contains(interp, "quantum_uncertainty(psi, X, Z)", "0");
    expect_error_contains(interp, "quantum_uncertainty(missing, X, Z)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_schrodinger_final_noassign) {
    Interpreter interp;
    expect_ok(interp, "H = [0.5, 0; 0, -0.5]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_contains(interp, "quantum_schrodinger_final(H, psi0, 0, 0.1, 10)", "psi =");
    expect_error_contains(interp, "quantum_schrodinger_final(missing, psi0, 0, 0.1, 10)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, quantum_time_evolution_noassign) {
    Interpreter interp;
    expect_ok(interp, "H = [0, 0.5; 0.5, 0]");
    expect_contains(interp, "quantum_time_evolution(H, 0)", "U =");
    expect_error_contains(interp, "quantum_time_evolution(missing, 0)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_op_apply_noassign) {
    Interpreter interp;
    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "psi = [1; 0]");
    expect_contains(interp, "quantum_op_apply(X, psi)", "psi =");
    expect_error_contains(interp, "quantum_op_apply(missing, psi)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_commutator_noassign) {
    Interpreter interp;
    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "Z = quantum_pauli_z()");
    expect_contains(interp, "quantum_commutator(X, Z)", "comm =");
    expect_error_contains(interp, "quantum_commutator(missing, Z)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_ket_tensor_product_noassign) {
    Interpreter interp;
    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_contains(interp, "quantum_ket_tensor_product(a, b)", "tp =");
    expect_error_contains(interp, "quantum_ket_tensor_product(missing, b)", "unknown matrix");
}

TEST(ReplCommandsTest, quantum_outer_noassign) {
    Interpreter interp;
    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_contains(interp, "quantum_outer(a, b)", "outer =");
    expect_error_contains(interp, "quantum_outer(missing, b)", "unknown matrix");
}
