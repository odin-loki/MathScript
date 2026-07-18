// MathScript Integration Tests: REPL Interpreter – Wave 233 Control/Quantum Bindings

#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave233Pipeline, ControlAnalysisBindings) {
    Interpreter interp;

    // H(s) = 1/(s+1)^2 → repeated pole at -1
    expect_contains(interp, "control_poles([1], [1, 2, 1])", "-1");
    expect_contains(interp, "control_zeros([1, 1], [1, 2, 1])", "-1");
    expect_ok(interp, "info = control_step_info([1], [1, 1])");
    expect_ok(interp, "nyq = control_nyquist([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("nyq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("nyq").cols(), 2u);
    EXPECT_GT(interp.state().matrices.at("nyq").rows(), 0u);

    expect_contains(interp, "help", "control_poles(num,den)");
    expect_contains(interp, "help", "control_zeros(num,den)");
    expect_contains(interp, "help", "control_step_info(num,den)");
    expect_contains(interp, "help", "control_nyquist(num,den)");
}

TEST(ReplWave233Pipeline, QuantumInfoBindings) {
    Interpreter interp;

    expect_ok(interp, "psi0 = quantum_ket_basis(2, 0)");
    expect_ok(interp, "rho0 = quantum_density_matrix(psi0)");
    expect_contains(interp, "quantum_purity(rho0)", "1");

    expect_ok(interp, "psi4 = quantum_ket_basis(4, 0)");
    expect_contains(interp, "quantum_schmidt_rank(psi4, 2, 2)", "1");

    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "Z = quantum_pauli_z()");
    expect_ok(interp, "quantum_uncertainty(psi0, X, Z)");

    expect_contains(interp, "quantum_grover_optimal_iterations(3, 1)", "2");

    expect_contains(interp, "help", "quantum_purity(rho)");
    expect_contains(interp, "help", "quantum_schmidt_rank(psi,dim_a,dim_b)");
    expect_contains(interp, "help", "quantum_uncertainty(psi,A,B)");
    expect_contains(interp, "help", "quantum_grover_optimal_iterations(n_qubits,n_marked)");
}
