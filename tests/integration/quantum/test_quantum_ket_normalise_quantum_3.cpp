
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationQuantum,  QuantumKetDensityOpCommTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_normalise");
    expect_contains(interp, "help", "quantum_density_matrix");
    expect_contains(interp, "help", "quantum_op_apply");
    expect_contains(interp, "help", "quantum_commutator");

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

TEST(IntegrationQuantum,  LaguerreLScalar) {
    Interpreter interp;
    expect_ok(interp, "ll = laguerre_l(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ll"), ms::laguerre_l(2, 0.5), 1e-8);
}
