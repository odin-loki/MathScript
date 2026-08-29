
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

TEST(IntegrationQuantum,  QuantumEigenGrover) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_eigenspectrum");
    expect_contains(interp, "help", "quantum_ground_state");
    expect_contains(interp, "help", "quantum_grover_search");

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    EXPECT_EQ(interp.state().matrices.at("evals").rows(), 2u);
    expect_ok(interp, "gs = quantum_ground_state(H)");
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "g = quantum_grover_search(2, [1], 1)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(IntegrationQuantum,  KelvinKeiScalar) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}
