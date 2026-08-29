
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

TEST(IntegrationSpecial,  QuantumAntiTensor) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_anticommutator");
    expect_contains(interp, "help", "quantum_ket_tensor_product");

    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "B = [0, 1; 1, 0]");
    expect_ok(interp, "ac = quantum_anticommutator(A, B)");
    EXPECT_EQ(interp.state().matrices.at("ac").rows(), 2u);

    expect_ok(interp, "a = [1; 0]");
    expect_ok(interp, "b = [0; 1]");
    expect_ok(interp, "tp = quantum_ket_tensor_product(a, b)");
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
}

TEST(IntegrationSpecial,  ChebyshevVScalar) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}
