
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

TEST(IntegrationQuantum,  QuantumDaggerMatmulTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_dagger");
    expect_contains(interp, "help", "quantum_matmul_dm");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 1), 3.0, 1e-9);

    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);
}

TEST(IntegrationQuantum,  Theta1Scalar) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}
