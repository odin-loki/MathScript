
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

TEST(IntegrationLinalg,  BidiagEigTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "bidiag");
    expect_contains(interp, "help", "eig");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(IntegrationLinalg,  Theta1PrimeScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}
