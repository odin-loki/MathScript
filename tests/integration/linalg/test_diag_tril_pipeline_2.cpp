
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

TEST(IntegrationLinalg,  DiagTrilTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "diag");
    expect_contains(interp, "help", "tril");

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(IntegrationLinalg,  Theta1Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}
