
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

TEST(IntegrationLinalg,  SqrtmLogmTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "sqrtm");
    expect_contains(interp, "help", "logm");

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(IntegrationLinalg,  Theta2Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}
