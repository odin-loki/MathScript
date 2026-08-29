
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

TEST(IntegrationLinalg,  GmresJacobiTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "gmres");
    expect_contains(interp, "help", "jacobi");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(IntegrationLinalg,  LambertWScalar) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}
