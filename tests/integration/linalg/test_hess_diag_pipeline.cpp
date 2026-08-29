
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

TEST(IntegrationLinalg,  LinalgGeo) {
    Interpreter interp;

    expect_contains(interp, "help", "hess(A)");
    expect_contains(interp, "help", "geo_bezier_eval");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "H = hess(A)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 2u);

    expect_ok(interp, "D = diag([1; 2; 3])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "pt = geo_bezier_eval([0, 0; 1, 2; 2, 0], 0.5)");
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
}

TEST(IntegrationLinalg,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "ln = laguerre_ln(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ln"), ms::laguerre_ln(2, 0, 0.5), 1e-9);
}
