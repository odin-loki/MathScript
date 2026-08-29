
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

TEST(IntegrationPoly,  NewtonRootsTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_interp_newton");
    expect_contains(interp, "help", "poly_roots");

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(IntegrationPoly,  KelvinBerScalar) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}
