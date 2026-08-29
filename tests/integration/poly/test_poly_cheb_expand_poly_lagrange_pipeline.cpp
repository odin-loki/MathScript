
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

TEST(IntegrationPoly,  ChebLagrangeTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_cheb_expand");
    expect_contains(interp, "help", "poly_lagrange");

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(IntegrationPoly,  StruveYnScalar) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}
