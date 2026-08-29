
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

TEST(IntegrationFem,  FemLoad1dLagrangeEval) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_load_1d");
    expect_contains(interp, "help", "fem_lagrange_eval");

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);

    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(IntegrationFem,  JacobiDsScalar) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}
