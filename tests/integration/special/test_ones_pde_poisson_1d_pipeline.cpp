
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

TEST(IntegrationSpecial,  PoissonAdv1d) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_poisson_1d");
    expect_contains(interp, "help", "pde_advection_1d");

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(IntegrationSpecial,  JacobiScScalar) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}
