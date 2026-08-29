
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

TEST(IntegrationPde,  Poisson2dBurgersTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_poisson_2d");
    expect_contains(interp, "help", "pde_burgers_1d");

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(IntegrationPde,  KelvinKerScalar) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1.0), 1e-8);
}
