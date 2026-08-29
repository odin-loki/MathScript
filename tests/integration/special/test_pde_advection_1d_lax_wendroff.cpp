
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

TEST(IntegrationSpecial,  LaxWendroffReactionDiffTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_advection_1d_lax_wendroff");
    expect_contains(interp, "help", "pde_reaction_diffusion_1d");

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
}

TEST(IntegrationSpecial,  BesselZeroYnuScalar) {
    Interpreter interp;
    expect_ok(interp, "z = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("z"), ms::bessel_zero_ynu(0, 1), 1e-8);
}
