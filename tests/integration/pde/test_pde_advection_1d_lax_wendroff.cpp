
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

TEST(IntegrationPde,  LaxWendroffReactionDiff) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_advection_1d_lax_wendroff(u0,v,dx,dt,steps)");
    expect_contains(interp, "help", "pde_reaction_diffusion_1d(u0,D,r,dx,dt,steps)");

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(IntegrationPde,  JacobiScScalar) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}
