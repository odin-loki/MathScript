
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

TEST(IntegrationCfd,  Cfd1d2dAdvectionTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_grid1d");
    expect_contains(interp, "help", "cfd_run_advection");

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(IntegrationCfd,  ChebyshevUScalar) {
    Interpreter interp;
    expect_ok(interp, "u = chebyshev_u(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("u"), ms::chebyshev_u(1, 0.5), 1e-8);
}
