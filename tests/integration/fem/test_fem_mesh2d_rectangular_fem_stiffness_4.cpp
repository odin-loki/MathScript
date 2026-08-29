
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

TEST(IntegrationFem,  FemSolveCfdGridTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_solve");
    expect_contains(interp, "help", "cfd_grid2d");

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
}

TEST(IntegrationFem,  JacobiCdScalar) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.2, 0.3), 1e-8);
}
