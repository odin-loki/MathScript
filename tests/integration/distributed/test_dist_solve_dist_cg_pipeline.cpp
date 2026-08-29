
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

TEST(IntegrationDistributed,  DistSolveCgGmresJacobi) {
    Interpreter interp;
    expect_contains(interp, "help", "dist_solve");
    expect_contains(interp, "help", "dist_cg");
    expect_contains(interp, "help", "dist_gmres");
    expect_contains(interp, "help", "dist_jacobi");

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(IntegrationDistributed,  ZetaHurwitzScalar) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}
