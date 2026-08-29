
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

TEST(IntegrationSpecial,  Rk4SparseCoo) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_rk4");
    expect_contains(interp, "help", "sparse_from_coo");

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(IntegrationSpecial,  ChebyshevUScalar) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}
