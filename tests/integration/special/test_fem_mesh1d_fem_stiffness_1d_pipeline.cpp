
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

TEST(IntegrationSpecial,  FemMesh1dStiffness) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_mesh1d");
    expect_contains(interp, "help", "fem_stiffness_1d");

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(IntegrationSpecial,  JacobiNsScalar) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}
