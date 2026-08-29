
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

TEST(IntegrationSpecial,  Wave2dFemPoisson1dTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_wave_2d");
    expect_contains(interp, "help", "fem_poisson1d");

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(IntegrationSpecial,  StruveHScalar) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(1, 1.0), 1e-8);
}
