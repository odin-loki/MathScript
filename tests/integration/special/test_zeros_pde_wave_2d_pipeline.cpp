
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

TEST(IntegrationSpecial,  Wave2d) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_wave_2d");

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(IntegrationSpecial,  JacobiDsScalar) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}
