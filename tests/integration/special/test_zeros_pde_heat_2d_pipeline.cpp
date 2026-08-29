
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

TEST(IntegrationSpecial,  Heat2d) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_heat_2d");
    expect_contains(interp, "help", "pde_heat_2d_cn_adi");

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(IntegrationSpecial,  JacobiAmScalar) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}
