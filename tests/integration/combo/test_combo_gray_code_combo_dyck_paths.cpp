
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

TEST(IntegrationCombo,  GrayDyckTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_gray_code");
    expect_contains(interp, "help", "combo_dyck_paths");

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(IntegrationCombo,  KelvinKerScalar) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}
