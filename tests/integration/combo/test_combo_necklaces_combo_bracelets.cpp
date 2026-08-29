
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

TEST(IntegrationCombo,  NecklacesBraceletsTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_necklaces");
    expect_contains(interp, "help", "combo_bracelets");

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(IntegrationCombo,  KelvinKeiScalar) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}
