
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

TEST(IntegrationControl,  GramTf2ssSeriesTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "control_ctrb_gram");
    expect_contains(interp, "help", "control_tf2ss");
    expect_contains(interp, "help", "control_series");

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wc").rows(), 0u);

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(IntegrationControl,  KelvinBerScalar) {
    Interpreter interp;
    expect_ok(interp, "ber = kelvin_ber(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("ber"), ms::kelvin_ber(0, 1.0), 1e-8);
}
