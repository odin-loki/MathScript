
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

TEST(IntegrationCombo,  MotzkinPartitionsTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_motzkin_paths");
    expect_contains(interp, "help", "combo_set_partitions");

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(IntegrationCombo,  HermiteHnScalar) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}
