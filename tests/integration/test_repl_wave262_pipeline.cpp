// MathScript Integration Tests: REPL Interpreter – Wave 262 Pipeline
//
// combo_motzkin_paths / combo_set_partitions REPL bindings.
// Smoke: Wave 261 APIs on main (combo_dyck_paths, combo_gray_code).

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave262Pipeline, ComboDyckPathsSmoke) {
    Interpreter interp;

    expect_ok(interp, "d2 = combo_dyck_paths(2)");
    ASSERT_GT(interp.state().matrices.count("d2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d2").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("d2").cols(), 4u);
    expect_contains(interp, "help", "combo_dyck_paths(n)");
}

TEST(ReplWave262Pipeline, ComboGrayCodeSmoke) {
    Interpreter interp;

    expect_ok(interp, "gc = combo_gray_code(2)");
    ASSERT_GT(interp.state().matrices.count("gc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gc").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("gc").cols(), 2u);
    expect_contains(interp, "help", "combo_gray_code(n)");
}

TEST(ReplWave262Pipeline, ComboMotzkinPaths) {
    Interpreter interp;

    expect_ok(interp, "mp = combo_motzkin_paths(3)");
    ASSERT_GT(interp.state().matrices.count("mp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mp").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("mp").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("mp")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp")(0, 2), -1.0, 1e-9);
    expect_contains(interp, "help", "combo_motzkin_paths(n)");
}

TEST(ReplWave262Pipeline, ComboSetPartitions) {
    Interpreter interp;

    expect_ok(interp, "sp = combo_set_partitions(3)");
    ASSERT_GT(interp.state().matrices.count("sp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("sp").cols(), 3u);
    expect_contains(interp, "help", "combo_set_partitions(n)");
}
