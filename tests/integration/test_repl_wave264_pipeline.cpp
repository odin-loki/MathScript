// MathScript Integration Tests: REPL Interpreter – Wave 264 Pipeline
//
// Wave 264 REPL smoke: info_blahut_arimoto / info_channel_capacity on channel matrices.

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

TEST(ReplWave264Pipeline, InfoBlahutArimotoIdentityChannel) {
    Interpreter interp;

    expect_ok(interp, "W = [1, 0; 0, 1]");
    expect_ok(interp, "c = info_blahut_arimoto(W)");
    ASSERT_GT(interp.state().scalars.count("c"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("c"), std::log2(2.0), 1e-6);
    expect_contains(interp, "help", "info_blahut_arimoto(W)");
}

TEST(ReplWave264Pipeline, InfoChannelCapacityBSC) {
    Interpreter interp;

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "cap = info_channel_capacity(W)");
    ASSERT_GT(interp.state().scalars.count("cap"), 0u);
    const double p = 0.2;
    const double h = -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
    EXPECT_NEAR(interp.state().scalars.at("cap"), 1.0 - h, 1e-4);
    expect_contains(interp, "help", "info_channel_capacity(W)");
}

TEST(ReplWave264Pipeline, InfoChannelCapacityMatchesBlahutArimoto) {
    Interpreter interp;

    expect_ok(interp, "W = [0.9, 0.1; 0.2, 0.8]");
    expect_ok(interp, "ba = info_blahut_arimoto(W)");
    expect_ok(interp, "cc = info_channel_capacity(W)");
    EXPECT_NEAR(interp.state().scalars.at("ba"), interp.state().scalars.at("cc"), 1e-9);
}
