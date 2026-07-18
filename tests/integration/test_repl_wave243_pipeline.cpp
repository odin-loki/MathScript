// MathScript Integration Tests: REPL Interpreter – Wave 243 Bindings Pipeline
//
// Wired: qmr, lsqr, signal_instantaneous_phase.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/signal/signal.hpp"

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave243Pipeline, QmrBinding) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("x")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("x")(1, 0)));

    const ms::Matrix<double> A{{3.0, 1.0}, {1.0, 2.0}};
    const ms::Matrix<double> b{{1.0}, {1.0}};
    const auto expected = ms::qmr(A, b);
    ASSERT_TRUE(expected.has_value());
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), (*expected)(0, 0), 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), (*expected)(1, 0), 1e-6);

    expect_error(interp, "qmr(A)");
    expect_contains(interp, "help", "qmr");
}

TEST(ReplWave243Pipeline, LsqrBinding) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2; 1, 0]");
    expect_ok(interp, "b = [1; 1; 1]");
    expect_ok(interp, "x = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("x")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("x")(1, 0)));

    expect_error(interp, "lsqr(A)");
    expect_contains(interp, "help", "lsqr");
}

TEST(ReplWave243Pipeline, SignalInstantaneousPhaseBinding) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    const auto& phase = interp.state().matrices.at("ph");
    EXPECT_EQ(phase.cols(), 1u);
    EXPECT_EQ(phase.rows(), 8u);
    for (size_t i = 0; i < phase.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(phase(i, 0)));
    }

    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const auto expected = ms::instantaneous_phase(x);
    ASSERT_EQ(expected.size(), phase.rows());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(phase(i, 0), expected[i], 1e-10);
    }

    expect_error(interp, "signal_instantaneous_phase()");
    expect_contains(interp, "help", "signal_instantaneous_phase");
}
