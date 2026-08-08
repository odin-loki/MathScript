// MathScript Integration Tests: REPL Interpreter – Wave 296 Pipeline
//
// Wave 296 REPL smoke: combo/poly tail11 extensions, hypergeo_0f1n/1f1n scalar.

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

TEST(ReplWave296Pipeline, ComboPoly) {
    Interpreter interp;

    expect_contains(interp, "help", "combo_lyndon_words");
    expect_contains(interp, "help", "poly_squarefree");

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 2u);

    expect_ok(interp, "mp = combo_motzkin_paths(2)");
    EXPECT_GT(interp.state().matrices.at("mp").rows(), 0u);

    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    EXPECT_GT(interp.state().matrices.at("sf").rows(), 0u);
}

TEST(ReplWave296Pipeline, SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "h0 = hypergeo_0f1n(2, 1.5, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1n(2, 1.5, 0.2), 1e-9);

    expect_ok(interp, "h1 = hypergeo_1f1n(1, 0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1n(1, 0.5, 0.3), 1e-9);
}
