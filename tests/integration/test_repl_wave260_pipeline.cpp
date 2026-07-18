// MathScript Integration Tests: REPL Interpreter – Wave 260 Pipeline
//
// Smoke: Wave 259 APIs on main (hess/schur, combo_binomial,
// geo_bezier_eval, numthy_factor). Deterministic; Wave 260
// features belong in separate tests once merged.

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

TEST(ReplWave260Pipeline, Hessenberg) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "H = hess(A)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
    expect_contains(interp, "help", "hess(A)");
}

TEST(ReplWave260Pipeline, Schur) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "T = schur(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Ts, Qs = schur(A)");
    ASSERT_GT(interp.state().matrices.count("Ts"), 0u);
    ASSERT_GT(interp.state().matrices.count("Qs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ts").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Qs").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Qs").cols(), 3u);
    expect_contains(interp, "help", "T, Q = schur(A)");
}

TEST(ReplWave260Pipeline, ComboBinomial) {
    Interpreter interp;

    expect_ok(interp, "bin = combo_binomial(5, 2)");
    ASSERT_GT(interp.state().scalars.count("bin"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bin"), 10.0, 1e-9);
    expect_contains(interp, "help", "combo_binomial(n,k)");
}

TEST(ReplWave260Pipeline, GeoBezierEval) {
    Interpreter interp;

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 1.0, 1e-9);
    expect_contains(interp, "help", "geo_bezier_eval(P,t)");
}

TEST(ReplWave260Pipeline, NumthyFactor) {
    Interpreter interp;

    expect_ok(interp, "f = numthy_factor(12)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& factors = interp.state().matrices.at("f");
    EXPECT_EQ(factors.rows(), 3u);
    EXPECT_EQ(factors.cols(), 1u);
    EXPECT_NEAR(factors(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(2, 0), 3.0, 1e-9);

    expect_ok(interp, "p = numthy_factor(17)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 17.0, 1e-9);
    expect_contains(interp, "help", "numthy_factor(n)");
}
