// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The second audit's findings, at the REPL surface a user actually types at.
//
// A read-only sweep along eight dimensions produced 40 claims; each went to an
// independent verifier told to refute it, and 36 survived. The unit-level fixes are
// tested next to the code they fix; these are the ones whose whole point is what the
// REPL says back, and every one of them is a case that used to print an answer.
//
// The numbers here were computed outside this program.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

// sym_eval returns 0.0 for a name it was not given, and 0.0 is a perfectly good answer,
// so `sym_eval("x*y", "x=3")` printed 0.000000. It accepts one binding, so y could never
// have been bound: the expression the user wrote was never the expression evaluated.
TEST(ReplAuditFixes, AFormulaCannotUseAVariableTheCallerWillNotBind) {
    Interpreter interp;
    expect_error_contains(interp, "sym_eval(\"x*y\", \"x=3\")", "unknown variable");
    expect_contains(interp, "sym_eval(\"x*x\", \"x=3\")", "9");

    // The optimisers bind x0, x1, ...; an objective written in x is the constant f(0).
    // bfgs("(x-3)^2", [0]) used to report converged = 1 at x_opt = 0 with f_val = 9,
    // which is the value of the constant it was actually minimising.
    expect_error_contains(interp, "bfgs(\"(x-3)^2\", [0])", "unknown variable");
    expect_contains(interp, "bfgs(\"(x0-3)^2\", [0])", "x_opt");

    // A root finder given the wrong name is looking for the root of a constant.
    expect_error_contains(interp, "bisection(\"x^2 - 2\", 0, 2)", "unknown variable");
}

// The five optimisers that hard-coded converged = true and iterations = max_iter.
TEST(ReplAuditFixes, OptimisersReportWhatTheyActuallyDid) {
    Interpreter interp;
    // One step from a thousand away is not convergence.
    expect_contains(interp, "adam(\"(x0-1000)^2\", [0], 0.001, 1)", "converged = 0");
    // Starting at the optimum, the gradient test fires on the first pass -- and the
    // count reported is 1, not the 1000 it was allowed.
    const auto at_optimum = interp.execute("adam(\"(x0-1)^2\", [1], 0.001, 1000)");
    ASSERT_TRUE(at_optimum.has_value());
    EXPECT_NE(at_optimum->find("converged = 1"), std::string::npos) << *at_optimum;
    EXPECT_NE(at_optimum->find("iterations = 1\n"), std::string::npos) << *at_optimum;
    expect_contains(interp, "nelder_mead(\"(x0-1000)^2\", [0], 1e-12, 5)", "converged = 0");
}

// The adaptive solvers stop after 50000 attempted steps. Returning the partial
// trajectory unmarked made it read as a solution over the whole interval:
// ode_rk45("cos(1000*t)", 0, 0, 100, ...) covered [0, 17.45] and printed 33330 rows.
TEST(ReplAuditFixes, AnIncompleteIntegrationSaysSo) {
    Interpreter interp;
    expect_error_contains(interp, "ode_rk45(\"cos(1000*t)\", 0, 0, 100, 1e-6, 1e-9)",
                          "step budget exhausted");
    // An ordinary problem is unaffected.
    expect_contains(interp, "ode_rk45(\"-y\", 0, 1, 1, 1e-6, 1e-9)", "traj =");
}

// 1/x has no limit at 0. The two probes are -1/h and +1/h, whose average is exactly 0
// at every step, so the samples converged perfectly on a value neither side approaches.
TEST(ReplAuditFixes, ATwoSidedDivergenceIsNotAveragedIntoAnAnswer) {
    Interpreter interp;
    for (const char* command : {"sym_limit(\"1/x\", \"x\", 0)",
                                "sym_limit(\"1/x^3\", \"x\", 0)",
                                "sym_limit(\"1/sin(x)\", \"x\", 0)"}) {
        const auto result = interp.execute(command);
        if (result.has_value()) {
            EXPECT_EQ(result->find("0.000000"), std::string::npos) << command << " -> " << *result;
        }
    }
    // The limits that do exist still do.
    expect_contains(interp, "sym_limit(\"sin(x)/x\", \"x\", 0)", "1");
    expect_contains(interp, "sym_limit(\"(1-cos(x))/x^2\", \"x\", 0)", "0.5");
}

// M{e^{-a t}}(s) is Gamma(s)/a^s. The table answered 1/a^s -- right only at s = 1 --
// under the same convention its rational and log rows are correct in.
TEST(ReplAuditFixes, TheMellinRowsThatDroppedGammaDecline) {
    Interpreter interp;
    expect_error_contains(interp, "sym_mellin(\"exp(-2*t)\", \"t\", \"s\")", "no closed form");
    expect_error_contains(interp, "sym_mellin(\"t^2*exp(-2*t)\", \"t\", \"s\")",
                          "no closed form");
    // The rows that are right are untouched.
    expect_contains(interp, "sym_mellin(\"1/(1+t)\", \"t\", \"s\")", "sin");
}

// bellman_ford exists for negative weights and never saw one: graph_from_adjacency
// treated anything not > 0 as no edge, for every caller.
TEST(ReplAuditFixes, BellmanFordSeesNegativeWeights) {
    Interpreter interp;
    // 0 -> 1 costs 4, 0 -> 2 costs 5, 2 -> 1 costs -3, so the best route to 1 is 2.
    expect_ok(interp, "A = [0, 4, 5; 0, 0, 0; 0, -3, 0]");
    const auto result = interp.execute("d = graph_bellman_ford(A, 0)");
    ASSERT_TRUE(result.has_value()) << ms::format_error(result.error());
    ASSERT_GT(interp.state().matrices.count("d"), 0U);
    // One row per vertex: (distance from the source, predecessor).
    const auto& d = interp.state().matrices.at("d");
    ASSERT_EQ(d.rows(), 3U);
    EXPECT_NEAR(d(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(d(1, 0), 2.0, 1e-9) << "vertex 1 is reached through 2 for 5 - 3 = 2";
    EXPECT_NEAR(d(1, 1), 2.0, 1e-9) << "...so its predecessor is vertex 2";
    EXPECT_NEAR(d(2, 0), 5.0, 1e-9);
}

// matrix_to_bytes multiplied by 255 whenever the largest entry was <= 1.0, so a legal
// byte vector came back from a round trip as different data.
TEST(ReplAuditFixes, CompressRoundTripsReturnWhatTheyWereGiven) {
    Interpreter interp;
    expect_ok(interp, "E = rle_encode_vec([0; 1])");
    expect_ok(interp, "D = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("D"), 0U);
    const auto& d = interp.state().matrices.at("D");
    ASSERT_EQ(d.rows(), 2U);
    EXPECT_NEAR(d(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(d(1, 0), 1.0, 1e-12) << "1 came back as 255";
    // And something that is not a byte is reported rather than clamped.
    expect_error_contains(interp, "F = rle_encode_vec([0; 300])", "byte values");
    expect_error_contains(interp, "F = rle_encode_vec([0; 0.5])", "byte values");
}

// BigInt's string constructor turns anything unparsable into zero.
TEST(ReplAuditFixes, ABadBigIntLiteralIsReported) {
    Interpreter interp;
    expect_error_contains(interp, "x = bigint(\"495.0\")", "invalid decimal literal");
    // The bare form had no reading at all, so the same literal came back "could not
    // read 'bigint(\"495.0\")' as a matrix call, a matrix constructor, or a scalar
    // expression" -- true, and useless when what is wrong is one character. It reaches
    // the same reporting parse the assignment form does now, and a valid literal
    // prints instead of failing to parse.
    expect_error_contains(interp, "bigint(\"495.0\")", "invalid decimal literal: 495.0");
    expect_error_contains(interp, "bigint(\"1e3\")", "invalid decimal literal: 1e3");
    expect_contains(interp, "bigint(\"495\")", "495");
    expect_contains(interp, "bigint('495')", "495");
    expect_error_contains(interp, "x = bigint(\"1e3\")", "invalid decimal literal");
    expect_ok(interp, "x = bigint(\"495\")");
    EXPECT_NEAR(interp.state().scalars.at("x"), 495.0, 1e-9);
}

// The assignment target was never checked, so this stored a variable named "A(1, 1)"
// that no lookup can ever spell back, and echoed it as a successful element write.
TEST(ReplAuditFixes, AnElementAssignmentIsRefusedRatherThanDiscarded) {
    Interpreter interp;
    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_error_contains(interp, "A(1, 1) = 9", "invalid variable name");
    ASSERT_GT(interp.state().matrices.count("A"), 0U);
    EXPECT_NEAR(interp.state().matrices.at("A")(1, 1), 4.0, 1e-12);
    EXPECT_EQ(interp.state().scalars.count("A(1, 1)"), 0U);
    // Multi-target assignment is still a valid target list.
    expect_ok(interp, "L, U = lu(A)");
}

// Scalars and matrices live in separate maps and neither store cleared the other.
TEST(ReplAuditFixes, ANameHoldsOneValue) {
    Interpreter interp;
    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "A = 5");
    EXPECT_EQ(interp.state().matrices.count("A"), 0U);
    expect_contains(interp, "A", "5");

    expect_ok(interp, "B = 7");
    expect_ok(interp, "B = [9, 9]");
    EXPECT_EQ(interp.state().scalars.count("B"), 0U);
}

// Every call built a fresh mt19937 from the constant 0.
TEST(ReplAuditFixes, SuccessiveRandomMatricesAreDifferentDraws) {
    Interpreter interp;
    expect_ok(interp, "A = rand(2, 2)");
    expect_ok(interp, "B = rand(2, 2)");
    const auto& a = interp.state().matrices.at("A");
    const auto& b = interp.state().matrices.at("B");
    bool any_different = false;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            any_different = any_different || a(i, j) != b(i, j);
        }
    }
    EXPECT_TRUE(any_different) << "two rand(2,2) calls returned the same matrix";

    // ...and a session still replays: a second interpreter run the same way agrees.
    Interpreter replay;
    expect_ok(replay, "A = rand(2, 2)");
    EXPECT_EQ(replay.state().matrices.at("A")(0, 0), a(0, 0));
}

// The last group still streaming into a default ostringstream: six SIGNIFICANT digits,
// which is far coarser than the six decimal places std::to_string gives.
TEST(ReplAuditFixes, TheLastSixDigitPrintersAreGone) {
    Interpreter interp;
    // gamma(20) is 121645100408832000 exactly (19!), printed as 1.21645e+17.
    const auto gamma = interp.execute("gamma(20)");
    ASSERT_TRUE(gamma.has_value());
    // 19! is 121645100408832000. The printed value has to be that number to within the
    // accuracy of the gamma implementation -- what used to come back was
    // "1.21645e+17", which reads as 121645000000000000: short by 100,408,832,000.
    const double printed = std::stod(*gamma);
    EXPECT_NEAR(printed / 121645100408832000.0, 1.0, 1e-12) << *gamma;
    EXPECT_GT(std::abs(printed - 121645000000000000.0), 1e8) << *gamma;
    // Every fft bin of a single non-zero sample has magnitude 1234567 exactly.
    expect_contains(interp, "fft([1234567, 0, 0, 0])", "1234567");
    const auto spectrum = interp.execute("fft([1234567, 0, 0, 0])");
    ASSERT_TRUE(spectrum.has_value());
    EXPECT_EQ(spectrum->find("1.23457e+06"), std::string::npos) << *spectrum;
}

// The degenerate ANOVA returned its zero-initialised f_stat alongside p = 0, a pair no
// F-test can produce and one that reads as a confident null result.
TEST(ReplAuditFixes, AnUndefinedAnovaIsReported) {
    Interpreter interp;
    // Two groups, no within-group variation at all.
    expect_error_contains(interp, "b = stats_one_way_anova([1, 1, 1; 2, 2, 2])",
                          "not defined");
    EXPECT_EQ(interp.state().matrices.count("b"), 0U);
    // An ordinary input still works.
    expect_ok(interp, "a = stats_one_way_anova([1, 2, 3; 4, 5, 7])");
    EXPECT_GT(interp.state().matrices.at("a")(0, 0), 0.0);
}

// The result of a matrix call written without a target is stored under `_`, so it can
// be used on the next line. Five callees -- matmul, tensorops_matmul,
// tensorops_einsum, signal_conv2, ml_mat_mul and dist_matmul -- had hand-written
// branches predating the registry fallback, and those printed under an invented `C`
// and stored nothing: the value could be read and not used, while `rand(2, 2)` on the
// next line could be both.
TEST(ReplAuditFixes, ABareMatrixCallLeavesItsResultWhereItSaysItIs) {
    Interpreter interp;
    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_contains(interp, "matmul(A, A)", "_ =");
    ASSERT_GT(interp.state().matrices.count("_"), 0U)
        << "printed a result under a name that holds nothing";
    const auto& product = interp.state().matrices.at("_");
    EXPECT_NEAR(product(0, 0), 7.0, 1e-12);
    EXPECT_NEAR(product(1, 1), 22.0, 1e-12);
    // And it is usable, which is the whole point of naming it something.
    expect_ok(interp, "B = matmul(_, A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(0, 0), 37.0, 1e-12);
}

} // namespace
