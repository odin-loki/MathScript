// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4 over `src/interp/repl_engine.cpp`: the argument guards, and the one
// place where the mutant was right.
//
// Nine of twenty-four mutants survived the twenty-three suites that cover this
// file, and eight of them are the same shape as the survivors in
// `repl_engine_internal.cpp`: a guard that names several conditions on one line,
// exercised only by arguments that are far from all of them. Moving any one
// bound, or turning any one `||` into an `&&`, is invisible to a suite whose
// arguments are all comfortably valid.
//
// The ninth is different. `if (steps < 0)` admitted a binomial tree of ZERO
// steps, where `dt` is T/0 and the risk-neutral probability is inf/inf, and the
// number that came back was the undiscounted intrinsic value -- 10 for a put
// struck at 110 on a spot of 100, where one step gives 11.3042. The mutant that
// moved that bound to `steps < 1` was not a mutant, it was the fix.

#include "repl_test_helpers.hpp"

#include <cmath>
#include <string>
#include <vector>

using ms::interp::Interpreter;

TEST(ReplEngineGuards, ATreeWithNoStepsIsNotAPrice) {
    // Zero steps is not a coarse tree, it is no tree. Every one of these
    // returned a number before: the terminal node is S*pow(u,0)*pow(d,0), and
    // `pow(anything, 0)` is 1, so the inf and NaN upstream never reached the
    // answer and the answer was the intrinsic value with neither the rate nor
    // the volatility in it.
    Interpreter interp;
    for (const std::string& cmd : {
             std::string("finance_binomial_put(100, 110, 1, 0.05, 0.2, 0)"),
             std::string("finance_binomial_call(100, 90, 1, 0.05, 0.2, 0)"),
             std::string("finance_american_option(100, 110, 1, 0.05, 0.2, 0, 0)"),
             std::string("finance_trinomial_option(100, 110, 1, 0.05, 0.2, 0, 1, 0)"),
         }) {
        expect_error_contains(interp, cmd, "positive integer");
        expect_error_contains(interp, "v = " + cmd.substr(0), "positive integer");
    }

    // One step is the smallest tree there is, and it has a closed form: with
    // dt = 1, u = e^0.2 and p = (e^0.05 - 1/u)/(u - 1/u), the only node in the
    // money is the down node.
    expect_ok(interp, "p1 = finance_binomial_put(100, 110, 1, 0.05, 0.2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("p1"), 11.304236451473336, 1e-9);
    expect_ok(interp, "c1 = finance_binomial_call(100, 90, 1, 0.05, 0.2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("c1"), 17.655570172853082, 1e-9);

    // And the undiscounted intrinsic value the old code returned is not it.
    EXPECT_GT(interp.state().scalars.at("p1"), 10.0);
}

TEST(ReplEngineGuards, PrimePiRejectsNegativeAndFractionalCountsSeparately) {
    // `arg < 0.0 || std::floor(arg) != arg` -- as `&&`, neither input fires it
    // alone, and -1 reaches `numthy::prime_pi` with a count it cannot have.
    Interpreter interp;
    expect_error_contains(interp, "a = numthy_prime_pi(-1)", "non-negative integer");
    expect_error_contains(interp, "a = numthy_prime_pi(2.5)", "non-negative integer");
    expect_ok(interp, "a = numthy_prime_pi(10)");
    EXPECT_EQ(interp.state().scalars.at("a"), 4.0) << "2, 3, 5, 7";
    expect_ok(interp, "a = numthy_prime_pi(0)");
    EXPECT_EQ(interp.state().scalars.at("a"), 0.0) << "zero is a count, and admissible";
}

TEST(ReplEngineGuards, GegenbauerRejectsNegativeAndFractionalDegreesSeparately) {
    // The same guard on `gegenbauer_c(n, alpha, x)`. A negative degree returned
    // NaN through the no-assignment path, which is how little was looking.
    Interpreter interp;
    expect_error_contains(interp, "g = gegenbauer_c(-1, 0.5, 0.5)", "non-negative integer");
    expect_error_contains(interp, "g = gegenbauer_c(2.5, 0.5, 0.5)", "non-negative integer");

    // C_2^(1/2)(x) is the Legendre polynomial P_2(x) = (3x^2 - 1)/2.
    expect_ok(interp, "g = gegenbauer_c(2, 0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("g"), (3.0 * 0.25 - 1.0) / 2.0, 1e-12);
    expect_ok(interp, "g = gegenbauer_c(0, 0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 1.0, 1e-12) << "degree zero is admissible";
}

TEST(ReplEngineGuards, TheGammaDensityUsesItsFirstArgumentAsTheAbscissa) {
    // `gamma_pdf(args[0], args[1], args[2])` with the first index moved to 1
    // evaluates the density of the SHAPE rather than of x, and still returns a
    // plausible positive number. Nothing had pinned a value.
    // f(2; k=3, theta=1) = x^(k-1) e^(-x/theta) / (Gamma(k) theta^k) = 4 e^-2 / 2.
    Interpreter interp;
    expect_ok(interp, "d = prob_gamma_pdf(2, 3, 1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 4.0 * std::exp(-2.0) / 2.0, 1e-12);
    expect_ok(interp, "d = prob_gamma_pdf(3, 3, 1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 9.0 * std::exp(-3.0) / 2.0, 1e-12);
}

TEST(ReplEngineGuards, ACellularAutomatonRuleIsCheckedAtBothEndsOfTheByte) {
    // `rule < 0 || rule > 255` -- as `&&` no rule is ever out of range, and the
    // rule number indexes an eight-bit table.
    //
    // There are TWO of this guard: the REPL checks the rule before calling, and
    // `eval_gria_settling_time` checks it again. So the outer one's whole
    // contribution is its wording -- "expected INTEGER rule in [0,255]", against
    // the inner "expected rule in [0,255]" -- and a test that asked only for the
    // shorter string passed with the outer guard deleted, because the inner one
    // answered. What is pinned here is therefore the message, which is exactly
    // what that line is for. The first version of this test did not, and the
    // mutant walked through it.
    Interpreter interp;
    expect_ok(interp, "a = [1; 0; 1; 0]");
    expect_ok(interp, "b = [0; 1; 0; 1]");
    expect_error_contains(interp, "s = gria_settling_time(a, b, -1, 10)",
                          "expected integer rule in [0,255]");
    expect_error_contains(interp, "s = gria_settling_time(a, b, 256, 10)",
                          "expected integer rule in [0,255]");
    for (const std::string& rule : {std::string("0"), std::string("255")}) {
        expect_ok(interp, "s = gria_settling_time(a, b, " + rule + ", 10)");
        EXPECT_TRUE(std::isfinite(interp.state().scalars.at("s"))) << "rule " << rule;
    }
}

TEST(ReplEngineGuards, PartialTraceWantsExactlyFourArguments) {
    // `!call_args || call_args->size() != 4` as `&&` is worse than a missing
    // guard: when the split fails it dereferences the failure to ask for a size,
    // and when it succeeds it accepts any arity and reads `(*call_args)[3]` out
    // of a three-element vector.
    Interpreter interp;
    expect_ok(interp, "rho = [0.5, 0; 0, 0.5]");
    expect_error_contains(interp, "t = quantum_partial_trace(rho, 2, 1)",
                          "quantum_partial_trace(rho, d1, d2, subsystem)");
    expect_error_contains(interp, "t = quantum_partial_trace(rho, 2, 1, 0, 1)",
                          "quantum_partial_trace(rho, d1, d2, subsystem)");
    expect_ok(interp, "t = quantum_partial_trace(rho, 2, 1, 0)");
}

TEST(ReplEngineGuards, FixedPointTakesAtMostFourArguments) {
    // `size() < 2 || size() > 4`. The upper bound had never been reached, so it
    // could become 5 and the extra argument would be read as nothing at all.
    Interpreter interp;
    expect_error_contains(interp, "fixed_point(\"cos(x0)\", 0.5, 1e-10, 100, 3)",
                          "fixed_point(\"formula\", x0[, tol[, max_iter]])");
    expect_error_contains(interp, "fixed_point(\"cos(x0)\")",
                          "fixed_point(\"formula\", x0[, tol[, max_iter]])");
    // Four is admissible, and the fixed point of cos is the Dottie number.
    expect_contains(interp, "fixed_point(\"cos(x0)\", 0.5, 1e-10, 100)", "0.739085");
    expect_contains(interp, "fixed_point(\"cos(x0)\", 0.5)", "0.739085");
}

TEST(ReplEngineGuards, AnAmericanOptionRejectsEachNonNumericArgumentOnItsOwn) {
    // Seven `parse_number` calls joined by `||` on one expression: any single
    // `||` turned into an `&&` lets one bad argument through, and the variable
    // it was meant to fill keeps whatever it was initialised to. Each position
    // has to be driven alone; six valid arguments and one bad one is the only
    // input that tells the seven apart.
    Interpreter interp;
    const std::vector<std::string> good{"100", "100", "1", "0.05", "0.2", "1", "10"};
    for (size_t i = 0; i < good.size(); ++i) {
        std::vector<std::string> args = good;
        args[i] = "zz";
        std::string cmd = "finance_american_option(";
        for (size_t j = 0; j < args.size(); ++j) {
            cmd += (j ? "," : "") + args[j];
        }
        cmd += ")";
        expect_error_contains(interp, cmd, "finance_american_option(S,K,T,r,sigma,call,steps)");
    }
    expect_contains(interp, "finance_american_option(100,100,1,0.05,0.2,1,10)", ".");
}

TEST(ReplEngineGuards, TheLookbackMonteCarloRejectsEachNonNumericArgumentOnItsOwn) {
    // The same seven-way guard on the floating-strike lookback estimators.
    Interpreter interp;
    const std::vector<std::string> good{"100", "1", "0.05", "0.2", "8", "4", "7"};
    for (const std::string& fn : {std::string("finance_mc_lookback_floating_call"),
                                  std::string("finance_mc_lookback_floating_put")}) {
        for (size_t i = 0; i < good.size(); ++i) {
            std::vector<std::string> args = good;
            args[i] = "zz";
            std::string cmd = fn + "(";
            for (size_t j = 0; j < args.size(); ++j) {
                cmd += (j ? "," : "") + args[j];
            }
            cmd += ")";
            expect_error_contains(interp, cmd, fn + "(S,T,r,sigma,n_paths,n_steps,seed)");
        }
        expect_contains(interp, fn + "(100,1,0.05,0.2,8,4,7)", ".");
    }
}

TEST(ReplEngineGuards, ABareCallWithAMatrixLiteralIsTheSameCallAsAnAssignedOne) {
    // `execute` reads a bare call through twelve regexes whose argument groups
    // are `[^,]+`, so an inline matrix literal was cut at its first comma and
    // the command failed with `unknown matrix: [1` -- while the SAME call with
    // an `x = ` in front of it worked, because the assignment path splits on
    // brackets. The retry hands such a line to the reading that knows how.
    Interpreter interp;
    expect_ok(interp, "x = [1, 0, 0, 0]");
    expect_ok(interp, "z = signal_czt_zoom(x, 0, 1, 1, 4)");
    const auto assigned = interp.execute("z");
    ASSERT_TRUE(assigned.has_value());

    const auto bare = interp.execute("signal_czt_zoom([1, 0, 0, 0], 0, 1, 1, 4)");
    ASSERT_TRUE(bare.has_value()) << "a bare call is the same call";
    // The chirp-z transform of a unit impulse is 1 at every bin, either way.
    EXPECT_NE(bare->find("1.000000, 0.000000"), std::string::npos) << *bare;

    // And when the arguments are wrong, the message is about the arguments and
    // not about the fragment the regex cut off.
    expect_error_contains(interp, "signal_czt_zoom([1, 0, 0, 0], 0, 1, 0, 4)",
                          "positive integer m");
}

TEST(ReplEngineGuards, TheRetryDoesNotChangeWhatAlreadyWorkedOrWhatAlreadyFailed) {
    // The retry runs only after the direct reading has failed, so nothing that
    // answers today answers differently -- including the labels bare calls
    // print, which are per-command and not `_`.
    Interpreter interp;
    expect_contains(interp, "stats_mean([1, 2, 3])", "2");
    expect_contains(interp, "signal_upsample([1, 2], 2)", "upsampled");
    expect_contains(interp, "quantum_partial_trace([0.5, 0; 0, 0.5], 2, 1, 0)", "rho");

    // A callee nothing knows still reports the reading of the line as typed,
    // rather than whatever the retry made of it.
    expect_error_contains(interp, "no_such_fn([1, 2], 3)", "could not read");

    // One line typed is one line of history, whether or not it was retried.
    const std::size_t before = interp.state().history.size();
    expect_ok(interp, "signal_czt_zoom([1, 0, 0, 0], 0, 1, 1, 4)");
    EXPECT_EQ(interp.state().history.size(), before + 1);
    EXPECT_EQ(interp.state().history.back(), "signal_czt_zoom([1, 0, 0, 0], 0, 1, 1, 4)");
}
