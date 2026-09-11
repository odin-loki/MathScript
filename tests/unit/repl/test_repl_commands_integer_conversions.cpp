// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Arguments that were converted out of range, and answered anyway.
//
// Eighty-odd REPL commands hand a `double` to a `uint64_t` or an `unsigned` parameter.
// The guard in front of every one of them read
//
//     if (arg < 0.0 || std::floor(arg) != arg) { /* reject */ }
//     ... static_cast<uint64_t>(arg) ...
//
// which rejects negatives and fractions and then converts whatever is left. A double
// outside `[0, 2^64)` has no `uint64_t` to convert to; the standard calls that
// undefined, and what it looked like from the prompt was an answer:
//
//     numthy_gcd(1e300, 18)               18
//     numthy_lcm(1e300, 3)                0
//     numthy_num_divisors(1e300)          1
//     numthy_euler_phi(1e300)             0
//     numthy_sum_divisors(18446744073709551615)   0
//
// The last one is the sharpest: the literal is not representable as a double and rounds
// UP to exactly 2^64, one past the last value a `uint64_t` holds, so the conversion had
// nothing to return and sigma was reported as zero.
//
// The seeds are worse than wrong, because two of them agreed:
//
//     finance_mc_european_call(100,100,1,0.05,0.2,1000,42)           10.799620
//     finance_mc_european_call(100,100,1,0.05,0.2,1000,4294967296)   10.757478
//     finance_mc_european_call(100,100,1,0.05,0.2,1000,1e300)        10.757478
//
// Somebody varying the seed to see the Monte Carlo spread would have been reading one
// sample twice and calling it two.

#include <string>

#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"

#include "repl/repl_test_helpers.hpp"

using ms::interp::Interpreter;

TEST(ReplIntegerConversions, AnArgumentPastTheRangeIsRefusedRatherThanConverted) {
    Interpreter interp;
    for (const auto* call : {"numthy_gcd(1e300, 18)", "numthy_lcm(1e300, 3)",
                             "numthy_num_divisors(1e300)", "numthy_euler_phi(1e300)",
                             "numthy_isprime(1e300)", "numthy_mobius(1e300)",
                             "numthy_factor_count(1e300)", "numthy_sum_divisors(1e300)",
                             "numthy_nextprime(1e300)", "numthy_prime_pi(1e300)"}) {
        expect_error_contains(interp, call, "is too large");
    }
    // The named argument comes from the signature the REPL prints, so the message points
    // at something the user can see in what they typed.
    expect_error_contains(interp, "numthy_gcd(1e300, 18)", "a 1");
    expect_error_contains(interp, "numthy_mod_pow(1e300, 2, 7)", "base 1");
}

TEST(ReplIntegerConversions, TheBoundaryIsTwoToTheSixtyFourAndNotTheLiteral) {
    Interpreter interp;
    // 18446744073709551615 is 2^64-1, which no double represents: it rounds up to 2^64,
    // which is one past the end. It used to answer 0.
    expect_error_contains(interp, "numthy_sum_divisors(18446744073709551615)", "is too large");
    // The largest double that IS a uint64_t is 2^64 - 2048, and it is accepted.
    expect_ok(interp, "numthy_num_divisors(18446744073709549568)");
}

TEST(ReplIntegerConversions, AnArgumentInsideTheRangeStillAnswers) {
    // A guard that refuses everything is not a fix: these are the same calls with values
    // the destination holds, and their answers are the arithmetic ones.
    Interpreter interp;
    expect_contains(interp, "numthy_gcd(12, 18)", "6");
    expect_contains(interp, "numthy_lcm(4, 6)", "12");
    expect_contains(interp, "numthy_num_divisors(28)", "6");
    expect_contains(interp, "numthy_euler_phi(9)", "6");
    // sigma(28) = 1+2+4+7+14+28 = 56, and 28 is perfect, so sigma(28) == 2*28.
    expect_contains(interp, "numthy_sum_divisors(28)", "56");
    expect_contains(interp, "numthy_mod_pow(2, 10, 1000)", "24");
}

TEST(ReplIntegerConversions, TwoDifferentSeedsAreTwoDifferentSeeds) {
    Interpreter interp;
    // Out of an `unsigned`'s range is refused rather than folded onto some other seed.
    expect_error_contains(interp, "finance_mc_european_call(100,100,1,0.05,0.2,1000,4294967296)",
                          "a seed is bounded at 4294967295");
    expect_error_contains(interp, "finance_mc_european_call(100,100,1,0.05,0.2,1000,1e300)",
                          "a seed is bounded at 4294967295");
    // And the largest seed there is, is still a seed.
    expect_ok(interp, "finance_mc_european_call(100,100,1,0.05,0.2,1000,4294967295)");
}

TEST(ReplIntegerConversions, TheNthPrimeIsSievedRatherThanCountedTo) {
    Interpreter interp;
    // The loop this replaced ran one Miller-Rabin test per prime: prime_nth(1000000) took
    // 4.3 s and prime_nth(3000000000) was still going at half a minute. It was one of
    // two entries on the malformed sweep's proportional-cost exclusion list.
    expect_contains(interp, "numthy_prime_nth(1)", "2");
    expect_contains(interp, "numthy_prime_nth(2)", "3");
    expect_contains(interp, "numthy_prime_nth(5)", "11");
    // p_6 = 13 is the first one the Rosser-Schoenfeld bound covers, so it is the first
    // that goes through the sieve rather than the listed head.
    expect_contains(interp, "numthy_prime_nth(6)", "13");
    expect_contains(interp, "numthy_prime_nth(100)", "541");
    expect_contains(interp, "numthy_prime_nth(10000)", "104729");
    // Past what a sieve of kMaxSieveSpan can hold there is no answer, and saying so is
    // different from saying the answer does not fit in 64 bits -- which is what the REPL
    // used to say about pi(3000000000), a number near 1.4e8.
    expect_error_contains(interp, "numthy_prime_nth(3000000000)", "past what this can sieve");
    expect_error_contains(interp, "numthy_prime_pi(3000000000)", "past what this can sieve");
}

TEST(ReplIntegerConversions, SigmaComesOutOfTheFactorisation) {
    Interpreter interp;
    // sum_divisors used to build the divisor list by trial division to sqrt(n), which is
    // 1e9 iterations at n = 1e18 and was the other proportional-cost exclusion. sigma is
    // multiplicative, so the exponents are enough -- and the answers have to agree.
    expect_contains(interp, "numthy_sum_divisors(1)", "1");
    expect_contains(interp, "numthy_sum_divisors(6)", "12");
    expect_contains(interp, "numthy_sum_divisors(12)", "28");
    expect_contains(interp, "numthy_sum_divisors(496)", "992");
    // A prime p has sigma(p) = p + 1.
    expect_contains(interp, "numthy_sum_divisors(9973)", "9974");
    // And a prime power: sigma(2^10) = 2^11 - 1 = 2047.
    expect_contains(interp, "numthy_sum_divisors(1024)", "2047");
}
