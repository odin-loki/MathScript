// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The new core against the old engine, on random expressions at random points.
//
// §10.5 sets the discipline: each ported function must agree with the one it replaces
// numerically at many random points before the old one is deleted. This file applies
// it to the bridge and the constructors, which are what everything else will be built
// on -- if `from_legacy` and the simplifying constructors change the value of an
// expression, every function ported later inherits that.
//
// The generator is seeded from a constant, so a failure names an expression that can
// be rebuilt exactly. That matters more here than fresh randomness each run: a
// differential failure is only useful if it can be looked at.
//
// Where they are allowed to disagree, and why:
//
//   - The new core is exact where the old one rounded, so the two differ by rounding.
//     The comparison is relative, with an absolute floor for values near zero.
//   - The old engine returns 0.0 for an unbound variable and NaN or infinity for a
//     domain error; the new one reports both. A case where either side is not finite
//     is skipped rather than counted as a disagreement -- but the count of skips is
//     asserted to be a minority, so a bug that made everything non-finite could not
//     pass by skipping the whole corpus.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ms/symbolic/symbolic.hpp"
#include "ms/sym2/bridge.hpp"
#include "ms/sym2/expr.hpp"

using namespace ms;

namespace {

/// A small deterministic generator, so the corpus is the same on every machine and in
/// every run. std::mt19937 would do as well; this is here to make the seed obviously
/// the only source of variation.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed | 1u) {}
    std::uint64_t next() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return state_;
    }
    int below(int bound) { return static_cast<int>(next() % static_cast<std::uint64_t>(bound)); }
    double unit() { return static_cast<double>(next() % 1000000u) / 1000000.0; }

private:
    std::uint64_t state_;
};

const char* const kVariables[] = {"x", "y", "z"};

/// Random legacy expressions of bounded depth. The constants are small and mostly
/// short decimals, which is what a person types and what `from_legacy` is meant to
/// recover exactly; the arguments of sin, cos, exp and log are kept modest so the
/// values stay in a range where a relative comparison means something.
SymExpr random_expr(Rng& rng, int depth) {
    if (depth <= 0 || rng.below(100) < 25) {
        switch (rng.below(4)) {
        case 0:
            return sym_var(kVariables[rng.below(3)]);
        case 1:
            return sym_const(static_cast<double>(rng.below(21) - 10));
        case 2:
            return sym_const(static_cast<double>(rng.below(2001) - 1000) / 100.0);
        default:
            return sym_const(static_cast<double>(rng.below(9) + 1) /
                             static_cast<double>(rng.below(9) + 1));
        }
    }
    switch (rng.below(12)) {
    case 0:
        return sym_add(random_expr(rng, depth - 1), random_expr(rng, depth - 1));
    case 1:
        return sym_sub(random_expr(rng, depth - 1), random_expr(rng, depth - 1));
    case 2:
        return sym_mul(random_expr(rng, depth - 1), random_expr(rng, depth - 1));
    case 3:
        return sym_div(random_expr(rng, depth - 1), random_expr(rng, depth - 1));
    case 4:
        return sym_neg(random_expr(rng, depth - 1));
    case 5:
        return sym_sin(random_expr(rng, depth - 1));
    case 6:
        return sym_cos(random_expr(rng, depth - 1));
    case 7:
        return sym_tan(random_expr(rng, depth - 1));
    case 8:
        return sym_exp(random_expr(rng, depth - 1));
    case 9:
        return sym_log(random_expr(rng, depth - 1));
    case 10:
        return sym_sqrt(random_expr(rng, depth - 1));
    default:
        return sym_pow(random_expr(rng, depth - 1),
                       sym_const(static_cast<double>(rng.below(5) - 2)));
    }
}

bool close_enough(double a, double b) {
    if (a == b) {
        return true;
    }
    const double scale = std::max({std::abs(a), std::abs(b), 1.0});
    return std::abs(a - b) <= 1e-9 * scale;
}

} // namespace

// The core claim: converting into the new representation, with every constructor
// simplifying as it builds, does not change what the expression is worth.
TEST(Sym2Differential, FromLegacyPreservesTheValue) {
    Rng rng(0x5eed1234u);
    int compared = 0;
    int skipped = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const SymExpr legacy = random_expr(rng, 4);
        const sym2::ExprRef converted = sym2::from_legacy(legacy);
        for (int point = 0; point < 3; ++point) {
            const std::map<std::string, double> env{
                {"x", rng.unit() * 4.0 - 2.0},
                {"y", rng.unit() * 4.0 - 2.0},
                {"z", rng.unit() * 4.0 - 2.0},
            };
            const double old_value = sym_eval(legacy, env);
            const auto new_value = sym2::evaluate(converted, env);
            if (!std::isfinite(old_value) || !new_value.has_value() ||
                !std::isfinite(*new_value)) {
                ++skipped;
                continue;
            }
            ++compared;
            ASSERT_TRUE(close_enough(old_value, *new_value))
                << "expression: " << sym_to_string(legacy) << "\n"
                << "converted : " << sym2::to_string(converted) << "\n"
                << "at x=" << env.at("x") << " y=" << env.at("y") << " z=" << env.at("z") << "\n"
                << "old=" << old_value << " new=" << *new_value;
        }
    }
    // A corpus that mostly skipped would prove nothing, so the skip rate is part of
    // the assertion rather than a footnote.
    EXPECT_GT(compared, skipped) << "compared " << compared << ", skipped " << skipped;
    EXPECT_GT(compared, 3000);
}

// Round-tripping through the old representation and back is not an identity -- an
// exact 1/3 cannot survive it -- but it must not change the value beyond the rounding
// that the trip through a double explains.
TEST(Sym2Differential, RoundTripThroughTheOldCoreKeepsTheValue) {
    Rng rng(0xd1ffe12u);
    int compared = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        const SymExpr legacy = random_expr(rng, 3);
        const sym2::ExprRef converted = sym2::from_legacy(legacy);
        bool lossy = false;
        const SymExpr back = sym2::to_legacy(converted, &lossy);
        const sym2::ExprRef again = sym2::from_legacy(back);
        for (int point = 0; point < 2; ++point) {
            const std::map<std::string, double> env{
                {"x", rng.unit() * 4.0 - 2.0},
                {"y", rng.unit() * 4.0 - 2.0},
                {"z", rng.unit() * 4.0 - 2.0},
            };
            const auto first = sym2::evaluate(converted, env);
            const auto second = sym2::evaluate(again, env);
            if (!first.has_value() || !second.has_value() || !std::isfinite(*first) ||
                !std::isfinite(*second)) {
                continue;
            }
            ++compared;
            ASSERT_TRUE(close_enough(*first, *second))
                << "expression: " << sym2::to_string(converted) << "\n"
                << "after round trip: " << sym2::to_string(again) << "\n"
                << "lossy=" << lossy << " first=" << *first << " second=" << *second;
        }
    }
    EXPECT_GT(compared, 1000);
}

// The printer has to be readable, but first it has to be right: what it prints must
// parse back to the same expression. A printer that drops a pair of parentheses turns
// a - (b - c) into a - b - c and is wrong in a way that reads perfectly well.
TEST(Sym2Differential, WhatThePrinterWritesParsesBackToItself) {
    Rng rng(0x9917bcdu);
    int compared = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const sym2::ExprRef original = sym2::from_legacy(random_expr(rng, 3));
        const std::string text = sym2::to_string(original);
        const auto reparsed = sym2::parse(text);
        if (!reparsed.has_value()) {
            // The printer emits `undefined` for what has no value, and the legacy
            // parser has no such word. That is the only case allowed to fail here.
            ASSERT_NE(text.find("undefined"), std::string::npos)
                << "printed but could not parse: " << text;
            continue;
        }
        ++compared;
        // Values first: the shapes may differ, since parsing goes through the legacy
        // core's doubles and 1/3 does not survive that.
        for (int point = 0; point < 2; ++point) {
            const std::map<std::string, double> env{
                {"x", rng.unit() * 4.0 - 2.0},
                {"y", rng.unit() * 4.0 - 2.0},
                {"z", rng.unit() * 4.0 - 2.0},
            };
            const auto a = sym2::evaluate(original, env);
            const auto b = sym2::evaluate(*reparsed, env);
            if (!a.has_value() || !b.has_value() || !std::isfinite(*a) || !std::isfinite(*b)) {
                continue;
            }
            ASSERT_TRUE(close_enough(*a, *b))
                << "printed: " << text << "\n"
                << "reparsed: " << sym2::to_string(*reparsed) << "\n"
                << *a << " vs " << *b;
        }
    }
    EXPECT_GT(compared, 2000);
}

// Canonical form has to be idempotent: rebuilding an expression from its own parts
// must give the same node back, or nothing downstream can rely on pointer equality.
TEST(Sym2Differential, TheCanonicalFormIsAFixedPoint) {
    Rng rng(0xfeed77u);
    for (int trial = 0; trial < 3000; ++trial) {
        const sym2::ExprRef e = sym2::from_legacy(random_expr(rng, 4));
        std::vector<sym2::ExprRef> args(e->args.begin(), e->args.end());
        sym2::ExprRef rebuilt = e;
        switch (sym2::head_of(e)) {
        case sym2::Head::Add:
            rebuilt = sym2::add(args);
            break;
        case sym2::Head::Mul:
            rebuilt = sym2::mul(args);
            break;
        case sym2::Head::Pow:
            rebuilt = sym2::pow(args[0], args[1]);
            break;
        default:
            continue;
        }
        ASSERT_EQ(rebuilt.get(), e.get())
            << "original: " << sym2::to_string(e) << "\n"
            << "rebuilt : " << sym2::to_string(rebuilt);
    }
}
