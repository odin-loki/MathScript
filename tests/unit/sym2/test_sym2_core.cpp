// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The §10 core, tested on the properties the old representation could not have.
//
// Each of these is a direct answer to something in §10.1. They are written as
// properties rather than as pinned strings wherever a property is available, because a
// canonical form has to hold for every input and not only for the ones someone thought
// to write down.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/sym2/expr.hpp"

using ms::bignum::BigInt;
using namespace ms::sym2;

namespace {

ExprRef frac(long long numerator, long long denominator) {
    return rational(BigInt(numerator), BigInt(denominator));
}

// §10.1: "A CAS that cannot represent 1/3 is not a CAS." The old core stored every
// number as a double, so sym_simplify(x/3*3) could not return x -- it returned
// x*0.333333*3, which is x*0.999999.
TEST(Sym2Core, ExactArithmeticMakesCancellationExact) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(mul({div(x, integer(3)), integer(3)}), x);
    EXPECT_EQ(add({frac(1, 3), frac(1, 6)}), frac(1, 2));
    EXPECT_EQ(add({frac(1, 3), frac(2, 3)}), integer(1));
    // 1/3 + 1/3 + 1/3 is exactly 1, which no chain of doubles gives.
    EXPECT_EQ(add({frac(1, 3), frac(1, 3), frac(1, 3)}), integer(1));
    EXPECT_EQ(mul({frac(2, 3), frac(3, 2)}), integer(1));
    // A fraction that reduces to an integer is an integer, not a Rational with a 1.
    EXPECT_EQ(head_of(frac(4, 2)), Head::Integer);
    EXPECT_EQ(frac(4, 2), integer(2));
}

TEST(Sym2Core, ExactArithmeticIsUnbounded) {
    // 2^200 has no double. It has a BigInt.
    const ExprRef big = pow(integer(2), integer(200));
    EXPECT_EQ(head_of(big), Head::Integer);
    EXPECT_EQ(to_string(big),
              "1606938044258990275541962092341162602522202993782792835301376");
    // And it divides back out exactly.
    EXPECT_EQ(div(big, pow(integer(2), integer(199))), integer(2));
}

// §10.1: a+b+c parsed as (a+b)+c, so term collection and structural equality both
// fought the tree shape. n-ary sorted arguments make commutativity a property of the
// representation rather than something every caller has to normalise for.
TEST(Sym2Core, SumsAndProductsAreCanonical) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    const ExprRef z = symbol("z");
    EXPECT_EQ(add({x, y}), add({y, x}));
    EXPECT_EQ(add({x, y, z}), add({z, x, y}));
    EXPECT_EQ(mul({x, y, z}), mul({y, z, x}));
    // Associativity too: the nesting is flattened away.
    EXPECT_EQ(add({add({x, y}), z}), add({x, add({y, z})}));
    EXPECT_EQ(mul({mul({x, y}), z}), mul({x, mul({y, z})}));
}

TEST(Sym2Core, LikeTermsCollectOnConstruction) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(add({x, x}), mul({integer(2), x}));
    EXPECT_EQ(add({x, x, x}), mul({integer(3), x}));
    EXPECT_EQ(add({x, neg(x)}), integer(0));
    EXPECT_EQ(mul({x, x}), pow(x, integer(2)));
    EXPECT_EQ(mul({x, div(integer(1), x)}), integer(1));
    EXPECT_EQ(add({function("sin", {x}), function("sin", {x})}),
              mul({integer(2), function("sin", {x})}));
    // Collection sees through a coefficient.
    EXPECT_EQ(add({mul({integer(2), x}), mul({integer(3), x})}), mul({integer(5), x}));
}

// Interning: equal expressions are the same pointer, which is what makes structural
// equality O(1) and what bounds the memory of an expansion.
TEST(Sym2Core, EqualExpressionsShareOneNode) {
    const ExprRef a = add({mul({integer(2), symbol("x")}), integer(1)});
    const ExprRef b = add({integer(1), mul({symbol("x"), integer(2)})});
    EXPECT_EQ(a.get(), b.get());
    EXPECT_TRUE(structurally_equal(a, b));
    EXPECT_EQ(compare(a, b), 0);
    // ...and unequal ones are not.
    const ExprRef c = add({mul({integer(2), symbol("x")}), integer(2)});
    EXPECT_NE(a.get(), c.get());
    EXPECT_NE(compare(a, c), 0);
}

// compare() is what Add and Mul sort by, so it has to be a strict weak ordering. A
// sort with an inconsistent comparator is not a slow sort, it is undefined behaviour.
TEST(Sym2Core, CompareIsATotalOrder) {
    const std::vector<ExprRef> values{
        integer(-2),          integer(0),           integer(7),
        frac(1, 2),           real(0.5),            real(-0.0),
        symbol("a"),          symbol("b"),          constant("pi"),
        add({symbol("a"), symbol("b")}),            mul({integer(2), symbol("a")}),
        pow(symbol("a"), integer(2)),               function("sin", {symbol("a")}),
        function("sin", {symbol("b")}),             function("atan2", {symbol("a"), symbol("b")}),
        undefined(),
    };
    for (const ExprRef& a : values) {
        EXPECT_EQ(compare(a, a), 0) << to_string(a);
        for (const ExprRef& b : values) {
            EXPECT_EQ(compare(a, b), -compare(b, a)) << to_string(a) << " vs " << to_string(b);
            for (const ExprRef& c : values) {
                if (compare(a, b) < 0 && compare(b, c) < 0) {
                    EXPECT_LT(compare(a, c), 0)
                        << to_string(a) << " < " << to_string(b) << " < " << to_string(c);
                }
            }
        }
    }
}

// §10.3: the old printer parenthesised every node and printed every number with
// std::to_string, so 2*x + 1 came out as ((2.000000 * x) + 1.000000). Every output
// format in §11 would have inherited both faults.
TEST(Sym2Core, ThePrinterIsPrecedenceAware) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    EXPECT_EQ(to_string(add({mul({integer(2), x}), integer(1)})), "2*x + 1");
    EXPECT_EQ(to_string(mul({add({x, integer(1)}), y})), "(x + 1)*y");
    EXPECT_EQ(to_string(pow(add({x, integer(1)}), integer(2))), "(x + 1)^2");
    EXPECT_EQ(to_string(div(add({x, integer(1)}), integer(2))), "(x + 1)/2");
    EXPECT_EQ(to_string(sub(x, y)), "x - y");
    EXPECT_EQ(to_string(neg(x)), "-x");
    EXPECT_EQ(to_string(add({x, mul({integer(-3), y})})), "x - 3*y");
    EXPECT_EQ(to_string(function("sin", {mul({integer(2), x})})), "sin(2*x)");
    EXPECT_EQ(to_string(integer(1024)), "1024");
    EXPECT_EQ(to_string(frac(1, 8)), "1/8");
    EXPECT_EQ(to_string(pow(x, integer(-1))), "1/x");
    // The exponent binds tighter than a product on the left and is right-associative.
    EXPECT_EQ(to_string(mul({integer(2), pow(x, integer(3))})), "2*x^3");
    EXPECT_EQ(to_string(pow(x, pow(y, integer(2)))), "x^y^2");
}

// §10.4's automatic simplification is only allowed to apply identities. These are the
// rewrites that look like identities and are not.
TEST(Sym2Core, UnsoundRewritesAreNotApplied) {
    const ExprRef x = symbol("x");
    // (x^2)^(1/2) is |x|. At x = -2 the two answers differ by the whole answer, so the
    // expression stays as it is.
    const ExprRef root = pow(pow(x, integer(2)), frac(1, 2));
    EXPECT_EQ(head_of(root), Head::Pow);
    EXPECT_EQ(head_of(root->args[0]), Head::Pow);
    EXPECT_NE(root, x);
    // The two cases that ARE identities do fold: an integer outer exponent, and a
    // positive numeric base.
    EXPECT_EQ(pow(pow(x, frac(1, 2)), integer(2)), x);
    EXPECT_EQ(pow(pow(integer(4), frac(1, 2)), integer(2)), integer(4));
}

TEST(Sym2Core, WhatHasNoValueSaysSo) {
    const ExprRef x = symbol("x");
    // The old core had no way to say "undefined": division by zero produced a node
    // that looked like an expression and evaluated to inf or NaN much later.
    EXPECT_TRUE(is_undefined(div(x, integer(0))));
    EXPECT_TRUE(is_undefined(div(integer(1), integer(0))));
    EXPECT_TRUE(is_undefined(rational(BigInt(1LL), BigInt(0LL))));
    // 0^0 has no value that every branch of mathematics agrees on, so it is not given
    // one silently.
    EXPECT_TRUE(is_undefined(pow(integer(0), integer(0))));
    EXPECT_TRUE(is_undefined(pow(integer(0), integer(-1))));
    // The two above are the BUILDER folding a literal, which is a different code path
    // from `evaluate` and does not exercise it: the fold means no Pow node with a zero
    // base ever reaches the evaluator that way. §8.4 found the evaluator's own guard
    // unasserted -- a mutant that widened `exponent < 0` to `<= 0`, turning 0^0 into a
    // division-by-zero error, survived all eight sym2 suites.
    const ExprRef general = pow(symbol("b"), symbol("e"));
    const auto zero_to_zero = evaluate(general, {{"b", 0.0}, {"e", 0.0}});
    ASSERT_TRUE(zero_to_zero.has_value()) << "0^0 reported an error at evaluation time";
    EXPECT_DOUBLE_EQ(*zero_to_zero, 1.0)
        << "the evaluator follows std::pow, which is 1; the builder's `undefined` is a "
           "statement about the SYMBOL 0^0, not about the limit of b^e at the origin";
    // A negative exponent over a zero base is the case the guard is actually for.
    EXPECT_FALSE(evaluate(general, {{"b", 0.0}, {"e", -1.0}}).has_value());
    // ...and undefined is contagious rather than being swallowed by a zero factor.
    EXPECT_TRUE(is_undefined(mul({integer(0), div(x, integer(0))})));
    EXPECT_TRUE(is_undefined(add({integer(1), div(x, integer(0))})));
}

// §10.2: nine functions returned a valid-looking SymExpr to signal failure, and the
// caller had to know the convention. An unevaluated head is the answer instead.
TEST(Sym2Core, UnevaluatedHeadsAreRepresentable) {
    const ExprRef x = symbol("x");
    const ExprRef d = derivative(function("f", {x}), {x});
    EXPECT_EQ(head_of(d), Head::Derivative);
    EXPECT_EQ(to_string(d), "d/dx(f(x))");
    const ExprRef i = integral(function("exp", {mul({x, x})}), {x});
    EXPECT_EQ(head_of(i), Head::Integral);
    EXPECT_EQ(to_string(i), "integral(exp(x^2), x)");
    const ExprRef l = limit(div(function("sin", {x}), x), x, integer(0));
    EXPECT_EQ(head_of(l), Head::Limit);
}

TEST(Sym2Core, EvaluationReportsWhatItCannotDo) {
    const ExprRef x = symbol("x");
    const ExprRef e = add({mul({integer(2), x}), integer(1)});
    const auto value = evaluate(e, {{"x", 4.0}});
    ASSERT_TRUE(value.has_value());
    EXPECT_DOUBLE_EQ(*value, 9.0);

    // The old sym_eval returned 0.0 for an unbound variable, which is why
    // sym_eval("pi") reported 0.000000: indistinguishable from an answer.
    EXPECT_FALSE(evaluate(e, {}).has_value());
    EXPECT_FALSE(evaluate(function("log", {integer(-1)}), {}).has_value());
    EXPECT_FALSE(evaluate(function("sqrt", {integer(-1)}), {}).has_value());
    EXPECT_FALSE(evaluate(function("nosuchfunction", {x}), {{"x", 1.0}}).has_value());
    EXPECT_FALSE(evaluate(undefined(), {}).has_value());

    // pi and e are values, not free variables that happen to be spelled that way.
    const auto pi = evaluate(constant("pi"), {});
    ASSERT_TRUE(pi.has_value());
    EXPECT_NEAR(*pi, 3.14159265358979, 1e-12);
}

TEST(Sym2Core, ExactAtomsEvaluateToTheirValue) {
    double value = 0.0;
    ASSERT_TRUE(as_double(frac(1, 4), value));
    EXPECT_DOUBLE_EQ(value, 0.25);
    ms::bignum::Rational exact;
    ASSERT_TRUE(as_rational(frac(3, 4), exact));
    EXPECT_EQ(exact.num, BigInt(3LL));
    EXPECT_EQ(exact.den, BigInt(4LL));
    // A Real is not exact and does not claim to be.
    EXPECT_FALSE(is_exact(real(0.25)));
    EXPECT_TRUE(is_exact(frac(1, 4)));
    EXPECT_FALSE(as_rational(real(0.25), exact));
}

// An inexact value in a sum makes the sum inexact; it does not make it exact by being
// absorbed, and it does not make the exact parts inexact before they are added.
TEST(Sym2Core, ExactnessPropagatesOutwardNotInward) {
    EXPECT_TRUE(is_exact(add({frac(1, 3), frac(1, 6)})));
    EXPECT_FALSE(is_exact(add({frac(1, 3), real(0.5)})));
    EXPECT_EQ(head_of(add({frac(1, 3), real(0.5)})), Head::Real);
    EXPECT_TRUE(is_exact(mul({frac(1, 3), integer(3)})));
    EXPECT_FALSE(is_exact(mul({frac(1, 3), real(3.0)})));
}

TEST(Sym2Core, SubstitutionRebuildsThroughTheConstructors) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    // x + y with y := x must become 2*x, not sit as a two-term sum of the same thing.
    EXPECT_EQ(substitute(add({x, y}), y, x), mul({integer(2), x}));
    EXPECT_EQ(substitute(pow(x, integer(2)), x, integer(3)), integer(9));
    EXPECT_EQ(substitute(div(x, y), y, integer(0)), undefined());
    // Substituting something that is not there changes nothing, and returns the same
    // node rather than a copy of it.
    const ExprRef e = add({x, integer(1)});
    EXPECT_EQ(substitute(e, y, integer(5)).get(), e.get());
}

TEST(Sym2Core, FreeSymbolsAreFoundEverywhere) {
    const ExprRef e = add({mul({symbol("a"), function("sin", {symbol("b")})}),
                           pow(symbol("c"), integer(2)), integer(1)});
    const std::vector<std::string> names = free_symbols(e);
    ASSERT_EQ(names.size(), 3U);
    EXPECT_EQ(names[0], "a");
    EXPECT_EQ(names[1], "b");
    EXPECT_EQ(names[2], "c");
    EXPECT_TRUE(contains(e, symbol("b")));
    EXPECT_FALSE(contains(e, symbol("z")));
    EXPECT_TRUE(free_symbols(integer(3)).empty());
}

// The interner is process-wide mutable state, which is the shape that produces wrong
// answers when it is wrong. Building the same expression many times must give one node
// and must not lose one that is still referenced.
TEST(Sym2Core, InterningSurvivesChurn) {
    const ExprRef kept = add({symbol("kept"), integer(1)});
    for (int i = 0; i < 2000; ++i) {
        // Each of these dies immediately, so the bucket it landed in fills with expired
        // slots that the next insert has to prune without disturbing anything live.
        const ExprRef temporary = add({symbol("t" + std::to_string(i)), integer(i)});
        EXPECT_FALSE(is_undefined(temporary));
    }
    EXPECT_EQ(add({symbol("kept"), integer(1)}).get(), kept.get());
    EXPECT_EQ(to_string(kept), "kept + 1");
}

// The interning table keeps weak references, and a bucket is pruned when something
// hashes into it again. A hash whose nodes have all died is never hashed into again, so
// without a sweep the table grows by one empty entry for every distinct expression the
// process has ever built and released. Nothing about a *value* can see that: every
// answer stays correct while the table grows without bound.
//
// The bound asserted is deliberately loose. What is being tested is that the growth is
// not one-per-expression, not the exact sweep interval -- an assertion on the interval
// would fail the day somebody tunes it, which is not a regression.
TEST(Sym2Core, TheInternerDoesNotKeepBucketsForNodesThatDied) {
    constexpr int kTemporaries = 40000;
    const std::size_t before = interned_bucket_count();
    for (int i = 0; i < kTemporaries; ++i) {
        const ExprRef temporary = add({symbol("sweep" + std::to_string(i)), integer(i)});
        EXPECT_FALSE(is_undefined(temporary));
    }
    const std::size_t after = interned_bucket_count();
    EXPECT_LT(after, before + kTemporaries / 2)
        << "the table kept " << after - before << " buckets for " << kTemporaries
        << " expressions that are all gone";
}

// Every constructor here takes an `ExprRef`, which is a `shared_ptr` and so can be
// null, and `make` hashes what it is given by dereferencing it. `add`, `mul`, `pow` and
// `function` all refuse a null already. `derivative`, `integral` and `limit` did not,
// which is the shape of gap worth a test rather than a comment: a caller who checked
// one of them and found it safe has no reason to check the rest.
TEST(Sym2Core, EveryConstructorRefusesANullArgument) {
    const ExprRef nothing;
    ASSERT_FALSE(static_cast<bool>(nothing));
    const ExprRef x = symbol("x");

    EXPECT_TRUE(is_undefined(add({x, nothing})));
    EXPECT_TRUE(is_undefined(mul({nothing, x})));
    EXPECT_TRUE(is_undefined(pow(nothing, x)));
    EXPECT_TRUE(is_undefined(pow(x, nothing)));
    EXPECT_TRUE(is_undefined(function("f", {nothing})));

    EXPECT_TRUE(is_undefined(derivative(nothing, {x})));
    EXPECT_TRUE(is_undefined(derivative(x, {nothing})));
    EXPECT_TRUE(is_undefined(integral(nothing, {x})));
    EXPECT_TRUE(is_undefined(integral(x, {nothing})));
    EXPECT_TRUE(is_undefined(limit(nothing, x, integer(0))));
    EXPECT_TRUE(is_undefined(limit(x, nothing, integer(0))));
    EXPECT_TRUE(is_undefined(limit(x, x, nothing)));
}

} // namespace
