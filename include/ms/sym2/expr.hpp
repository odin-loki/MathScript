// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

/// @file
/// @brief The symbolic core of §10: an immutable, hash-consed expression DAG with
///   exact rational arithmetic, n-ary Add and Mul, and a Result<T> API.
///
/// This is built beside `ms::symbolic` rather than replacing it, as §10.5 directs.
/// The old engine carries real capability -- Laplace, Mellin, Hankel and Fourier
/// transforms, Z-transform, series, limits, linear solve, separable ODEs -- and that
/// is worth keeping while functions are ported one at a time against differential
/// tests. `to_legacy` and `from_legacy` are the bridge for that period.
///
/// What the old representation could not do, and why each of these fields exists:
///
/// **Exact arithmetic.** `SymExpr::value` is a `double`, so 1/3 is 0.333... and
/// `sym_simplify(x/3*3)` cannot return `x`. Approximate simplification is not
/// simplification. `Integer` and `Rational` atoms carry `bignum::BigInt`, which was
/// already in the tree and unused by `src/symbolic`.
///
/// **N-ary Add and Mul.** `a+b+c` used to parse as `(a+b)+c`, so term collection,
/// canonical ordering and structural equality all fought the tree shape --
/// `flatten_linear_sum` existed only to undo it at each call site. Add and Mul hold a
/// sorted argument vector, and the sort is what makes collection linear.
///
/// **Shared subexpressions.** `unique_ptr` made a strict tree, so expansion of nested
/// products was exponential in memory and not merely in time. `shared_ptr<const Node>`
/// with interning gives structural sharing and O(1) equality on top of it.
///
/// **Named heads for what cannot be evaluated.** An integral with no closed form is an
/// `Integral` node, not a sentinel that looks like an answer. This is the §10.2 fix:
/// nine functions in the old API returned a valid-looking `SymExpr` to signal failure,
/// and every caller had to know the convention.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/error/error_types.hpp"

namespace ms::sym2 {

/// What kind of expression a node is. The atom variant that goes with each is named
/// in the comment; heads with no atom carry `std::monostate`.
enum class Head : std::uint8_t {
    Integer,     ///< bignum::BigInt. Exact, unbounded.
    Rational,    ///< bignum::Rational, always reduced, denominator > 1.
    Real,        ///< double. For a value with no exact form, and only then.
    Symbol,      ///< std::string. A free variable.
    Constant,    ///< std::string. "pi", "e", "i", "inf", "-inf", "nan", "undefined".
    Add,         ///< n-ary, sorted, at least two arguments after simplification.
    Mul,         ///< n-ary, sorted, at least two arguments after simplification.
    Pow,         ///< exactly two arguments: base, exponent.
    Function,    ///< std::string name, n-ary arguments.
    Derivative,  ///< d/dx of args[0]; args[1..] are the Symbols differentiated by.
    Integral,    ///< the integral of args[0] with respect to args[1..], unevaluated.
    Limit,       ///< args[0] as args[1] approaches args[2], unevaluated.
};

struct Node;

/// A node is immutable once built, so a reference is all anyone ever needs and copying
/// one is a pointer copy. Nothing in this namespace hands out a mutable node.
using ExprRef = std::shared_ptr<const Node>;

struct Node {
    Head head = Head::Integer;
    std::variant<std::monostate, bignum::BigInt, bignum::Rational, double, std::string>
        atom{};
    std::vector<ExprRef> args;
    /// Computed once at construction, over the head, the atom and the children's own
    /// hashes. Two structurally equal expressions always hash equally; the converse is
    /// checked rather than assumed (see `structurally_equal`).
    std::size_t hash = 0;
};

// --- Construction -----------------------------------------------------------------
//
// Every builder simplifies as it constructs, which is the decision §10.4 calls the one
// that matters most: Add sorts and folds and collects at build time, so `simplify`
// handles only the hard cases rather than everything. Retrofitting this later is much
// harder than doing it now, and it is how every serious CAS is built.

ExprRef integer(long long value);
ExprRef integer(bignum::BigInt value);
/// Exact p/q. Reduces, and returns an Integer when the denominator divides out.
/// A zero denominator is `undefined()`, never a node claiming to be a number.
ExprRef rational(bignum::BigInt numerator, bignum::BigInt denominator);
ExprRef real(double value);
ExprRef symbol(const std::string& name);
ExprRef constant(const std::string& name);
ExprRef undefined();

/// Sum of `args`, flattened, sorted, numerically folded, like terms collected.
/// An empty sum is 0 and a one-element sum is that element.
ExprRef add(std::vector<ExprRef> args);
/// Product of `args`, with the same treatment. An empty product is 1. A factor of
/// exact zero annihilates -- but only exact zero: a Real 0.0 does not, because it may
/// stand for an underflowed non-zero and multiplying by an unevaluated head would then
/// be discarding it.
ExprRef mul(std::vector<ExprRef> args);
ExprRef pow(ExprRef base, ExprRef exponent);
ExprRef neg(ExprRef a);
ExprRef sub(ExprRef a, ExprRef b);
/// a / b. Division by exact zero is `undefined()`.
ExprRef div(ExprRef a, ExprRef b);
ExprRef function(const std::string& name, std::vector<ExprRef> args);
ExprRef derivative(ExprRef expr, std::vector<ExprRef> vars);
ExprRef integral(ExprRef expr, std::vector<ExprRef> vars);
ExprRef limit(ExprRef expr, ExprRef var, ExprRef point);

// --- Inspection -------------------------------------------------------------------

inline Head head_of(const ExprRef& e) { return e->head; }
bool is_number(const ExprRef& e);   ///< Integer, Rational or Real.
bool is_exact(const ExprRef& e);    ///< Integer or Rational: no rounding has happened.
bool is_zero(const ExprRef& e);     ///< Exactly zero. A Real 0.0 is not exactly zero.
bool is_one(const ExprRef& e);
bool is_undefined(const ExprRef& e);

/// The value as a rational, when the node is one exactly. Nothing rounds here.
bool as_rational(const ExprRef& e, bignum::Rational& out);
/// The value as a double, for any numeric head. Exact atoms are converted, which is
/// the only place in this namespace where an exact value becomes approximate.
bool as_double(const ExprRef& e, double& out);

/// How many hash buckets the interning table currently holds.
///
/// A diagnostic, not a tuning knob: the table keeps weak references, so a bucket whose
/// nodes have all died holds nothing and is swept, and this is how a test can tell that
/// the sweeping happens. Without it the table grows by one entry per distinct
/// expression the process ever built, which a REPL session survives and a long-lived
/// worker does not -- and a leak that slow is not visible in any assertion about
/// values.
std::size_t interned_bucket_count();

/// Structural equality. Cheap when the hashes differ, which is the common case; a
/// hash collision falls through to a full comparison rather than being trusted.
bool structurally_equal(const ExprRef& a, const ExprRef& b);

/// A total order over expressions, deterministic across runs and processes: by head,
/// then by atom, then by arity, then lexicographically over arguments. It is what Add
/// and Mul sort by, so it must not depend on pointer values or on allocation order --
/// a canonical form that changes between runs is not canonical.
int compare(const ExprRef& a, const ExprRef& b);

/// Every Symbol appearing anywhere in `e`, in canonical order, without duplicates.
std::vector<std::string> free_symbols(const ExprRef& e);
bool contains(const ExprRef& e, const ExprRef& sub);

// --- Printing ---------------------------------------------------------------------

/// Precedence-aware. `2*x + 1` prints as `2*x + 1`, where the old printer fully
/// parenthesised every node and formatted every number with `std::to_string`, giving
/// `((2.000000 * x) + 1.000000)`. §10.3 calls this a prerequisite for every output
/// format in §11 rather than a separate task, and it is: the LaTeX printer would have
/// inherited both faults.
///
/// What comes out is an expression, so it reads back as the same expression. That is
/// a stronger requirement than the REPL's scalar display, which is allowed to round a
/// value for legibility -- here a rounded constant would denote a different expression.
/// Exact atoms print exactly and briefly; only a value that is genuinely inexact costs
/// the digits its exactness needs.
std::string to_string(const ExprRef& e);

// --- Evaluation -------------------------------------------------------------------

/// Numeric value under an environment. Unlike the old `double sym_eval(...)`, an
/// unbound symbol, a division by zero and an undefined result are reported rather than
/// being returned as a number indistinguishable from a legitimate one.
Result<double> evaluate(const ExprRef& e, const std::map<std::string, double>& env);

/// Replace every occurrence of `target` with `replacement`, rebuilding through the
/// simplifying constructors so the result is canonical.
ExprRef substitute(const ExprRef& e, const ExprRef& target, const ExprRef& replacement);

} // namespace ms::sym2
