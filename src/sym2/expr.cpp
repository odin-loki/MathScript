// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/sym2/expr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "ms/core/format.hpp"

namespace ms::sym2 {

namespace {

using bignum::BigInt;
using bignum::Rational;

// --- Hashing and interning ---------------------------------------------------------

void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::size_t hash_bigint(const BigInt& value) {
    std::size_t h = value.negative ? 1u : 0u;
    for (const std::uint32_t limb : value.digits) {
        hash_combine(h, static_cast<std::size_t>(limb));
    }
    return h;
}

std::size_t compute_hash(const Node& node) {
    std::size_t h = static_cast<std::size_t>(node.head) + 0x1000193ULL;
    if (const auto* v = std::get_if<BigInt>(&node.atom)) {
        hash_combine(h, hash_bigint(*v));
    } else if (const auto* v = std::get_if<Rational>(&node.atom)) {
        hash_combine(h, hash_bigint(v->num));
        hash_combine(h, hash_bigint(v->den));
    } else if (const auto* v = std::get_if<double>(&node.atom)) {
        // Through the bit pattern, so that -0.0 and 0.0 hash apart exactly as they
        // compare apart, and so that a NaN hashes to itself.
        std::uint64_t bits = 0;
        std::memcpy(&bits, v, sizeof(bits));
        hash_combine(h, static_cast<std::size_t>(bits));
    } else if (const auto* v = std::get_if<std::string>(&node.atom)) {
        hash_combine(h, std::hash<std::string>{}(*v));
    }
    for (const ExprRef& arg : node.args) {
        hash_combine(h, arg->hash);
    }
    return h;
}

bool same_atom(const Node& a, const Node& b) {
    if (a.atom.index() != b.atom.index()) {
        return false;
    }
    if (const auto* x = std::get_if<BigInt>(&a.atom)) {
        return *x == *std::get_if<BigInt>(&b.atom);
    }
    if (const auto* x = std::get_if<Rational>(&a.atom)) {
        const auto* y = std::get_if<Rational>(&b.atom);
        return x->num == y->num && x->den == y->den;
    }
    if (const auto* x = std::get_if<double>(&a.atom)) {
        // Bitwise, for the same reason the hash is: an interning table that folds
        // -0.0 into 0.0, or that never matches two NaNs, is one that changes values.
        const auto* y = std::get_if<double>(&b.atom);
        return std::memcmp(x, y, sizeof(double)) == 0;
    }
    if (const auto* x = std::get_if<std::string>(&a.atom)) {
        return *x == *std::get_if<std::string>(&b.atom);
    }
    return true; // both monostate
}

bool shallow_equal(const Node& a, const Node& b) {
    if (a.head != b.head || a.args.size() != b.args.size() || !same_atom(a, b)) {
        return false;
    }
    // Interned children are pointer-identical when equal, so this is a pointer scan.
    for (std::size_t i = 0; i < a.args.size(); ++i) {
        if (a.args[i] != b.args[i]) {
            return false;
        }
    }
    return true;
}

/// Every node the process has built and still holds, keyed by hash.
///
/// Interning is what buys O(1) structural equality: two expressions are equal exactly
/// when their `ExprRef`s are the same pointer, because a node with the same head, atom
/// and (already interned) children is never built twice. It is also what bounds the
/// memory of an expansion -- a repeated subexpression is stored once however many
/// places refer to it.
///
/// The table holds weak references, so a node dies when the last expression using it
/// does; expired slots are pruned from a bucket when that bucket is next touched.
/// A mutex guards it because this is process-wide state and the library is used from
/// more than one thread (the GUI runs the interpreter on a worker thread, and the
/// distributed layer runs several). A table like this one is exactly the shape of
/// shared mutable state that produces wrong answers under threading if left unguarded.
struct Interner {
    std::mutex mutex;
    std::unordered_map<std::size_t, std::vector<std::weak_ptr<const Node>>> buckets;
};

Interner& interner() {
    static Interner instance;
    return instance;
}

ExprRef intern(Node&& node) {
    node.hash = compute_hash(node);
    Interner& table = interner();
    const std::lock_guard<std::mutex> lock(table.mutex);
    auto& bucket = table.buckets[node.hash];
    std::size_t write = 0;
    ExprRef found;
    for (std::size_t read = 0; read < bucket.size(); ++read) {
        ExprRef live = bucket[read].lock();
        if (!live) {
            continue; // expired: drop it while we are here
        }
        if (!found && shallow_equal(*live, node)) {
            found = live;
        }
        bucket[write++] = std::move(bucket[read]);
    }
    bucket.resize(write);
    if (found) {
        return found;
    }
    auto created = std::make_shared<const Node>(std::move(node));
    bucket.push_back(created);
    return created;
}

ExprRef make(Head head, std::variant<std::monostate, BigInt, Rational, double, std::string> atom,
            std::vector<ExprRef> args) {
    Node node;
    node.head = head;
    node.atom = std::move(atom);
    node.args = std::move(args);
    return intern(std::move(node));
}

// --- Exact numeric helpers ---------------------------------------------------------

const BigInt& big_one() {
    static const BigInt value(1LL);
    return value;
}

bool rational_is_integer(const Rational& value) { return value.den == big_one(); }

ExprRef from_rational(const Rational& value) {
    if (rational_is_integer(value)) {
        return integer(value.num);
    }
    return make(Head::Rational, value, {});
}

/// The exponent range an exact power is allowed to reach. 2^1000000 is representable
/// in BigInt and would be 300kB of digits produced from a five-character input, so the
/// answer stays symbolic past this point rather than being materialised.
constexpr long long kMaxExactPowerExponent = 4096;

bool bigint_to_ll(const BigInt& value, long long& out) {
    if (value.digits.size() > 3) {
        return false;
    }
    unsigned long long magnitude = 0;
    for (std::size_t i = value.digits.size(); i-- > 0;) {
        if (magnitude > (~0ULL) / static_cast<unsigned long long>(BigInt::BASE)) {
            return false;
        }
        magnitude = magnitude * static_cast<unsigned long long>(BigInt::BASE) + value.digits[i];
    }
    if (magnitude > 9223372036854775807ULL) {
        return false;
    }
    out = value.negative ? -static_cast<long long>(magnitude) : static_cast<long long>(magnitude);
    return true;
}

} // namespace

// --- Construction ------------------------------------------------------------------

ExprRef integer(long long value) { return integer(BigInt(value)); }

ExprRef integer(BigInt value) { return make(Head::Integer, std::move(value), {}); }

ExprRef rational(BigInt numerator, BigInt denominator) {
    if (denominator.is_zero()) {
        return undefined();
    }
    return from_rational(Rational(std::move(numerator), std::move(denominator)));
}

ExprRef real(double value) { return make(Head::Real, value, {}); }

ExprRef symbol(const std::string& name) { return make(Head::Symbol, name, {}); }

ExprRef constant(const std::string& name) { return make(Head::Constant, name, {}); }

ExprRef undefined() { return constant("undefined"); }

// --- Inspection --------------------------------------------------------------------

bool is_number(const ExprRef& e) {
    return e->head == Head::Integer || e->head == Head::Rational || e->head == Head::Real;
}

bool is_exact(const ExprRef& e) {
    return e->head == Head::Integer || e->head == Head::Rational;
}

bool is_zero(const ExprRef& e) {
    const auto* value = std::get_if<BigInt>(&e->atom);
    return e->head == Head::Integer && value != nullptr && value->is_zero();
}

bool is_one(const ExprRef& e) {
    const auto* value = std::get_if<BigInt>(&e->atom);
    return e->head == Head::Integer && value != nullptr && value->is_one() && !value->negative;
}

bool is_undefined(const ExprRef& e) {
    const auto* name = std::get_if<std::string>(&e->atom);
    return e->head == Head::Constant && name != nullptr && *name == "undefined";
}

bool as_rational(const ExprRef& e, Rational& out) {
    if (e->head == Head::Integer) {
        out = Rational(*std::get_if<BigInt>(&e->atom), big_one());
        return true;
    }
    if (e->head == Head::Rational) {
        out = *std::get_if<Rational>(&e->atom);
        return true;
    }
    return false;
}

bool as_double(const ExprRef& e, double& out) {
    Rational exact;
    if (as_rational(e, exact)) {
        out = exact.to_double();
        return true;
    }
    if (e->head == Head::Real) {
        out = *std::get_if<double>(&e->atom);
        return true;
    }
    return false;
}

bool structurally_equal(const ExprRef& a, const ExprRef& b) {
    // Interning makes this a pointer comparison. The structural fallback is kept
    // because a node built before an interner reset, or handed in from a bridge that
    // constructed one directly, would otherwise compare unequal to its own twin.
    if (a == b) {
        return true;
    }
    if (!a || !b || a->hash != b->hash || a->head != b->head ||
        a->args.size() != b->args.size() || !same_atom(*a, *b)) {
        return false;
    }
    for (std::size_t i = 0; i < a->args.size(); ++i) {
        if (!structurally_equal(a->args[i], b->args[i])) {
            return false;
        }
    }
    return true;
}

int compare(const ExprRef& a, const ExprRef& b) {
    if (a == b) {
        return 0;
    }
    if (a->head != b->head) {
        return static_cast<int>(a->head) < static_cast<int>(b->head) ? -1 : 1;
    }
    switch (a->head) {
    case Head::Integer: {
        const BigInt& x = *std::get_if<BigInt>(&a->atom);
        const BigInt& y = *std::get_if<BigInt>(&b->atom);
        if (x == y) {
            return 0;
        }
        return x < y ? -1 : 1;
    }
    case Head::Rational: {
        const Rational& x = *std::get_if<Rational>(&a->atom);
        const Rational& y = *std::get_if<Rational>(&b->atom);
        if (x == y) {
            return 0;
        }
        return x < y ? -1 : 1;
    }
    case Head::Real: {
        const double x = *std::get_if<double>(&a->atom);
        const double y = *std::get_if<double>(&b->atom);
        if (x < y) {
            return -1;
        }
        if (y < x) {
            return 1;
        }
        // Equal, or one of them is NaN. Order by bit pattern so that the comparison is
        // a total order even there: a sort with an inconsistent comparator is not a
        // slow sort, it is undefined behaviour.
        std::uint64_t xb = 0;
        std::uint64_t yb = 0;
        std::memcpy(&xb, &x, sizeof(xb));
        std::memcpy(&yb, &y, sizeof(yb));
        if (xb == yb) {
            return 0;
        }
        return xb < yb ? -1 : 1;
    }
    case Head::Symbol:
    case Head::Constant:
    case Head::Function: {
        const std::string& x = *std::get_if<std::string>(&a->atom);
        const std::string& y = *std::get_if<std::string>(&b->atom);
        if (x != y) {
            return x < y ? -1 : 1;
        }
        break;
    }
    default:
        break;
    }
    if (a->args.size() != b->args.size()) {
        return a->args.size() < b->args.size() ? -1 : 1;
    }
    for (std::size_t i = 0; i < a->args.size(); ++i) {
        const int order = compare(a->args[i], b->args[i]);
        if (order != 0) {
            return order;
        }
    }
    return 0;
}

namespace {

struct ByCanonicalOrder {
    bool operator()(const ExprRef& a, const ExprRef& b) const { return compare(a, b) < 0; }
};

/// Split a term into its numeric coefficient and the rest of it, which is the key Add
/// collects on. `3*x*y` gives (3, x*y); `x` gives (1, x); `7` gives (7, {}).
void split_coefficient(const ExprRef& term, ExprRef& coefficient, ExprRef& base) {
    if (is_number(term)) {
        coefficient = term;
        base = nullptr;
        return;
    }
    if (term->head == Head::Mul && !term->args.empty() && is_number(term->args.front())) {
        coefficient = term->args.front();
        std::vector<ExprRef> rest(term->args.begin() + 1, term->args.end());
        base = rest.size() == 1 ? rest.front() : mul(std::move(rest));
        return;
    }
    coefficient = integer(1);
    base = term;
}

/// Split a factor into its base and exponent, which is the key Mul collects on.
void split_exponent(const ExprRef& factor, ExprRef& base, ExprRef& exponent) {
    if (factor->head == Head::Pow) {
        base = factor->args[0];
        exponent = factor->args[1];
        return;
    }
    base = factor;
    exponent = integer(1);
}

void flatten(Head head, const ExprRef& e, std::vector<ExprRef>& out) {
    if (e->head == head) {
        for (const ExprRef& arg : e->args) {
            flatten(head, arg, out);
        }
        return;
    }
    out.push_back(e);
}

/// The sum of two numeric nodes, exact where both are.
ExprRef add_numbers(const ExprRef& a, const ExprRef& b) {
    Rational x;
    Rational y;
    if (as_rational(a, x) && as_rational(b, y)) {
        return from_rational(x + y);
    }
    double dx = 0.0;
    double dy = 0.0;
    as_double(a, dx);
    as_double(b, dy);
    return real(dx + dy);
}

ExprRef multiply_numbers(const ExprRef& a, const ExprRef& b) {
    Rational x;
    Rational y;
    if (as_rational(a, x) && as_rational(b, y)) {
        return from_rational(x * y);
    }
    double dx = 0.0;
    double dy = 0.0;
    as_double(a, dx);
    as_double(b, dy);
    return real(dx * dy);
}

bool is_positive_number(const ExprRef& e) {
    Rational exact;
    if (as_rational(e, exact)) {
        return !exact.num.negative && !exact.num.is_zero();
    }
    if (e->head == Head::Real) {
        return *std::get_if<double>(&e->atom) > 0.0;
    }
    return false;
}

bool is_negative_number(const ExprRef& e) {
    Rational exact;
    if (as_rational(e, exact)) {
        return exact.num.negative && !exact.num.is_zero();
    }
    if (e->head == Head::Real) {
        return *std::get_if<double>(&e->atom) < 0.0;
    }
    return false;
}

/// An `undefined` anywhere in a sum or product makes the whole thing undefined, and it
/// must not be swallowed by a zero factor or cancelled against its own twin.
bool any_undefined(const std::vector<ExprRef>& args) {
    return std::any_of(args.begin(), args.end(),
                       [](const ExprRef& a) { return is_undefined(a); });
}

} // namespace

ExprRef add(std::vector<ExprRef> args) {
    std::vector<ExprRef> flat;
    flat.reserve(args.size());
    for (const ExprRef& arg : args) {
        if (!arg) {
            return undefined();
        }
        flatten(Head::Add, arg, flat);
    }
    if (any_undefined(flat)) {
        return undefined();
    }

    ExprRef constant_term = integer(0);
    // Bases in canonical order, each with the coefficient collected so far. A map
    // rather than a sort-then-scan because the coefficients are summed as they arrive.
    std::map<ExprRef, ExprRef, ByCanonicalOrder> collected;
    for (const ExprRef& term : flat) {
        ExprRef coefficient;
        ExprRef base;
        split_coefficient(term, coefficient, base);
        if (!base) {
            constant_term = add_numbers(constant_term, coefficient);
            continue;
        }
        auto slot = collected.find(base);
        if (slot == collected.end()) {
            collected.emplace(base, coefficient);
        } else {
            slot->second = add_numbers(slot->second, coefficient);
        }
    }

    std::vector<ExprRef> terms;
    terms.reserve(collected.size() + 1);
    for (const auto& [base, coefficient] : collected) {
        if (is_zero(coefficient)) {
            continue;
        }
        if (is_one(coefficient)) {
            terms.push_back(base);
        } else {
            terms.push_back(mul({coefficient, base}));
        }
    }
    std::sort(terms.begin(), terms.end(), ByCanonicalOrder{});
    if (!is_zero(constant_term)) {
        terms.insert(terms.begin(), constant_term);
    }
    if (terms.empty()) {
        return integer(0);
    }
    if (terms.size() == 1) {
        return terms.front();
    }
    return make(Head::Add, std::monostate{}, std::move(terms));
}

ExprRef mul(std::vector<ExprRef> args) {
    std::vector<ExprRef> flat;
    flat.reserve(args.size());
    for (const ExprRef& arg : args) {
        if (!arg) {
            return undefined();
        }
        flatten(Head::Mul, arg, flat);
    }
    if (any_undefined(flat)) {
        return undefined();
    }

    ExprRef coefficient = integer(1);
    std::map<ExprRef, std::vector<ExprRef>, ByCanonicalOrder> exponents;
    for (const ExprRef& factor : flat) {
        if (is_number(factor)) {
            coefficient = multiply_numbers(coefficient, factor);
            continue;
        }
        ExprRef base;
        ExprRef exponent;
        split_exponent(factor, base, exponent);
        exponents[base].push_back(exponent);
    }

    // An exact zero annihilates -- but only after the arguments have been inspected,
    // so that 0 * undefined is undefined rather than 0. A Real 0.0 does not annihilate
    // an unevaluated factor: it may be an underflowed non-zero, and discarding the
    // factor would be asserting it is not.
    if (is_zero(coefficient)) {
        return integer(0);
    }

    std::vector<ExprRef> factors;
    factors.reserve(exponents.size() + 1);
    for (auto& [base, parts] : exponents) {
        ExprRef exponent = parts.size() == 1 ? parts.front() : add(parts);
        if (is_zero(exponent)) {
            continue; // b^0 is 1 for every b this branch can see
        }
        factors.push_back(is_one(exponent) ? base : pow(base, exponent));
    }
    std::sort(factors.begin(), factors.end(), ByCanonicalOrder{});
    if (factors.empty()) {
        return coefficient;
    }
    if (!is_one(coefficient)) {
        factors.insert(factors.begin(), coefficient);
    }
    if (factors.size() == 1) {
        return factors.front();
    }
    return make(Head::Mul, std::monostate{}, std::move(factors));
}

ExprRef pow(ExprRef base, ExprRef exponent) {
    if (!base || !exponent || is_undefined(base) || is_undefined(exponent)) {
        return undefined();
    }
    if (is_zero(exponent)) {
        // 0^0 has no value every branch of mathematics agrees on, and a CAS that picks
        // one silently is a CAS that will be wrong for somebody.
        return is_zero(base) ? undefined() : integer(1);
    }
    if (is_one(exponent)) {
        return base;
    }
    if (is_one(base)) {
        return integer(1);
    }
    if (is_zero(base)) {
        return is_negative_number(exponent) ? undefined() : integer(0);
    }

    Rational exact_base;
    Rational exact_exponent;
    if (as_rational(base, exact_base) && as_rational(exponent, exact_exponent) &&
        rational_is_integer(exact_exponent)) {
        long long power = 0;
        if (bigint_to_ll(exact_exponent.num, power) &&
            power >= -kMaxExactPowerExponent && power <= kMaxExactPowerExponent) {
            const bool invert = power < 0;
            const long long magnitude = invert ? -power : power;
            BigInt num = bignum::bigint_pow(exact_base.num, magnitude);
            BigInt den = bignum::bigint_pow(exact_base.den, magnitude);
            return invert ? rational(std::move(den), std::move(num))
                          : from_rational(Rational(std::move(num), std::move(den)));
        }
        // Past the cap the answer stays symbolic rather than being materialised.
    }

    double numeric_base = 0.0;
    double numeric_exponent = 0.0;
    if (base->head == Head::Real && as_double(exponent, numeric_exponent)) {
        as_double(base, numeric_base);
        return real(std::pow(numeric_base, numeric_exponent));
    }
    if (exponent->head == Head::Real && as_double(base, numeric_base)) {
        as_double(exponent, numeric_exponent);
        if (numeric_base < 0.0 && numeric_exponent != std::floor(numeric_exponent)) {
            return undefined(); // a real root of a negative number is not real
        }
        return real(std::pow(numeric_base, numeric_exponent));
    }

    // (b^p)^q is b^(p*q) only where the rewrite is an identity, and it is not in
    // general. (x^2)^(1/2) is |x|, not x -- at x = -2 the two differ by the whole
    // answer. Two cases are safe:
    //
    //   q an integer: (b^p)^n = b^(pn) for every b for which b^p is defined.
    //   b a positive number: no branch to choose, so any real p and q compose.
    //
    // "b is not a negative number" is NOT one of them, which is the mistake this
    // condition was written with the first time: a Symbol is not a negative number and
    // is not known to be non-negative either, so the fold applied to exactly the case
    // it was meant to exclude.
    if (base->head == Head::Pow) {
        Rational outer;
        const bool outer_integer = as_rational(exponent, outer) && rational_is_integer(outer);
        if (outer_integer || is_positive_number(base->args[0])) {
            return pow(base->args[0], mul({base->args[1], exponent}));
        }
    }
    return make(Head::Pow, std::monostate{}, {std::move(base), std::move(exponent)});
}

ExprRef neg(ExprRef a) { return mul({integer(-1), std::move(a)}); }

ExprRef sub(ExprRef a, ExprRef b) { return add({std::move(a), neg(std::move(b))}); }

ExprRef div(ExprRef a, ExprRef b) {
    if (!a || !b) {
        return undefined();
    }
    if (is_zero(b)) {
        return undefined();
    }
    return mul({std::move(a), pow(std::move(b), integer(-1))});
}

ExprRef function(const std::string& name, std::vector<ExprRef> args) {
    for (const ExprRef& arg : args) {
        if (!arg) {
            return undefined();
        }
    }
    return make(Head::Function, name, std::move(args));
}

namespace {

/// A null `ExprRef` among the arguments, in any of the three below.
///
/// `add`, `mul`, `pow` and `function` all refuse one already; these three did not, and
/// a null reaches `compute_hash`, which dereferences it. One guard here and none there
/// is how a caller comes to believe the whole namespace is safe against a value that
/// only most of it is safe against.
bool any_null(const std::vector<ExprRef>& args) {
    for (const ExprRef& arg : args) {
        if (!arg) {
            return true;
        }
    }
    return false;
}

} // namespace

ExprRef derivative(ExprRef expr, std::vector<ExprRef> vars) {
    std::vector<ExprRef> args;
    args.push_back(std::move(expr));
    for (ExprRef& var : vars) {
        args.push_back(std::move(var));
    }
    if (any_null(args)) {
        return undefined();
    }
    return make(Head::Derivative, std::monostate{}, std::move(args));
}

ExprRef integral(ExprRef expr, std::vector<ExprRef> vars) {
    std::vector<ExprRef> args;
    args.push_back(std::move(expr));
    for (ExprRef& var : vars) {
        args.push_back(std::move(var));
    }
    if (any_null(args)) {
        return undefined();
    }
    return make(Head::Integral, std::monostate{}, std::move(args));
}

ExprRef limit(ExprRef expr, ExprRef var, ExprRef point) {
    if (!expr || !var || !point) {
        return undefined();
    }
    return make(Head::Limit, std::monostate{},
                {std::move(expr), std::move(var), std::move(point)});
}

// --- Traversal ---------------------------------------------------------------------

namespace {

void gather_symbols(const ExprRef& e, std::vector<std::string>& out) {
    if (e->head == Head::Symbol) {
        out.push_back(*std::get_if<std::string>(&e->atom));
        return;
    }
    for (const ExprRef& arg : e->args) {
        gather_symbols(arg, out);
    }
}

} // namespace

std::vector<std::string> free_symbols(const ExprRef& e) {
    std::vector<std::string> names;
    if (e) {
        gather_symbols(e, names);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

bool contains(const ExprRef& e, const ExprRef& sub) {
    if (!e || !sub) {
        return false;
    }
    if (structurally_equal(e, sub)) {
        return true;
    }
    return std::any_of(e->args.begin(), e->args.end(),
                       [&sub](const ExprRef& arg) { return contains(arg, sub); });
}

ExprRef substitute(const ExprRef& e, const ExprRef& target, const ExprRef& replacement) {
    if (!e || !target || !replacement) {
        return undefined();
    }
    if (structurally_equal(e, target)) {
        return replacement;
    }
    if (e->args.empty()) {
        return e;
    }
    std::vector<ExprRef> args;
    args.reserve(e->args.size());
    bool changed = false;
    for (const ExprRef& arg : e->args) {
        ExprRef rebuilt = substitute(arg, target, replacement);
        changed = changed || rebuilt != arg;
        args.push_back(std::move(rebuilt));
    }
    if (!changed) {
        return e;
    }
    // Rebuild through the simplifying constructors, so a substitution that makes two
    // terms alike collects them instead of leaving x + x sitting in a sum.
    switch (e->head) {
    case Head::Add:
        return add(std::move(args));
    case Head::Mul:
        return mul(std::move(args));
    case Head::Pow:
        return pow(args[0], args[1]);
    case Head::Function:
        return function(*std::get_if<std::string>(&e->atom), std::move(args));
    default:
        return make(e->head, e->atom, std::move(args));
    }
}

// --- Printing ----------------------------------------------------------------------

namespace {

// Binding strength, loosest first. A child needs parentheses exactly when it binds
// more loosely than the context it is being placed into.
constexpr int kPrecAdd = 1;
constexpr int kPrecMul = 2;
constexpr int kPrecPow = 3;
constexpr int kPrecAtom = 4;

int precedence_of(const ExprRef& e) {
    switch (e->head) {
    case Head::Add:
        return kPrecAdd;
    case Head::Mul:
        return kPrecMul;
    case Head::Pow:
        return kPrecPow;
    case Head::Rational:
        return kPrecMul; // prints as p/q, which binds like a product
    case Head::Integer:
    case Head::Real:
        return is_negative_number(e) ? kPrecAdd : kPrecAtom;
    default:
        return kPrecAtom;
    }
}

std::string render(const ExprRef& e, int context);

std::string wrap(const ExprRef& e, int context) {
    const std::string text = render(e, context);
    return precedence_of(e) < context ? "(" + text + ")" : text;
}

/// The negation of a term, for printing `a - b` instead of `a + -1*b`. Returns false
/// when the term is not negative, so the caller keeps the `+`.
bool negated_for_display(const ExprRef& term, ExprRef& positive) {
    if (is_number(term)) {
        if (!is_negative_number(term)) {
            return false;
        }
        positive = neg(term);
        return true;
    }
    if (term->head == Head::Mul && !term->args.empty() && is_negative_number(term->args.front())) {
        positive = neg(term);
        return true;
    }
    return false;
}

std::string render_add(const ExprRef& e) {
    // The node keeps its arguments in canonical order, which puts the constant term
    // first because numeric heads sort first. Nobody writes sums that way: 2*x + 1, not
    // 1 + 2*x. The display order is the printer's business and the canonical order is
    // the node's, so they are allowed to differ.
    std::vector<ExprRef> display;
    display.reserve(e->args.size());
    ExprRef constant_term;
    for (const ExprRef& term : e->args) {
        if (!constant_term && is_number(term)) {
            constant_term = term;
            continue;
        }
        display.push_back(term);
    }
    if (constant_term) {
        display.push_back(constant_term);
    }

    std::string text;
    for (std::size_t i = 0; i < display.size(); ++i) {
        ExprRef positive;
        const bool negative = negated_for_display(display[i], positive);
        const ExprRef& shown = negative ? positive : display[i];
        if (i == 0) {
            text += negative ? "-" : "";
        } else {
            text += negative ? " - " : " + ";
        }
        text += wrap(shown, kPrecAdd + 1);
    }
    return text;
}

std::string render_mul(const ExprRef& e) {
    // Factors with a negative integer exponent are the denominator. Printing them as
    // x^-1 is correct and unreadable; printing a/b is what anyone writing the
    // expression down would do.
    std::vector<ExprRef> numerator;
    std::vector<ExprRef> denominator;
    bool negative = false;
    for (const ExprRef& factor : e->args) {
        if (factor->head == Head::Pow && is_negative_number(factor->args[1])) {
            denominator.push_back(pow(factor->args[0], neg(factor->args[1])));
            continue;
        }
        Rational exact;
        if (as_rational(factor, exact) && !rational_is_integer(exact)) {
            if (!exact.num.is_one() || exact.num.negative) {
                numerator.push_back(integer(exact.num));
            } else if (exact.num.negative) {
                negative = true;
            }
            denominator.push_back(integer(exact.den));
            continue;
        }
        if (is_number(factor) && is_negative_number(factor) && e->args.size() > 1) {
            ExprRef magnitude = neg(factor);
            negative = true;
            if (!is_one(magnitude)) {
                numerator.push_back(magnitude);
            }
            continue;
        }
        numerator.push_back(factor);
    }

    // Display order again, not canonical order: the node sorts a Symbol before a Pow,
    // so x^2*y comes out of the constructor as [y, x^2]. Nobody writes it that way.
    // Ordering the factors by the name of their base puts x^2 before y and leaves the
    // node itself untouched.
    const auto display_key = [](const ExprRef& factor) {
        if (is_number(factor)) {
            return std::string{"\x01"}; // the coefficient stays in front
        }
        return to_string(factor->head == Head::Pow ? factor->args[0] : factor);
    };
    std::stable_sort(numerator.begin(), numerator.end(),
                     [&display_key](const ExprRef& a, const ExprRef& b) {
                         return display_key(a) < display_key(b);
                     });

    std::string text = negative ? "-" : "";
    if (numerator.empty()) {
        text += "1";
    } else {
        for (std::size_t i = 0; i < numerator.size(); ++i) {
            if (i > 0) {
                text += "*";
            }
            text += wrap(numerator[i], kPrecMul);
        }
    }
    for (const ExprRef& factor : denominator) {
        text += "/" + wrap(factor, kPrecMul + 1);
    }
    return text;
}

std::string render(const ExprRef& e, int context) {
    (void)context;
    switch (e->head) {
    case Head::Integer:
        return std::get_if<BigInt>(&e->atom)->to_string();
    case Head::Rational: {
        const Rational& value = *std::get_if<Rational>(&e->atom);
        return value.num.to_string() + "/" + value.den.to_string();
    }
    case Head::Real:
        // format_exact, not format_scalar. What this function produces is an
        // expression, not a number for a person to skim: it is parsed back, compared,
        // and (once §11 lands) turned into LaTeX. A display spelling rounds -- 7/3 as
        // a double prints "2.333333" under six decimals -- and the text then denotes a
        // different expression from the one it was printed from. The REPL's scalar
        // output is a display and rounds; this is not and does not.
        return format_exact(*std::get_if<double>(&e->atom));
    case Head::Symbol:
    case Head::Constant:
        return *std::get_if<std::string>(&e->atom);
    case Head::Add:
        return render_add(e);
    case Head::Mul:
        return render_mul(e);
    case Head::Pow: {
        // A negative exponent is a reciprocal. x^(-1) is correct and nobody writes it;
        // inside a product render_mul already moves such a factor into the denominator,
        // and this is the same rule for a power standing on its own.
        if (is_negative_number(e->args[1])) {
            const ExprRef magnitude = neg(e->args[1]);
            const ExprRef reciprocal =
                is_one(magnitude) ? e->args[0] : pow(e->args[0], magnitude);
            return "1/" + wrap(reciprocal, kPrecMul + 1);
        }
        // Right-associative, so the exponent needs no parentheses of its own unless it
        // binds more loosely than a power.
        return wrap(e->args[0], kPrecPow + 1) + "^" + wrap(e->args[1], kPrecPow);
    }
    case Head::Function: {
        std::string text = *std::get_if<std::string>(&e->atom);
        text += "(";
        for (std::size_t i = 0; i < e->args.size(); ++i) {
            if (i > 0) {
                text += ", ";
            }
            text += render(e->args[i], kPrecAdd);
        }
        return text + ")";
    }
    case Head::Derivative: {
        std::string text = "d/d";
        for (std::size_t i = 1; i < e->args.size(); ++i) {
            text += render(e->args[i], kPrecAtom);
        }
        return text + "(" + render(e->args[0], kPrecAdd) + ")";
    }
    case Head::Integral: {
        std::string text = "integral(" + render(e->args[0], kPrecAdd);
        for (std::size_t i = 1; i < e->args.size(); ++i) {
            text += ", " + render(e->args[i], kPrecAdd);
        }
        return text + ")";
    }
    case Head::Limit:
        return "limit(" + render(e->args[0], kPrecAdd) + ", " + render(e->args[1], kPrecAdd) +
               ", " + render(e->args[2], kPrecAdd) + ")";
    }
    return "?";
}

} // namespace

std::string to_string(const ExprRef& e) {
    if (!e) {
        return "undefined";
    }
    return render(e, kPrecAdd);
}

// --- Evaluation --------------------------------------------------------------------

namespace {

Result<double> evaluate_function(const std::string& name, const std::vector<double>& args) {
    const auto arity_error = [&name](std::size_t expected) {
        return std::unexpected(DomainError{
            name, "expected " + std::to_string(expected) + " argument(s)"});
    };
    if (args.size() == 1) {
        const double x = args[0];
        if (name == "sin") return std::sin(x);
        if (name == "cos") return std::cos(x);
        if (name == "tan") return std::tan(x);
        if (name == "asin") return std::asin(x);
        if (name == "acos") return std::acos(x);
        if (name == "atan") return std::atan(x);
        if (name == "sinh") return std::sinh(x);
        if (name == "cosh") return std::cosh(x);
        if (name == "tanh") return std::tanh(x);
        if (name == "exp") return std::exp(x);
        if (name == "abs") return std::abs(x);
        if (name == "log") {
            if (x <= 0.0) {
                return std::unexpected(DomainError{"log", "expected a positive argument"});
            }
            return std::log(x);
        }
        if (name == "sqrt") {
            if (x < 0.0) {
                return std::unexpected(DomainError{"sqrt", "expected a non-negative argument"});
            }
            return std::sqrt(x);
        }
    }
    if (name == "atan2") {
        if (args.size() != 2) {
            return arity_error(2);
        }
        return std::atan2(args[0], args[1]);
    }
    if (name == "hypot") {
        if (args.size() != 2) {
            return arity_error(2);
        }
        return std::hypot(args[0], args[1]);
    }
    return std::unexpected(DomainError{"evaluate", "unknown function: " + name});
}

} // namespace

Result<double> evaluate(const ExprRef& e, const std::map<std::string, double>& env) {
    if (!e) {
        return std::unexpected(DomainError{"evaluate", "empty expression"});
    }
    switch (e->head) {
    case Head::Integer:
    case Head::Rational:
    case Head::Real: {
        double value = 0.0;
        as_double(e, value);
        return value;
    }
    case Head::Symbol: {
        const std::string& name = *std::get_if<std::string>(&e->atom);
        const auto found = env.find(name);
        if (found == env.end()) {
            // The old sym_eval returned 0.0 here, which is indistinguishable from a
            // legitimate answer -- and is how sym_eval("pi") came to report 0.000000.
            return std::unexpected(DomainError{"evaluate", "unbound symbol: " + name});
        }
        return found->second;
    }
    case Head::Constant: {
        const std::string& name = *std::get_if<std::string>(&e->atom);
        if (name == "pi") return 3.14159265358979323846;
        if (name == "e") return 2.71828182845904523536;
        if (name == "inf") return std::numeric_limits<double>::infinity();
        if (name == "-inf") return -std::numeric_limits<double>::infinity();
        return std::unexpected(DomainError{"evaluate", "no numeric value for " + name});
    }
    case Head::Add:
    case Head::Mul: {
        double total = e->head == Head::Add ? 0.0 : 1.0;
        for (const ExprRef& arg : e->args) {
            const auto value = evaluate(arg, env);
            if (!value) {
                return value;
            }
            total = e->head == Head::Add ? total + *value : total * *value;
        }
        return total;
    }
    case Head::Pow: {
        const auto base = evaluate(e->args[0], env);
        if (!base) {
            return base;
        }
        const auto exponent = evaluate(e->args[1], env);
        if (!exponent) {
            return exponent;
        }
        if (*base == 0.0 && *exponent < 0.0) {
            return std::unexpected(DomainError{"evaluate", "division by zero"});
        }
        if (*base < 0.0 && *exponent != std::floor(*exponent)) {
            return std::unexpected(
                DomainError{"evaluate", "a real root of a negative number is not real"});
        }
        return std::pow(*base, *exponent);
    }
    case Head::Function: {
        std::vector<double> args;
        args.reserve(e->args.size());
        for (const ExprRef& arg : e->args) {
            const auto value = evaluate(arg, env);
            if (!value) {
                return value;
            }
            args.push_back(*value);
        }
        return evaluate_function(*std::get_if<std::string>(&e->atom), args);
    }
    default:
        break;
    }
    return std::unexpected(
        DomainError{"evaluate", "no numeric value for " + to_string(e)});
}

} // namespace ms::sym2
