// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief The one tree walk of §11.1, and the structural decisions every notation
///   shares.
///
/// `to_string` in `expr.cpp` made these decisions once for ASCII. This makes them once
/// for everything, `to_string` included -- `Notation::Ascii` goes through here, so the
/// ASCII output cannot drift away from the other six by being maintained separately.

#include "ms/sym2/notation.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "ms/core/format.hpp"

#include "notation_syntax.hpp"

namespace ms::sym2::detail {

using bignum::BigInt;
using bignum::Rational;

namespace {

// Binding strength, loosest first, exactly as `expr.cpp` uses it. A fragment needs a
// grouping when it binds more loosely than the slot it is going into.
constexpr int kPrecOpen = 0; ///< a slot the notation fences itself: nothing groups
constexpr int kPrecAdd = 1;
constexpr int kPrecMul = 2;
constexpr int kPrecPow = 3;
constexpr int kPrecAtom = 4;

/// A rendered subexpression together with how tightly what came out actually binds.
/// The precedence is a property of the *text*, not of the node: `x*y^-1` is a Mul but
/// comes out of a LaTeX table as `\frac{x}{y}`, which binds like an atom, and grouping
/// it would be wrong. A notation that spells the same node differently gets a
/// different precedence here, which is why this travels with the string.
struct Fragment {
    std::string text;
    int precedence = kPrecAtom;
};

bool is_negative_number(const ExprRef& e) {
    switch (e->head) {
    case Head::Integer:
        return std::get_if<BigInt>(&e->atom)->negative;
    case Head::Rational:
        return std::get_if<Rational>(&e->atom)->num.negative;
    case Head::Real:
        return *std::get_if<double>(&e->atom) < 0.0;
    default:
        return false;
    }
}

bool rational_is_integer(const Rational& value) { return value.den.is_one(); }

Fragment walk(const ExprRef& e, const Syntax& syntax);

/// Put `f` into a slot of strength `context`, adding a grouping if what was rendered
/// binds more loosely than the slot. In a notation whose nesting groups on its own
/// this is the identity -- `needs_grouping()` is asked rather than assumed, because a
/// stray `<mrow>(</mrow>` in MathML is not a delimiter, it is a literal parenthesis
/// character in the rendered output.
std::string place(const ExprRef& e, int context, const Syntax& syntax) {
    const Fragment f = walk(e, syntax);
    if (!syntax.needs_grouping() || f.precedence >= context) {
        return f.text;
    }
    return syntax.group(f.text);
}

/// The negation of a term, for `a - b` rather than `a + -1*b`. False when the term is
/// not negative, so the caller keeps the `+`.
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

Fragment walk_add(const ExprRef& e, const Syntax& syntax) {
    // The node holds its arguments in canonical order, which puts the constant first
    // because numeric heads sort first. Nobody writes sums that way -- `2x + 1`, not
    // `1 + 2x`. Display order is the printer's business and canonical order is the
    // node's, and they are allowed to differ.
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

    std::vector<SumTerm> terms;
    terms.reserve(display.size());
    for (const ExprRef& term : display) {
        ExprRef positive;
        const bool negative = negated_for_display(term, positive);
        terms.push_back({place(negative ? positive : term, kPrecAdd + 1, syntax), negative});
    }
    return {syntax.sum(terms), kPrecAdd};
}

Fragment walk_mul(const ExprRef& e, const Syntax& syntax) {
    // A factor with a negative exponent is the denominator. `x*y^-1` is correct and
    // unreadable; `x/y` is what anyone writing it down would put.
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
            // p/q inside a product is a numerator of p and a denominator of q, so
            // `2/3*x` comes out `\frac{2x}{3}` rather than `\frac{2}{3} \cdot x`.
            if (exact.num.negative) {
                negative = true;
                BigInt magnitude = exact.num;
                magnitude.negative = false;
                if (!magnitude.is_one()) {
                    numerator.push_back(integer(magnitude));
                }
            } else if (!exact.num.is_one()) {
                numerator.push_back(integer(exact.num));
            }
            denominator.push_back(integer(exact.den));
            continue;
        }
        if (is_number(factor) && is_negative_number(factor) && e->args.size() > 1) {
            const ExprRef magnitude = neg(factor);
            negative = true;
            if (!is_one(magnitude)) {
                numerator.push_back(magnitude);
            }
            continue;
        }
        numerator.push_back(factor);
    }

    // Display order again. The node sorts a Symbol before a Pow, so `x^2*y` leaves the
    // constructor as [y, x^2]; ordering the factors by the name of their base puts
    // `x^2` first and leaves the node itself untouched.
    const auto display_key = [&syntax](const ExprRef& factor) {
        if (is_number(factor)) {
            return std::string{"\x01"}; // the coefficient stays in front
        }
        return to_string(factor->head == Head::Pow ? factor->args[0] : factor);
    };
    (void)syntax;
    std::stable_sort(numerator.begin(), numerator.end(),
                     [&display_key](const ExprRef& a, const ExprRef& b) {
                         return display_key(a) < display_key(b);
                     });

    const auto join = [&syntax](const std::vector<ExprRef>& factors, int context) {
        std::vector<std::string> rendered;
        rendered.reserve(factors.size());
        for (const ExprRef& factor : factors) {
            rendered.push_back(place(factor, context, syntax));
        }
        return rendered;
    };

    Fragment result;
    if (denominator.empty()) {
        if (numerator.empty()) {
            result = {syntax.integer(BigInt(1)), kPrecAtom};
        } else if (numerator.size() == 1) {
            result = walk(numerator.front(), syntax);
        } else {
            result = {syntax.product(join(numerator, kPrecMul)), kPrecMul};
        }
    } else {
        // A fenced quotient (`\frac{}{}`, `<mfrac>`) needs nothing grouped on either
        // side; an infix one needs the denominator grouped as soon as it has more than
        // one factor, or `a/b*c` would say something else entirely.
        const int inner = syntax.quotient_is_fenced() ? kPrecOpen : kPrecMul;
        std::string top;
        if (numerator.empty()) {
            top = syntax.integer(BigInt(1));
        } else if (numerator.size() == 1) {
            top = place(numerator.front(), inner, syntax);
        } else {
            // No grouping on the numerator even when the quotient is infix: an infix
            // `/` binds exactly as tightly as the product and associates to the left in
            // every notation here, so `a*b/c` already groups as `(a*b)/c`. The
            // denominator is the side that needs it, because `a/b*c` does not group as
            // `a/(b*c)`.
            top = syntax.product(join(numerator, std::max(inner, kPrecMul)));
        }
        std::string bottom;
        if (denominator.size() == 1) {
            bottom = place(denominator.front(), syntax.quotient_is_fenced() ? kPrecOpen
                                                                            : kPrecMul + 1,
                           syntax);
        } else {
            bottom = syntax.product(join(denominator, std::max(inner, kPrecMul)));
            if (!syntax.quotient_is_fenced()) {
                bottom = syntax.group(bottom);
            }
        }
        result = {syntax.quotient(top, bottom),
                  syntax.quotient_is_fenced() ? kPrecAtom : kPrecMul};
    }

    if (negative) {
        result = {syntax.negate(result.precedence < kPrecMul && syntax.needs_grouping()
                                    ? syntax.group(result.text)
                                    : result.text),
                  kPrecAdd};
    }
    return result;
}

Fragment walk_pow(const ExprRef& e, const Syntax& syntax) {
    // A negative exponent is a reciprocal. `x^{-1}` is correct and nobody writes it;
    // inside a product `walk_mul` has already moved such a factor into the denominator,
    // and this is the same rule for a power standing on its own.
    if (is_negative_number(e->args[1])) {
        const ExprRef magnitude = neg(e->args[1]);
        const ExprRef reciprocal = is_one(magnitude) ? e->args[0] : pow(e->args[0], magnitude);
        const int inner = syntax.quotient_is_fenced() ? kPrecOpen : kPrecMul + 1;
        return {syntax.quotient(syntax.integer(BigInt(1)), place(reciprocal, inner, syntax)),
                syntax.quotient_is_fenced() ? kPrecAtom : kPrecMul};
    }

    // `x^(1/n)` is an nth root. Only 1/n: `x^(2/3)` is a root of a power or a power of a
    // root depending on the sign of x, and picking one silently is exactly the class of
    // decision this file exists not to make. It stays a power, which is unambiguous.
    Rational exponent;
    if (syntax.options().roots_as_radicals && as_rational(e->args[1], exponent) &&
        exponent.num.is_one() && !exponent.den.is_one()) {
        const std::string degree =
            exponent.den == BigInt(2) ? std::string{} : syntax.integer(exponent.den);
        return {syntax.root(place(e->args[0], syntax.root_is_fenced() ? kPrecOpen : kPrecAtom,
                                  syntax),
                            degree),
                kPrecAtom};
    }

    if (syntax.power_is_a_call()) {
        // A call fences both arguments, so neither is grouped and the result is an atom.
        return {syntax.power(place(e->args[0], kPrecOpen, syntax),
                             place(e->args[1], kPrecOpen, syntax)),
                kPrecAtom};
    }
    // Right-associative: the exponent needs no grouping of its own unless the notation
    // leaves it unfenced, in which case anything looser than a power does.
    return {syntax.power(place(e->args[0], kPrecPow + 1, syntax),
                         place(e->args[1], syntax.exponent_is_fenced() ? kPrecOpen : kPrecPow,
                               syntax)),
            kPrecPow};
}

Fragment walk(const ExprRef& e, const Syntax& syntax) {
    switch (e->head) {
    case Head::Integer: {
        const BigInt& value = *std::get_if<BigInt>(&e->atom);
        return {syntax.integer(value), value.negative ? kPrecAdd : kPrecAtom};
    }
    case Head::Rational: {
        const Rational& value = *std::get_if<Rational>(&e->atom);
        const int precedence = syntax.quotient_is_fenced()
                                   ? (value.num.negative ? kPrecAdd : kPrecAtom)
                                   : kPrecMul;
        return {syntax.rational(value.num, value.den), precedence};
    }
    case Head::Real: {
        const double value = *std::get_if<double>(&e->atom);
        if (!syntax.real_is_atomic(value)) {
            // The spelling is a product (LaTeX's `1 \times 10^{20}`), so it binds like
            // one and a tighter slot has to group it.
            return {syntax.real(value), kPrecMul};
        }
        return {syntax.real(value), value < 0.0 ? kPrecAdd : kPrecAtom};
    }
    case Head::Symbol: {
        std::string base;
        std::string subscript;
        split_subscript(*std::get_if<std::string>(&e->atom), base, subscript);
        return {syntax.symbol(base, subscript), kPrecAtom};
    }
    case Head::Constant:
        return {syntax.constant(*std::get_if<std::string>(&e->atom)), kPrecAtom};
    case Head::Add:
        return walk_add(e, syntax);
    case Head::Mul:
        return walk_mul(e, syntax);
    case Head::Pow:
        return walk_pow(e, syntax);
    case Head::Function: {
        std::vector<std::string> args;
        args.reserve(e->args.size());
        for (const ExprRef& arg : e->args) {
            args.push_back(place(arg, kPrecOpen, syntax));
        }
        return {syntax.call(*std::get_if<std::string>(&e->atom), args), kPrecAtom};
    }
    case Head::Derivative: {
        std::vector<std::string> vars;
        vars.reserve(e->args.size() - 1);
        for (std::size_t i = 1; i < e->args.size(); ++i) {
            vars.push_back(place(e->args[i], kPrecOpen, syntax));
        }
        // `\frac{d}{dx} f + g` reads as the derivative of f plus g, so the operators
        // that carry a trailing body bind loosely and get grouped in any tighter slot.
        return {syntax.derivative(place(e->args[0], kPrecOpen, syntax), vars), kPrecAdd};
    }
    case Head::Integral: {
        std::vector<std::string> vars;
        vars.reserve(e->args.size() - 1);
        for (std::size_t i = 1; i < e->args.size(); ++i) {
            vars.push_back(place(e->args[i], kPrecOpen, syntax));
        }
        return {syntax.integral(place(e->args[0], kPrecOpen, syntax), vars), kPrecAdd};
    }
    case Head::Limit:
        return {syntax.limit(place(e->args[0], kPrecOpen, syntax),
                             place(e->args[1], kPrecOpen, syntax),
                             place(e->args[2], kPrecOpen, syntax)),
                kPrecAdd};
    }
    return {syntax.constant("undefined"), kPrecAtom};
}

} // namespace

void split_subscript(const std::string& name, std::string& base, std::string& subscript) {
    const std::size_t at = name.rfind('_');
    if (at == std::string::npos || at == 0 || at + 1 == name.size()) {
        base = name;
        subscript.clear();
        return;
    }
    base = name.substr(0, at);
    subscript = name.substr(at + 1);
}

std::string greek_letter(const std::string& base, bool& capital) {
    static const char* const kLetters[] = {
        "alpha", "beta",  "gamma",   "delta", "epsilon", "zeta",  "eta",     "theta",
        "iota",  "kappa", "lambda",  "mu",    "nu",      "xi",    "omicron", "pi",
        "rho",   "sigma", "tau",     "upsilon", "phi",   "chi",   "psi",     "omega",
        "varepsilon", "vartheta", "varpi", "varrho", "varsigma", "varphi",
    };
    capital = false;
    if (base.empty()) {
        return {};
    }
    std::string lowered;
    lowered.reserve(base.size());
    for (const char c : base) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    for (const char* letter : kLetters) {
        if (lowered == letter) {
            // `Alpha` and `ALPHA` are the capital; `alpha` is not. A name that differs
            // from its lower-case form only after the first character (`aLPHA`) is not
            // a spelling anyone means, and is treated as lower case rather than guessed
            // at.
            capital = std::isupper(static_cast<unsigned char>(base[0])) != 0;
            return lowered;
        }
    }
    return {};
}

std::string apply_decimal_separator(std::string value, char separator) {
    if (separator == '.') {
        return value;
    }
    const std::size_t at = value.find('.');
    if (at != std::string::npos) {
        value[at] = separator;
    }
    return value;
}

std::string xml_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        switch (c) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default: out.push_back(c); break;
        }
    }
    return out;
}

std::string render(const ExprRef& e, const Syntax& syntax) {
    if (!e) {
        return syntax.constant("undefined");
    }
    return walk(e, syntax).text;
}

std::string render_matrix(const Matrix<double>& m, const Syntax& syntax) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(m.rows());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        std::vector<std::string> row;
        row.reserve(m.cols());
        for (std::size_t j = 0; j < m.cols(); ++j) {
            row.push_back(syntax.real(m(i, j)));
        }
        rows.push_back(std::move(row));
    }
    return syntax.matrix(rows);
}

} // namespace ms::sym2::detail
