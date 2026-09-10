// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

/// @file
/// @brief The syntax table a notation supplies to the §11.1 walker.
///
/// Everything here is spelling. Nothing here decides structure -- by the time a method
/// on this interface is called, the walker in `notation.cpp` has already worked out
/// that this product is really a quotient, that this power is really a cube root, that
/// this sum has a subtraction in it, and which of the pieces need a grouping. A table
/// receives the pieces already rendered and says how they go together in its notation.
///
/// That division is the whole point of §11.1's "one visitor interface and five tables".
/// A structural decision made in a table is a decision made seven times, and six of
/// those copies are in formats nobody reads closely enough to catch an error in.

#include <memory>
#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/sym2/notation.hpp"

namespace ms::sym2::detail {

/// One addend of a sum, with the sign the walker decided it should be shown with.
/// `text` is the *magnitude*: for `a - b` the walker hands over `b` with
/// `negative = true`, having already stripped the `-1` factor.
struct SumTerm {
    std::string text;
    bool negative = false;
};

class Syntax {
public:
    explicit Syntax(const NotationOptions& options) : options_(options) {}
    virtual ~Syntax() = default;

    Syntax(const Syntax&) = delete;
    Syntax& operator=(const Syntax&) = delete;

    const NotationOptions& options() const { return options_; }

    // --- Atoms ----------------------------------------------------------------------

    virtual std::string integer(const bignum::BigInt& value) const = 0;
    /// An exact p/q. The default spells it as a quotient of two integers, which is
    /// right for every notation that has one; a notation whose reader would turn that
    /// into a float (SymPy, and every source language) overrides it.
    virtual std::string rational(const bignum::BigInt& num, const bignum::BigInt& den) const {
        return quotient(integer(num), integer(den));
    }
    /// A value with no exact form. Whatever comes out has to read back as the same
    /// double -- this is an expression, not a display, so it may not round.
    virtual std::string real(double value) const = 0;
    /// The name split at its last underscore: `x_1` arrives as ("x", "1"), `x` as
    /// ("x", ""), `hbar` as ("hbar", ""). Recognising `alpha` or `hbar` in the base is
    /// the table's business, since the spelling differs in every notation.
    virtual std::string symbol(const std::string& base, const std::string& subscript) const = 0;
    /// One of "pi", "e", "i", "inf", "-inf", "nan", "undefined".
    virtual std::string constant(const std::string& name) const = 0;

    // --- Structure ------------------------------------------------------------------

    virtual std::string sum(const std::vector<SumTerm>& terms) const = 0;
    /// Two or more factors, already grouped where they needed it.
    virtual std::string product(const std::vector<std::string>& factors) const = 0;
    virtual std::string quotient(const std::string& numerator,
                                 const std::string& denominator) const = 0;
    virtual std::string power(const std::string& base, const std::string& exponent) const = 0;
    /// `degree` is empty for a square root.
    virtual std::string root(const std::string& radicand, const std::string& degree) const = 0;
    virtual std::string negate(const std::string& operand) const = 0;
    virtual std::string call(const std::string& name,
                             const std::vector<std::string>& args) const = 0;
    /// A grouping around `inner`. Never called when `needs_grouping()` is false.
    virtual std::string group(const std::string& inner) const = 0;

    // --- Heads that stand for something unevaluated ----------------------------------

    virtual std::string derivative(const std::string& body,
                                   const std::vector<std::string>& vars) const = 0;
    virtual std::string integral(const std::string& body,
                                 const std::vector<std::string>& vars) const = 0;
    virtual std::string limit(const std::string& body, const std::string& var,
                              const std::string& point) const = 0;
    virtual std::string matrix(const std::vector<std::vector<std::string>>& rows) const = 0;

    // --- What the walker must ask rather than assume ----------------------------------

    /// False when the notation's own element nesting groups on its own -- Content
    /// MathML's `<apply>` and Presentation MathML's `<mfrac>` each take a fixed number
    /// of children, so no arrangement of them is ambiguous and a parenthesis would be
    /// noise in the markup rather than a delimiter. The walker then emits no groupings
    /// at all; it does not call `group()` and get an empty string back, because a
    /// table that has to return "" for a structural request is a table being asked the
    /// wrong question.
    virtual bool needs_grouping() const { return true; }
    /// True when `power()` is a function call rather than an infix operator, as in C.
    /// A call fences its own arguments, so the walker skips grouping the base -- which
    /// otherwise would be grouped, since a power binds tighter than everything.
    virtual bool power_is_a_call() const { return false; }
    /// True when `quotient()` fences both sides itself -- `\frac{a+b}{c}` needs no
    /// parentheses where `(a+b)/c` does.
    virtual bool quotient_is_fenced() const { return false; }
    /// True when `root()` fences its radicand -- `\sqrt{a+b}` against `sqrt(a + b)`,
    /// both of which do, against a hypothetical `√a+b`, which does not.
    virtual bool root_is_fenced() const { return true; }
    /// True when the exponent of an infix power is fenced by the notation itself, as
    /// LaTeX's `x^{a+b}` is by its braces. False for `x**(a+b)`, which needs them.
    virtual bool exponent_is_fenced() const { return false; }

    /// True when what `real(value)` produced is a single token that binds like an atom.
    ///
    /// Almost always it is. LaTeX is the exception: `1e20` cannot be written `1e20`
    /// there -- math mode would set it as the product of 1 and Euler's number, plus 20
    /// -- so it comes out `1 \times 10^{20}`, which is a product. Treating that as an
    /// atom puts it straight into a superscript slot and produces `1 \times
    /// 10^{20}^{2}`: a double superscript, which is a LaTeX error rather than a
    /// mis-set formula. The walker asks, because the answer depends on the value as
    /// well as the notation.
    virtual bool real_is_atomic(double value) const {
        (void)value;
        return true;
    }

private:
    NotationOptions options_;
};

/// Split `name` at its last underscore into a base and a subscript, when doing so
/// leaves both halves non-empty. `x_1` -> ("x", "1"); `x_` and `_x` and `x` stay whole.
void split_subscript(const std::string& name, std::string& base, std::string& subscript);

/// The Greek letter `base` names, in lower case and without a backslash, or the empty
/// string when it names none. `alpha`, `Alpha`, `ALPHA` all give "alpha"; the caller
/// gets `capital` set for the two capitalised spellings so it can pick `\Alpha` or `α`.
std::string greek_letter(const std::string& base, bool& capital);

/// `value` with `separator` in place of the decimal point, when it is not '.'.
std::string apply_decimal_separator(std::string value, char separator);

/// XML text with `& < > " '` replaced by entities.
std::string xml_escape(const std::string& text);

// --- The tables ---------------------------------------------------------------------
//
// One per translation unit, so a notation's spellings are all in one file and adding a
// notation touches nothing that already works.

std::unique_ptr<Syntax> make_latex_syntax(const NotationOptions& options);
std::unique_ptr<Syntax> make_presentation_mathml_syntax(const NotationOptions& options);
std::unique_ptr<Syntax> make_content_mathml_syntax(const NotationOptions& options);
std::unique_ptr<Syntax> make_unicode_syntax(const NotationOptions& options);
std::unique_ptr<Syntax> make_ascii_syntax(const NotationOptions& options);
std::unique_ptr<Syntax> make_sympy_syntax(const NotationOptions& options);
std::unique_ptr<Syntax> make_mathematica_syntax(const NotationOptions& options);
/// `language` is `Notation::C`, `Notation::Cpp` or `Notation::Python`.
std::unique_ptr<Syntax> make_source_syntax(Notation language, const NotationOptions& options);

/// The walk itself. Every notation goes through this and differs only in `syntax`.
std::string render(const ExprRef& e, const Syntax& syntax);
std::string render_matrix(const Matrix<double>& m, const Syntax& syntax);

} // namespace ms::sym2::detail
