// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief SymPy and Wolfram Language spellings for the §11.1 walker.
///
/// These two exist to be pasted into a session someone already has open, so the
/// property that matters is not that the string looks right but that the *other
/// system's parser* builds the expression that was printed. Both have a trap, and the
/// two traps are opposites of each other, which is why they are in one file where the
/// contrast is visible:
///
///   - In Python, `1/3` is `0.333...`. An exact rational printed that way stops being
///     exact the moment the reader evaluates it, and every later result is a float.
///     SymPy's answer is `Rational(1, 3)`.
///   - In Wolfram Language, `1/3` *is* exact and stays exact. The same three
///     characters, the opposite meaning.
///
/// The second Wolfram trap is syntactic rather than numeric: square brackets are
/// function application and parentheses are grouping only. `Sin(x)` is not a call
/// there -- it is the symbol `Sin` multiplied by `x`, which evaluates happily and
/// silently to something else.

#include <cctype>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ms/core/format.hpp"

#include "notation_syntax.hpp"

namespace ms::sym2::detail {

namespace {

using bignum::BigInt;

/// A name in the target system for each name `ms::sym2` uses, or nothing when the
/// target spells it the same way.
struct NameMap {
    const char* ours;
    const char* theirs;
};

// --- SymPy ---------------------------------------------------------------------------

constexpr NameMap kSympyFunctions[] = {
    {"abs", "Abs"},   // Python's builtin abs is not SymPy's, and does not stay symbolic
    {"min", "Min"}, {"max", "Max"},
    {"asin", "asin"}, {"acos", "acos"}, {"atan", "atan"},
    {"atan2", "atan2"}, {"hypot", "sqrt"}, // hypot has no SymPy name; see call()
};

class SympySyntax : public Syntax {
public:
    using Syntax::Syntax;

    std::string integer(const BigInt& value) const override { return value.to_string(); }

    /// The whole reason this override exists. `quotient()` cannot do it, because by the
    /// time the walker calls `quotient()` it has already split `2/3 * x` into a
    /// numerator and a denominator and the exactness is no longer visible there. A
    /// standalone `Rational` atom reaches `rational()` and only `rational()`, so this
    /// is the one place the distinction survives.
    std::string rational(const BigInt& num, const BigInt& den) const override {
        return "Rational(" + num.to_string() + ", " + den.to_string() + ")";
    }

    std::string real(double value) const override {
        if (std::isnan(value)) return "nan";
        if (std::isinf(value)) return value < 0.0 ? "-oo" : "oo";
        return ms::format_exact(value);
    }

    /// A subscripted name stays `x_1`, which is a valid Python identifier and what
    /// `Symbol("x_1")` is called. A Greek name stays as its ASCII spelling for the same
    /// reason: `α` is a legal Python 3 identifier but not one anybody types.
    std::string symbol(const std::string& base, const std::string& subscript) const override {
        return subscript.empty() ? base : base + "_" + subscript;
    }

    std::string constant(const std::string& name) const override {
        if (name == "pi") return "pi";
        if (name == "e") return "E";
        if (name == "i") return "I";
        if (name == "inf") return "oo";
        if (name == "-inf") return "-oo";
        if (name == "nan" || name == "undefined") return "nan";
        return name;
    }

    std::string sum(const std::vector<SumTerm>& terms) const override {
        std::string out;
        for (std::size_t i = 0; i < terms.size(); ++i) {
            if (i == 0) {
                out += terms[i].negative ? "-" : "";
            } else {
                out += terms[i].negative ? " - " : " + ";
            }
            out += terms[i].text;
        }
        return out;
    }

    std::string product(const std::vector<std::string>& factors) const override {
        std::string out;
        for (std::size_t i = 0; i < factors.size(); ++i) {
            if (i > 0) out += "*";
            out += factors[i];
        }
        return out;
    }

    std::string quotient(const std::string& numerator,
                         const std::string& denominator) const override {
        return numerator + "/" + denominator;
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        return base + "**" + exponent;
    }

    /// `sqrt(x)` for a square root; anything else as `root(x, n)`, which SymPy has and
    /// which keeps the degree exact. Writing it `x**Rational(1, n)` would be the same
    /// expression, but `root` is what SymPy prints back and the round trip is shorter.
    std::string root(const std::string& radicand, const std::string& degree) const override {
        return degree.empty() ? "sqrt(" + radicand + ")"
                              : "root(" + radicand + ", " + degree + ")";
    }

    std::string negate(const std::string& operand) const override { return "-" + operand; }

    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        // hypot is not a SymPy function. Emitting `hypot(a, b)` would be a NameError at
        // best; written out as sqrt(a**2 + b**2) it is the same value and it evaluates.
        if (name == "hypot" && args.size() == 2) {
            return "sqrt(" + args[0] + "**2 + " + args[1] + "**2)";
        }
        std::string out = mapped(name, kSympyFunctions) + "(";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) out += ", ";
            out += args[i];
        }
        return out + ")";
    }

    std::string group(const std::string& inner) const override { return "(" + inner + ")"; }

    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        std::string out = "Derivative(" + body;
        for (const std::string& var : vars) out += ", " + var;
        return out + ")";
    }

    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        std::string out = "Integral(" + body;
        for (const std::string& var : vars) out += ", " + var;
        return out + ")";
    }

    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        return "Limit(" + body + ", " + var + ", " + point + ")";
    }

    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        std::string out = "Matrix([";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i > 0) out += ", ";
            out += "[";
            for (std::size_t j = 0; j < rows[i].size(); ++j) {
                if (j > 0) out += ", ";
                out += rows[i][j];
            }
            out += "]";
        }
        return out + "])";
    }

    bool needs_grouping() const override { return true; }
    bool quotient_is_fenced() const override { return false; }
    /// `sqrt(...)` and `root(...)` are calls.
    bool root_is_fenced() const override { return true; }
    /// `x**a + b` is `(x**a) + b`, so a loose exponent needs parentheses. This is also
    /// what keeps `x**-n` from ever being emitted: a negative exponent reaches the
    /// walker's reciprocal branch, and a symbolic negation arrives here already
    /// grouped.
    bool exponent_is_fenced() const override { return false; }

protected:
    template <std::size_t N>
    static std::string mapped(const std::string& name, const NameMap (&table)[N]) {
        for (const NameMap& entry : table) {
            if (name == entry.ours) {
                return entry.theirs;
            }
        }
        return name;
    }
};

// --- Wolfram Language ------------------------------------------------------------------

constexpr NameMap kWolframFunctions[] = {
    {"sin", "Sin"},   {"cos", "Cos"},   {"tan", "Tan"},
    {"asin", "ArcSin"}, {"acos", "ArcCos"}, {"atan", "ArcTan"},
    {"sinh", "Sinh"}, {"cosh", "Cosh"}, {"tanh", "Tanh"},
    {"exp", "Exp"},   {"log", "Log"},   {"sqrt", "Sqrt"},
    {"abs", "Abs"},   {"min", "Min"},   {"max", "Max"},
    {"atan2", "ArcTan"}, // ArcTan[y, x] is the two-argument form; see call()
};

class WolframSyntax : public SympySyntax {
public:
    using SympySyntax::SympySyntax;

    /// A caution this table cannot fix, recorded because it is the one remaining way
    /// a pasted expression can mean something else: Wolfram Language reserves every
    /// capitalised name, so a symbol genuinely called `E`, `D`, `I`, `N`, `C`, `K` or
    /// `O` collides with a built-in and evaluates to it. Renaming it here would be
    /// worse -- the reader would get an expression over a variable they never wrote --
    /// so the name is emitted as given and the collision is the caller's to avoid.

    /// Unlike SymPy, `1/3` in Wolfram Language is an exact rational and stays one.
    /// The same three characters that would silently become a float in a Python
    /// session are the correct spelling here.
    std::string rational(const BigInt& num, const BigInt& den) const override {
        return num.to_string() + "/" + den.to_string();
    }

    /// `1.5e-8` is not a number in Wolfram Language input -- `e` is not an exponent
    /// marker there, so it parses as a product involving a symbol named `e`. The
    /// exponent marker is `*^`.
    std::string real(double value) const override {
        if (std::isnan(value)) return "Indeterminate";
        if (std::isinf(value)) return value < 0.0 ? "-Infinity" : "Infinity";
        std::string text = ms::format_exact(value);
        const std::size_t at = text.find('e');
        if (at == std::string::npos) {
            return text;
        }
        std::string exponent = text.substr(at + 1);
        std::string sign;
        if (!exponent.empty() && (exponent[0] == '+' || exponent[0] == '-')) {
            if (exponent[0] == '-') {
                sign = "-";
            }
            exponent.erase(0, 1);
        }
        // `format_exact` pads the exponent to two digits, so 1.5e-8 arrives as
        // "1.5e-08". `*^-08` is read as `*^-8` by every Wolfram front end, but a
        // zero-padded exponent is not a spelling anyone writes and it is one more
        // thing between the reader and believing the paste.
        while (exponent.size() > 1 && exponent.front() == '0') {
            exponent.erase(0, 1);
        }
        return text.substr(0, at) + "*^" + sign + exponent;
    }

    std::string constant(const std::string& name) const override {
        if (name == "pi") return "Pi";
        if (name == "e") return "E";
        if (name == "i") return "I";
        if (name == "inf") return "Infinity";
        if (name == "-inf") return "-Infinity";
        if (name == "nan") return "Indeterminate";
        if (name == "undefined") return "Undefined";
        return name;
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        return base + "^" + exponent;
    }

    std::string root(const std::string& radicand, const std::string& degree) const override {
        return degree.empty() ? "Sqrt[" + radicand + "]"
                              : "Surd[" + radicand + ", " + degree + "]";
    }

    /// Square brackets, and this is the whole trap. `Sin(x)` is not a call in Wolfram
    /// Language: it is the symbol `Sin` multiplied by `x`, which evaluates without
    /// complaint to something that is not the sine of anything.
    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        if (name == "hypot" && args.size() == 2) {
            return "Sqrt[" + args[0] + "^2 + " + args[1] + "^2]";
        }
        std::string out = mapped(name, kWolframFunctions) + "[";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) out += ", ";
            out += args[i];
        }
        return out + "]";
    }

    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        std::string out = "D[" + body;
        for (const std::string& var : vars) out += ", " + var;
        return out + "]";
    }

    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        std::string out = "Integrate[" + body;
        for (const std::string& var : vars) out += ", " + var;
        return out + "]";
    }

    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        return "Limit[" + body + ", " + var + " -> " + point + "]";
    }

    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        std::string out = "{";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i > 0) out += ", ";
            out += "{";
            for (std::size_t j = 0; j < rows[i].size(); ++j) {
                if (j > 0) out += ", ";
                out += rows[i][j];
            }
            out += "}";
        }
        return out + "}";
    }

    /// `Sqrt[...]` and `Surd[...]` fence with brackets.
    bool root_is_fenced() const override { return true; }
};

} // namespace

std::unique_ptr<Syntax> make_sympy_syntax(const NotationOptions& options) {
    return std::make_unique<SympySyntax>(options);
}

std::unique_ptr<Syntax> make_mathematica_syntax(const NotationOptions& options) {
    return std::make_unique<WolframSyntax>(options);
}

} // namespace ms::sym2::detail
