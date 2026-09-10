// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief C, C++ and Python source for the §11.1 walker.
///
/// The contract here is stronger than for the other notations: what comes out has to
/// *compile*, and it has to evaluate to the same double the expression denotes. A
/// LaTeX printer that sets a formula slightly oddly is a cosmetic problem; a source
/// printer that emits `1/3` in C has emitted zero.
///
/// The three traps, one per language:
///
///   - **C and C++ have integer division.** `1/3` is `0`. Every rational becomes a
///     floating-point quotient and every integer that takes part in arithmetic carries
///     a `.0`. Python 3's `/` is true division, so it does not need this -- but a
///     Python `int` is unbounded where a C one is not, so the limit moves rather than
///     disappearing.
///   - **C and C++ have no power operator.** `pow(b, e)`, and because a call fences its
///     arguments the walker is told to stop grouping the base.
///   - **Python's `**` binds tighter than unary minus**, so `-x**2` is `-(x**2)`, and
///     `2**-1` is a syntax error outright. The walker handles both: a negative numeric
///     base has add-level precedence and gets grouped, and a negative exponent never
///     reaches `power()` at all because the walker turns it into a reciprocal first.
///
/// And one thing no language can express. A derivative, an integral or a limit that
/// was never evaluated has no source form. Emitting a plausible-looking call would be
/// exactly the defect this project spent two audits removing: a value a reader would
/// take for an answer that is not one. These emit an identifier that does not exist,
/// so the compiler stops and the message names the problem.

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

/// The reserved words of all three languages, merged. A symbol named `int` or `class`
/// or `lambda` is a legal MathScript variable and not a legal identifier anywhere here.
const char* const kReserved[] = {
    "alignas", "alignof", "and", "as", "asm", "assert", "auto", "await", "bool", "break",
    "case", "catch", "char", "class", "const", "constexpr", "continue", "def", "default",
    "del", "delete", "do", "double", "elif", "else", "enum", "except", "extern", "false",
    "finally", "float", "for", "from", "global", "goto", "if", "import", "in", "inline",
    "int", "is", "lambda", "long", "namespace", "new", "nonlocal", "none", "not",
    "nullptr", "operator", "or", "pass", "private", "protected", "public", "raise",
    "register", "return", "short", "signed", "sizeof", "static", "struct", "switch",
    "template", "this", "throw", "true", "try", "typedef", "typename", "union",
    "unsigned", "using", "virtual", "void", "volatile", "while", "with", "yield",
};

class SourceSyntax : public Syntax {
public:
    SourceSyntax(const NotationOptions& options, Notation language)
        : Syntax(options), language_(language) {}

    // --- Atoms ----------------------------------------------------------------------

    /// Every integer that reaches arithmetic here is a double, so it carries a `.0`.
    /// Without it, `1/3` in C is integer division and the printed expression denotes
    /// zero. Python does not need the suffix but is given it anyway: one spelling,
    /// and `1.0` is the same value there.
    ///
    /// A BigInt past 2^53 cannot be a double literal at all. It is emitted in full and
    /// the loss is real -- silently truncating it would be the worse of the two, since
    /// a truncated literal still compiles.
    std::string integer(const BigInt& value) const override {
        return value.to_string() + ".0";
    }

    std::string rational(const BigInt& num, const BigInt& den) const override {
        return "(" + num.to_string() + ".0 / " + den.to_string() + ".0)";
    }

    std::string real(double value) const override {
        if (std::isnan(value)) {
            return constant("nan");
        }
        if (std::isinf(value)) {
            return constant(value < 0.0 ? "-inf" : "inf");
        }
        std::string text = ms::format_exact(value);
        // `2` is an int literal in C; `2.0` is a double. format_exact emits the shortest
        // round-tripping form, which for an integral value has no point in it.
        if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
            text.find("inf") == std::string::npos) {
            text += ".0";
        }
        return text;
    }

    std::string symbol(const std::string& base, const std::string& subscript) const override {
        const std::string name = subscript.empty() ? base : base + "_" + subscript;
        if (!options().source_variable_array.empty()) {
            std::size_t index = 0;
            if (indexed_name(name, index)) {
                return options().source_variable_array + "[" + std::to_string(index) + "]";
            }
            // Anything that is not x0, x1, ... is emitted as an identifier. Assigning it
            // an array slot would need an ordering over the whole expression, which a
            // table method cannot see -- and inventing one would put two different
            // symbols in the same slot on two different calls.
        }
        return identifier(name);
    }

    std::string constant(const std::string& name) const override {
        const bool python = language_ == Notation::Python;
        if (name == "pi") {
            // M_PI is not standard C++ without _USE_MATH_DEFINES and is not standard C
            // at all, so a literal with enough digits to be the same double is the only
            // spelling that is correct in every configuration.
            return python ? "math.pi" : "3.141592653589793";
        }
        if (name == "e") {
            return python ? "math.e" : "2.718281828459045";
        }
        if (name == "inf") {
            return python ? "math.inf" : "INFINITY";
        }
        if (name == "-inf") {
            return python ? "-math.inf" : "-INFINITY";
        }
        if (name == "nan") {
            return python ? "math.nan" : "NAN";
        }
        if (name == "i") {
            // The imaginary unit has no value in a double expression. Emitting 0, or a
            // NaN, or anything at all that compiles would hand back a number that is
            // not the one the expression denotes.
            return unevaluated("imaginary_unit");
        }
        return unevaluated("undefined");
    }

    // --- Structure ------------------------------------------------------------------

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
            if (i > 0) out += " * ";
            out += factors[i];
        }
        return out;
    }

    std::string quotient(const std::string& numerator,
                         const std::string& denominator) const override {
        return numerator + " / " + denominator;
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        if (language_ == Notation::Python) {
            return base + "**" + exponent;
        }
        return function_name("pow") + "(" + base + ", " + exponent + ")";
    }

    std::string root(const std::string& radicand, const std::string& degree) const override {
        if (degree.empty()) {
            return function_name("sqrt") + "(" + radicand + ")";
        }
        if (degree == "3") {
            return function_name("cbrt") + "(" + radicand + ")";
        }
        // An nth root of a negative number is not what pow() computes, so this is only
        // correct where the radicand is non-negative -- which is the same restriction
        // the expression itself carries, since x^(1/n) is what was printed.
        return power(radicand, "(1.0 / " + degree + ".0)");
    }

    std::string negate(const std::string& operand) const override { return "-" + operand; }

    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        std::string out = function_name(name) + "(";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) out += ", ";
            out += args[i];
        }
        return out + ")";
    }

    std::string group(const std::string& inner) const override { return "(" + inner + ")"; }

    // --- Heads with no source form ----------------------------------------------------

    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        return unevaluated_call("derivative", body, vars);
    }

    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        return unevaluated_call("integral", body, vars);
    }

    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        return unevaluated_call("limit", body, {var, point});
    }

    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        const bool python = language_ == Notation::Python;
        const char open_outer = python ? '[' : '{';
        const char close_outer = python ? ']' : '}';
        std::string out(1, open_outer);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i > 0) out += ", ";
            out += open_outer;
            for (std::size_t j = 0; j < rows[i].size(); ++j) {
                if (j > 0) out += ", ";
                out += rows[i][j];
            }
            out += close_outer;
        }
        out += close_outer;
        return out;
    }

    // --- What the walker must ask rather than assume ----------------------------------

    bool needs_grouping() const override { return true; }
    bool quotient_is_fenced() const override { return false; }
    /// `sqrt(...)`, `cbrt(...)` and the `pow(...)` fallback are all calls.
    bool root_is_fenced() const override { return true; }
    /// True for C and C++, where `power()` is `pow(b, e)` and fences both arguments.
    bool power_is_a_call() const override { return language_ != Notation::Python; }
    /// Python's `**` is infix and does not fence, so `x**(a + b)` needs its
    /// parentheses.
    bool exponent_is_fenced() const override { return false; }

private:
    /// `ms_unevaluated_<what>` -- an identifier no translation unit declares. The code
    /// does not compile, and the message names what could not be expressed. That is the
    /// honest outcome: the alternative is a call that compiles and returns a number the
    /// expression does not denote.
    static std::string unevaluated(const std::string& what) {
        return "ms_unevaluated_" + what;
    }

    static std::string unevaluated_call(const std::string& what, const std::string& body,
                                        const std::vector<std::string>& rest) {
        std::string out = unevaluated(what) + "(" + body;
        for (const std::string& piece : rest) {
            out += ", " + piece;
        }
        return out + ")";
    }

    /// The library spelling of a libm function in the target language.
    std::string function_name(const std::string& name) const {
        if (language_ == Notation::Python) {
            if (name == "abs") return "abs";       // the builtin, which is correct for a float
            if (name == "pow") return "pow";
            if (name == "min") return "min";
            if (name == "max") return "max";
            return "math." + name;
        }
        std::string base = name;
        // fabs, not abs: in C, abs() takes an int, so abs(-1.5) truncates to 1 before
        // it is called. That is a wrong answer that compiles, which is why it is the
        // one rename worth making rather than leaving to the reader.
        if (name == "abs") base = "fabs";
        if (name == "min") base = "fmin";
        if (name == "max") base = "fmax";
        return language_ == Notation::Cpp ? "std::" + base : base;
    }

    /// `x12` -> 12, for the `source_variable_array` mapping. Anything else is not an
    /// indexed name and the caller keeps the identifier.
    static bool indexed_name(const std::string& name, std::size_t& index) {
        if (name.size() < 2 || name[0] != 'x') {
            return false;
        }
        index = 0;
        for (std::size_t i = 1; i < name.size(); ++i) {
            if (std::isdigit(static_cast<unsigned char>(name[i])) == 0) {
                return false;
            }
            index = index * 10 + static_cast<std::size_t>(name[i] - '0');
        }
        return true;
    }

    /// A MathScript name that is not a legal identifier here, made into one. Every
    /// illegal byte becomes `_xHH`, which is reversible and readable, and a name that
    /// collides with a reserved word or begins with a digit gets a `ms_` prefix.
    static std::string identifier(const std::string& name) {
        std::string out;
        out.reserve(name.size());
        for (const char c : name) {
            const unsigned char byte = static_cast<unsigned char>(c);
            if (std::isalnum(byte) != 0 || c == '_') {
                out.push_back(c);
                continue;
            }
            static const char* const kHex = "0123456789abcdef";
            out += "_x";
            out.push_back(kHex[byte >> 4]);
            out.push_back(kHex[byte & 0x0F]);
        }
        if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0])) != 0) {
            return "ms_" + out;
        }
        for (const char* word : kReserved) {
            if (out == word) {
                return "ms_" + out;
            }
        }
        return out;
    }

    Notation language_;
};

} // namespace

std::unique_ptr<Syntax> make_source_syntax(Notation language, const NotationOptions& options) {
    return std::make_unique<SourceSyntax>(options, language);
}

} // namespace ms::sym2::detail
