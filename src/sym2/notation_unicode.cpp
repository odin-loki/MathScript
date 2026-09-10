// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief The two plain-text spellings for the §11.1 walker: ASCII and Unicode.
///
/// They are one table with a different character set, because the only thing that
/// separates them is which glyphs are available. Writing them as two classes would
/// mean two copies of every structural answer, and the pair would drift the first time
/// one of them was edited.
///
/// The ASCII half has an obligation the others do not: `ms::sym2::to_string` is the
/// notation MathScript itself reads back, so what comes out here has to be an
/// expression the REPL's own parser accepts. That is what makes the round-trip
/// property in `test_sym2_notation_roundtrip` testable at all -- it prints, parses and
/// evaluates, and any precedence defect in the walker shows up as a different number.

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

/// The glyphs that differ between the two. Everything else is shared.
struct Charset {
    const char* times;        ///< the visible multiplication sign
    const char* cross;        ///< what `Multiplication::Cross` asks for
    const char* minus;        ///< the sign in `a - b`
    const char* pi;
    const char* infinity;
    const char* arrow;        ///< `x -> 0` in a limit
    const char* integral;
    bool superscripts;        ///< spell an integer exponent with superscript digits
    bool radicals;            ///< spell a square/cube/fourth root with a radical sign
    bool greek;               ///< spell a Greek name with its letter
    bool subscripts;          ///< spell an all-digit subscript with subscript digits
};

constexpr Charset kAscii{"*", "*", "-", "pi", "inf", "->", "integral", false, false, false, false};
constexpr Charset kUnicode{"·", "×", "-", "π", "∞", "→", "∫",
                           true,     true,     true,          true};

/// The Greek letters, in the order `greek_letter` returns them. Indexing by name keeps
/// the two halves of this file from needing a second copy of the list.
struct GreekGlyph {
    const char* name;
    const char* lower;
    const char* upper;
};

constexpr GreekGlyph kGreek[] = {
    {"alpha", "α", "Α"},   {"beta", "β", "Β"},
    {"gamma", "γ", "Γ"},   {"delta", "δ", "Δ"},
    {"epsilon", "ε", "Ε"}, {"zeta", "ζ", "Ζ"},
    {"eta", "η", "Η"},     {"theta", "θ", "Θ"},
    {"iota", "ι", "Ι"},    {"kappa", "κ", "Κ"},
    {"lambda", "λ", "Λ"},  {"mu", "μ", "Μ"},
    {"nu", "ν", "Ν"},      {"xi", "ξ", "Ξ"},
    {"omicron", "ο", "Ο"}, {"pi", "π", "Π"},
    {"rho", "ρ", "Ρ"},     {"sigma", "σ", "Σ"},
    {"tau", "τ", "Τ"},     {"upsilon", "υ", "Υ"},
    {"phi", "φ", "Φ"},     {"chi", "χ", "Χ"},
    {"psi", "ψ", "Ψ"},     {"omega", "ω", "Ω"},
    {"varepsilon", "ε", "Ε"}, {"vartheta", "ϑ", "Θ"},
    {"varpi", "ϖ", "Π"},      {"varrho", "ϱ", "Ρ"},
    {"varsigma", "ς", "Σ"},   {"varphi", "φ", "Φ"},
};

const char* kSuperscriptDigit[] = {"⁰", "¹", "²", "³", "⁴",
                                   "⁵", "⁶", "⁷", "⁸", "⁹"};
const char* kSubscriptDigit[] = {"₀", "₁", "₂", "₃", "₄",
                                 "₅", "₆", "₇", "₈", "₉"};

class TextSyntax : public Syntax {
public:
    TextSyntax(const NotationOptions& options, const Charset& charset)
        : Syntax(options), charset_(charset) {}

    // --- Atoms ----------------------------------------------------------------------

    std::string integer(const BigInt& value) const override { return value.to_string(); }

    std::string real(double value) const override {
        if (std::isnan(value)) {
            return "nan";
        }
        if (std::isinf(value)) {
            return value < 0.0 ? std::string("-") + charset_.infinity : charset_.infinity;
        }
        return apply_decimal_separator(ms::format_exact(value), options().decimal_separator);
    }

    std::string symbol(const std::string& base, const std::string& subscript) const override {
        std::string out = spell_base(base);
        if (subscript.empty()) {
            return out;
        }
        if (charset_.subscripts && all_digits(subscript)) {
            for (const char c : subscript) {
                out += kSubscriptDigit[c - '0'];
            }
            return out;
        }
        // ASCII keeps the underscore, which is how the name was spelled and how
        // MathScript's own parser reads it back.
        return out + "_" + subscript;
    }

    std::string constant(const std::string& name) const override {
        if (name == "pi") return charset_.pi;
        if (name == "inf") return charset_.infinity;
        if (name == "-inf") return std::string("-") + charset_.infinity;
        if (name == "nan") return "nan";
        if (name == "undefined") return "undefined";
        return name; // "e" and "i" are themselves in both character sets
    }

    // --- Structure ------------------------------------------------------------------

    std::string sum(const std::vector<SumTerm>& terms) const override {
        std::string out;
        for (std::size_t i = 0; i < terms.size(); ++i) {
            if (i == 0) {
                out += terms[i].negative ? charset_.minus : "";
            } else {
                out += terms[i].negative ? std::string(" ") + charset_.minus + " " : " + ";
            }
            out += terms[i].text;
        }
        return out;
    }

    std::string product(const std::vector<std::string>& factors) const override {
        const char* sign = options().multiplication == NotationOptions::Multiplication::Cross
                               ? charset_.cross
                               : charset_.times;
        std::string out;
        for (std::size_t i = 0; i < factors.size(); ++i) {
            if (i > 0) {
                out += sign;
            }
            out += factors[i];
        }
        return out;
    }

    std::string quotient(const std::string& numerator,
                         const std::string& denominator) const override {
        return numerator + "/" + denominator;
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        if (charset_.superscripts) {
            std::string raised;
            if (superscript(exponent, raised)) {
                return base + raised;
            }
        }
        return base + "^" + exponent;
    }

    std::string root(const std::string& radicand, const std::string& degree) const override {
        if (charset_.radicals) {
            if (degree.empty()) return "√" + radicand;
            if (degree == "3") return "∛" + radicand;
            if (degree == "4") return "∜" + radicand;
        }
        // A degree with no radical sign of its own has to be written as the power it
        // is: there is no Unicode glyph for a fifth root, and inventing one out of
        // "√5" would read as the square root of five.
        if (degree.empty()) {
            return "sqrt(" + radicand + ")";
        }
        return "root(" + radicand + ", " + degree + ")";
    }

    std::string negate(const std::string& operand) const override {
        return std::string(charset_.minus) + operand;
    }

    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        std::string out = name + "(";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) {
                out += ", ";
            }
            out += args[i];
        }
        return out + ")";
    }

    std::string group(const std::string& inner) const override { return "(" + inner + ")"; }

    // --- Heads that stand for something unevaluated ----------------------------------

    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        std::string out = "d/d";
        for (const std::string& var : vars) {
            out += var;
        }
        return out + "(" + body + ")";
    }

    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        if (!charset_.radicals) {
            // ASCII: a call, because `integral x dx` is not something MathScript's
            // parser reads and this notation has to be readable back.
            std::string out = "integral(" + body;
            for (const std::string& var : vars) {
                out += ", " + var;
            }
            return out + ")";
        }
        std::string out = std::string(charset_.integral) + " " + body;
        for (const std::string& var : vars) {
            out += " d" + var;
        }
        return out;
    }

    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        if (!charset_.radicals) {
            return "limit(" + body + ", " + var + ", " + point + ")";
        }
        return "lim(" + var + charset_.arrow + point + ") " + body;
    }

    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        // `[a, b; c, d]`, which is this project's own matrix literal syntax, so what
        // comes out of the ASCII table is something a user can paste back in.
        std::string out = "[";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i > 0) {
                out += "; ";
            }
            for (std::size_t j = 0; j < rows[i].size(); ++j) {
                if (j > 0) {
                    out += ", ";
                }
                out += rows[i][j];
            }
        }
        return out + "]";
    }

    // --- What the walker must ask rather than assume ----------------------------------

    bool needs_grouping() const override { return true; }

    /// `a/b` is infix, so both sides can need grouping. `\frac` this is not.
    bool quotient_is_fenced() const override { return false; }

    /// A radical sign in text has no vinculum, so it does not group its argument: `√`
    /// followed by `a + b` is the square root of a, plus b. Unicode therefore says no
    /// and the walker puts real parentheses in. ASCII always writes a call, which
    /// fences by construction, so it says yes and gets `sqrt(a + b)` rather than
    /// `sqrt((a + b))`.
    ///
    /// The one place the two disagree with themselves is Unicode's fallback for a
    /// degree with no glyph -- `root(x, 5)` does fence -- which costs a pair of
    /// parentheses that are correct and redundant. Being wrong in the other direction
    /// would cost the value.
    bool root_is_fenced() const override { return !charset_.radicals; }

    /// `x^a + b` is `(x^a) + b`, so an exponent that is not an atom needs grouping.
    /// The superscript spelling is only ever used for a run of digits, which is one.
    bool exponent_is_fenced() const override { return false; }

private:
    static bool all_digits(const std::string& text) {
        if (text.empty()) {
            return false;
        }
        for (const char c : text) {
            if (std::isdigit(static_cast<unsigned char>(c)) == 0) {
                return false;
            }
        }
        return true;
    }

    /// The superscript form of `text`, when every character of it has one. A minus sign
    /// does; a decimal point, a letter and a parenthesis do not, and a partial
    /// transcription would silently drop them.
    static bool superscript(const std::string& text, std::string& out) {
        out.clear();
        if (text.empty()) {
            return false;
        }
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '-' && i == 0) {
                out += "⁻";
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(text[i])) == 0) {
                return false;
            }
            out += kSuperscriptDigit[text[i] - '0'];
        }
        return true;
    }

    std::string spell_base(const std::string& base) const {
        if (!charset_.greek) {
            return base;
        }
        bool capital = false;
        const std::string letter = greek_letter(base, capital);
        if (letter.empty()) {
            return base;
        }
        for (const GreekGlyph& glyph : kGreek) {
            if (letter == glyph.name) {
                return capital ? glyph.upper : glyph.lower;
            }
        }
        return base;
    }

    Charset charset_;
};

} // namespace

std::unique_ptr<Syntax> make_ascii_syntax(const NotationOptions& options) {
    return std::make_unique<TextSyntax>(options, kAscii);
}

std::unique_ptr<Syntax> make_unicode_syntax(const NotationOptions& options) {
    return std::make_unique<TextSyntax>(options, kUnicode);
}

} // namespace ms::sym2::detail
