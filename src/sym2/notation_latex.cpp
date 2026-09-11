// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief The LaTeX spellings for the §11.1 walker.
///
/// Nothing here decides structure. `notation.cpp` has already worked out that a factor
/// belongs in a denominator, that a power is really a root, that a sum term is a
/// subtraction and where a grouping is needed; this file only says what those come out
/// looking like in LaTeX. Every method receives its pieces already rendered.
///
/// Two things make LaTeX harder to spell correctly than its output suggests, and both
/// are places where the obvious string is wrong rather than merely ugly:
///
///   - **Math mode discards source whitespace.** `2 x` and `2x` typeset identically,
///     which is why juxtaposition is safe for a coefficient -- and why
///     `\mathrm{foo} \mathrm{bar}` is not, since what reaches the page is the single
///     word "foobar". Where the page would run two factors together, `product()` puts
///     a thin space in.
///   - **Every character is potentially a command.** A symbol name is data: it can come
///     from a host program or a saved session. An unescaped `%` in one comments out the
///     rest of the line it lands in, and an unescaped `\` starts a control word. So a
///     name is escaped before it is ever emitted.
///
/// The `amsmath` dependency named in `notation.hpp` is real but small: `pmatrix` and
/// its relatives, and `\operatorname` for a function this table has no name for.

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

/// LaTeX's ten special characters, spelled so that they typeset as themselves instead
/// of being obeyed. A symbol or function name is data rather than markup, so this is
/// what stands between a variable called `a%b` and the rest of the document line
/// disappearing into a comment.
std::string escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        switch (c) {
        case '#':
        case '$':
        case '%':
        case '&':
        case '_':
        case '{':
        case '}':
            out.push_back('\\');
            out.push_back(c);
            break;
        // These three have no backslash-plus-character spelling: `\~` and `\^` are
        // accents and would consume whatever follows them, and `\\` is a line break.
        // The empty group after each keeps the control word from swallowing a space.
        case '~': out += "\\textasciitilde{}"; break;
        case '^': out += "\\textasciicircum{}"; break;
        case '\\': out += "\\textbackslash{}"; break;
        default: out.push_back(c); break;
        }
    }
    return out;
}

struct GreekName {
    const char* lower;   ///< the name `greek_letter` reports, always in lower case
    const char* capital; ///< how the capitalised spelling of it is written
};

/// The Greek letters, with their capitals.
///
/// Thirteen of the twenty-four Greek capitals are, as glyphs, the Latin letters they
/// gave rise to, so LaTeX gives a control sequence only to the eleven that are not:
/// there is no `\Alpha`, because it could only ever set an `A`. A capitalised name from
/// the other thirteen is therefore written as the Latin letter it is -- and as an
/// *upright* one, because a bare `A` in math mode is italic, and italic is precisely how
/// this table writes a variable named `A`. `\Gamma` is upright too, so this also keeps
/// the two halves of the alphabet looking like each other.
///
/// The `var` spellings have no capital of their own: `\varepsilon` and `\epsilon` are
/// two shapes of the same letter and share the one capital.
constexpr GreekName kGreek[] = {
    {"alpha", "\\mathrm{A}"},   {"beta", "\\mathrm{B}"},    {"gamma", "\\Gamma"},
    {"delta", "\\Delta"},       {"epsilon", "\\mathrm{E}"}, {"zeta", "\\mathrm{Z}"},
    {"eta", "\\mathrm{H}"},     {"theta", "\\Theta"},       {"iota", "\\mathrm{I}"},
    {"kappa", "\\mathrm{K}"},   {"lambda", "\\Lambda"},     {"mu", "\\mathrm{M}"},
    {"nu", "\\mathrm{N}"},      {"xi", "\\Xi"},             {"omicron", "\\mathrm{O}"},
    {"pi", "\\Pi"},             {"rho", "\\mathrm{P}"},     {"sigma", "\\Sigma"},
    {"tau", "\\mathrm{T}"},     {"upsilon", "\\Upsilon"},   {"phi", "\\Phi"},
    {"chi", "\\mathrm{X}"},     {"psi", "\\Psi"},           {"omega", "\\Omega"},
    {"varepsilon", "\\mathrm{E}"}, {"vartheta", "\\Theta"}, {"varpi", "\\Pi"},
    {"varrho", "\\mathrm{P}"},  {"varsigma", "\\Sigma"},    {"varphi", "\\Phi"},
};

/// Names that are not Greek but that mathematics still writes with a glyph of their
/// own. A user who calls a variable `hbar` means the constant, and `\mathrm{hbar}` --
/// which is what the fallback below would produce -- is not it.
constexpr const char* kNamedGlyphs[] = {"hbar", "ell", "infty", "nabla", "partial", "aleph"};

bool all_digits(const std::string& text) {
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

/// How a bare identifier is written in math mode.
///
/// The fallback for a name of more than one character is `\mathrm{}` rather than the
/// name itself, because math mode sets each letter as its own variable: `foo` typesets
/// as *f* *o* *o*, a product of three things, with the italic correction between them
/// that a product gets. That is a different expression, not a different font.
std::string spell_name(const std::string& name) {
    if (name.empty()) {
        return {};
    }
    bool capital = false;
    const std::string greek = greek_letter(name, capital);
    if (!greek.empty()) {
        for (const GreekName& entry : kGreek) {
            if (greek == entry.lower) {
                return capital ? std::string(entry.capital) : "\\" + greek;
            }
        }
    }
    for (const char* glyph : kNamedGlyphs) {
        if (name == glyph) {
            return "\\" + name;
        }
    }
    // Digits are already upright, and `\mathrm{12}` would only add a level of braces to
    // a subscript that reads perfectly well as `x_{12}`.
    if (all_digits(name)) {
        return name;
    }
    if (name.size() == 1 && std::isspace(static_cast<unsigned char>(name.front())) == 0) {
        return escape(name);  // §4.1: whitespace set bare is empty input, not a name
    }
    return "\\mathrm{" + escape(name) + "}";
}

/// True when `text` is one upright multi-letter name and nothing else. Such a fragment
/// is the one that cannot be juxtaposed, because on the page it is a *word*: put two of
/// them side by side and the reader sees a single longer word.
bool is_upright_word(const std::string& text) {
    std::size_t open = 0;
    if (text.rfind("\\mathrm{", 0) == 0) {
        open = sizeof("\\mathrm{") - 2;
    } else if (text.rfind("\\operatorname{", 0) == 0) {
        open = sizeof("\\operatorname{") - 2;
    } else {
        return false;
    }
    int depth = 0;
    for (std::size_t i = open; i < text.size(); ++i) {
        // An escaped brace from `escape` is a character in the name, not a group.
        if (text[i] == '\\') {
            ++i;
            continue;
        }
        if (text[i] == '{') {
            ++depth;
        } else if (text[i] == '}' && --depth == 0) {
            return i + 1 == text.size();
        }
    }
    return false;
}

/// Whether two juxtaposed factors would be read as one thing.
///
/// Juxtaposition is the notation's own spelling of multiplication and it is what a
/// mathematician writes, but it only works when neither side can be misread joined to
/// the other. `2x` is safe, and so is `xy`, because a reader takes each italic letter
/// as its own variable. `\mathrm{foo}\mathrm{bar}` is not: math mode drops the source
/// space and the page carries the word "foobar". Two numbers are worse still -- `2` and
/// `3` juxtaposed are not ambiguous, they are twenty-three. A thin space is the
/// smallest mark that keeps such a pair two factors.
bool needs_thin_space(const std::string& left, const std::string& right) {
    if (is_upright_word(left) || is_upright_word(right)) {
        return true;
    }
    if (left.empty() || right.empty()) {
        return false;
    }
    return std::isdigit(static_cast<unsigned char>(left.back())) != 0 &&
           std::isdigit(static_cast<unsigned char>(right.front())) != 0;
}

/// The LaTeX log-like operators, which set their name upright and carry the spacing an
/// operator gets. This is a convenience rather than a requirement: `\operatorname{}`
/// below produces the same typesetting for any name not listed, so a missing entry
/// costs nothing but length.
constexpr const char* kOperators[] = {
    "sin",  "cos",  "tan",    "log",    "ln",     "exp", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "sec", "csc",   "cot", "min",  "max",  "gcd",
    "det",  "arg",  "deg",    "dim",    "ker",    "lg",  "sup",
};

} // namespace

namespace {

class LatexSyntax final : public Syntax {
public:
    explicit LatexSyntax(const NotationOptions& options) : Syntax(options) {}

    // --- Atoms ----------------------------------------------------------------------

    std::string integer(const BigInt& value) const override { return value.to_string(); }

    std::string rational(const BigInt& num, const BigInt& den) const override {
        // `\frac{-3}{4}` is correct and nobody writes it. The walker agrees: it gives a
        // negative Rational add-level precedence in a notation whose quotient fences
        // itself, which is the precedence of a leading minus sign rather than of a
        // fraction, so it is already expecting the sign outside.
        if (num.negative) {
            BigInt magnitude = num;
            magnitude.negative = false;
            return "-" + quotient(integer(magnitude), integer(den));
        }
        return quotient(integer(num), integer(den));
    }

    std::string real(double value) const override {
        if (!std::isfinite(value)) {
            // `format_exact` hands infinities and NaN to `std::to_string`, which gives
            // `inf` and `nan`: three italic letters that math mode sets as a product of
            // three variables, one of which is the constant `e` in this very notation.
            // The constant spellings are the ones that denote the value.
            if (std::isnan(value)) {
                return constant("nan");
            }
            return value < 0.0 ? constant("-inf") : constant("inf");
        }
        // A Real is an expression, not a display, so it may not round: `format_exact`
        // emits the fewest digits that read back as this same double, which is a
        // stronger promise than the REPL's scalar formatting makes.
        const std::string text = ms::format_exact(value);
        const std::size_t at = text.find('e');
        if (at == std::string::npos || at + 1 >= text.size()) {
            return decimal(text);
        }
        // `1e+20` is not LaTeX for a number: math mode sets it as the product 1*e plus
        // 20, and `e` is Euler's number here. Scientific notation has to be written out
        // as the power of ten it stands for.
        std::string exponent = text.substr(at + 1);
        std::size_t i = 0;
        bool negative = false;
        if (exponent[i] == '+' || exponent[i] == '-') {
            negative = exponent[i] == '-';
            ++i;
        }
        while (i + 1 < exponent.size() && exponent[i] == '0') {
            ++i;
        }
        // `\times` regardless of the multiplication option: this factor of ten is part
        // of the numeral, and juxtaposing it would give `1 10^{20}`, which reads as one
        // number followed by another.
        return decimal(text.substr(0, at)) + " \\times 10^{" + (negative ? "-" : "") +
               exponent.substr(i) + "}";
    }

    std::string symbol(const std::string& base, const std::string& subscript) const override {
        std::string out = spell_name(base);
        if (!subscript.empty()) {
            // The braces are not optional: `x_12` is x subscript 1 followed by a 2, so a
            // subscript of more than one character silently becomes a different symbol
            // without them. The subscript goes through the same name spelling, which is
            // what makes `x_max` come out `x_{\mathrm{max}}` rather than as a product of
            // three subscripted letters.
            out += "_{" + spell_name(subscript) + "}";
        }
        return out;
    }

    std::string constant(const std::string& name) const override {
        if (name == "pi") {
            return "\\pi";
        }
        // `e` and `i` stay italic single letters, which is what the surrounding
        // notation makes of every other one-letter name; the upright `\mathrm{e}` of
        // ISO style would make them look like the start of a multi-letter word.
        if (name == "e" || name == "i") {
            return name;
        }
        if (name == "inf") {
            return "\\infty";
        }
        if (name == "-inf") {
            return "-\\infty";
        }
        if (name == "nan") {
            return "\\mathrm{NaN}";
        }
        if (name == "undefined") {
            return "\\mathrm{undefined}";
        }
        return "\\mathrm{" + escape(name) + "}";
    }

    // --- Structure ------------------------------------------------------------------

    std::string sum(const std::vector<SumTerm>& terms) const override {
        std::string out;
        bool first = true;
        for (const SumTerm& term : terms) {
            if (first) {
                if (term.negative) {
                    out += "-";
                }
                first = false;
            } else {
                out += term.negative ? " - " : " + ";
            }
            out += term.text;
        }
        return out;
    }

    std::string product(const std::vector<std::string>& factors) const override {
        std::string out;
        for (const std::string& factor : factors) {
            if (!out.empty()) {
                switch (options().multiplication) {
                case NotationOptions::Multiplication::Dot: out += " \\cdot "; break;
                case NotationOptions::Multiplication::Cross: out += " \\times "; break;
                case NotationOptions::Multiplication::Juxtaposition:
                    if (needs_thin_space(out, factor)) {
                        out += "\\,";
                    }
                    break;
                }
            }
            out += factor;
        }
        return out;
    }

    std::string quotient(const std::string& numerator,
                         const std::string& denominator) const override {
        return frac() + "{" + numerator + "}{" + denominator + "}";
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        // The braces carry the whole exponent. Without them `x^{10}` would be `x^10`,
        // which is x to the first power multiplied by ten -- a wrong value rather than
        // an ugly one, and one that still compiles.
        return base + "^{" + exponent + "}";
    }

    std::string root(const std::string& radicand, const std::string& degree) const override {
        if (degree.empty()) {
            return "\\sqrt{" + radicand + "}";
        }
        return "\\sqrt[" + degree + "]{" + radicand + "}";
    }

    std::string negate(const std::string& operand) const override { return "-" + operand; }

    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        if (name == "abs" && args.size() == 1) {
            // The bars are always the growing kind, whatever `sized_delimiters` says:
            // a plain `|` around a fraction stays one line high and the fraction hangs
            // out of it, which is a typesetting error rather than a preference.
            return "\\left| " + args[0] + " \\right|";
        }
        if (name == "sqrt" && args.size() == 1) {
            return "\\sqrt{" + args[0] + "}";
        }
        std::string out = operator_name(name);
        std::string arguments;
        for (const std::string& arg : args) {
            if (!arguments.empty()) {
                // The argument separator is a comma whatever `decimal_separator` is.
                // `f(1{,}5, 2{,}5)` is readable; `f(1,5, 2,5)` is not, in any notation.
                arguments += ", ";
            }
            arguments += arg;
        }
        // The walker places an argument in an open slot and leaves the fencing to the
        // table, so the parentheses are this method's to write.
        return out + group(arguments);
    }

    std::string group(const std::string& inner) const override {
        if (options().sized_delimiters) {
            return "\\left(" + inner + "\\right)";
        }
        return "(" + inner + ")";
    }

    // --- Heads that stand for something unevaluated ----------------------------------

    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        // `\frac{d}{dx}`, not `\frac{\partial}{\partial x}`: `Head::Derivative` carries
        // no distinction between a partial and a total derivative, so the partial
        // spelling would assert something the tree never said. Several variables chain
        // rather than being collected into `\frac{d^{2}}{dx\,dy}`, because collecting
        // them is a structural claim about repeated variables and structure is the
        // walker's to decide.
        std::string out;
        for (const std::string& var : vars) {
            out += frac() + "{d}{d" + var + "}";
        }
        if (out.empty()) {
            return body;
        }
        return out + " " + body;
    }

    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        // One integral sign per variable, and one for a malformed node that names none,
        // so that the sign is never dropped and the body left standing on its own.
        const std::size_t signs = vars.empty() ? std::size_t{1} : vars.size();
        std::string out;
        for (std::size_t i = 0; i < signs; ++i) {
            out += "\\int ";
        }
        out += body;
        for (const std::string& var : vars) {
            // The thin space is what separates the integrand from the differential.
            // Without it `\int f dx` sets `d` hard against `f`, and `f d x` reads as
            // three factors.
            out += " \\, d" + var;
        }
        return out;
    }

    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        return "\\lim_{" + var + " \\to " + point + "} " + body;
    }

    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        // An empty environment name would emit `\begin{}`, which does not compile, so an
        // unset option falls back to the same default the option itself carries.
        const std::string environment =
            options().matrix_environment.empty() ? std::string("pmatrix")
                                                 : options().matrix_environment;
        std::string body;
        for (const std::vector<std::string>& row : rows) {
            if (!body.empty()) {
                body += " \\\\ ";
            }
            bool first = true;
            for (const std::string& cell : row) {
                if (!first) {
                    body += " & ";
                }
                first = false;
                body += cell;
            }
        }
        if (body.empty()) {
            return "\\begin{" + environment + "}\\end{" + environment + "}";
        }
        return "\\begin{" + environment + "} " + body + " \\end{" + environment + "}";
    }

    // --- What the walker must ask rather than assume ----------------------------------

    /// A grouping in LaTeX is a pair of real parentheses, so the walker's precedence
    /// rule applies here exactly as it would in ASCII.
    bool needs_grouping() const override { return true; }

    /// `\frac{a+b}{c+d}` needs no parentheses on either side: the braces are the
    /// delimiter, and the rule bar is what the reader sees.
    bool quotient_is_fenced() const override { return true; }

    /// `\sqrt{a+b}` likewise -- the vinculum runs the width of the radicand.
    bool root_is_fenced() const override { return true; }

    /// `x^{a+b}` too. This is the one that stops the walker adding parentheses to every
    /// exponent that is not an atom.
    bool exponent_is_fenced() const override { return true; }

    /// A value that needs scientific notation is spelled as a product -- `1 \times
    /// 10^{20}` -- so it is not an atom, whatever the digits alone would suggest.
    /// Reported as an atom it would go straight into a superscript slot and produce
    /// `1 \times 10^{20}^{2}`, a double superscript, which is a LaTeX error rather
    /// than a formula set wrongly. Everything else here is a single numeral.
    bool real_is_atomic(double value) const override {
        if (!std::isfinite(value)) {
            return true; // \infty and \mathrm{NaN} are single tokens
        }
        const std::string text = ms::format_exact(value);
        return text.find('e') == std::string::npos;
    }

private:
    /// Display style makes a fraction full size, with the numerator above the
    /// denominator rather than beside it, which is what `display` asks for.
    std::string frac() const { return options().display ? "\\dfrac" : "\\frac"; }

    /// A decimal separator, applied to a numeral's digits only.
    std::string decimal(const std::string& digits) const {
        const char separator = options().decimal_separator;
        std::string out = apply_decimal_separator(digits, separator);
        if (separator == '.') {
            return out;
        }
        // A comma in math mode is punctuation, and punctuation is set with a space
        // after it, so `1,5` reaches the page as `1, 5` -- two numbers in a list rather
        // than one number. Bracing the separator makes it an ordinary symbol again.
        const std::size_t at = out.find(separator);
        if (at == std::string::npos) {
            return out;
        }
        return out.substr(0, at) + "{" + separator + "}" + out.substr(at + 1);
    }

    std::string operator_name(const std::string& name) const {
        for (const char* known : kOperators) {
            if (name == known) {
                return "\\" + name;
            }
        }
        // `\operatorname` is what gives an unlisted name the upright shape and the
        // operator spacing; writing the bare name instead would set it as a product of
        // its own letters.
        return "\\operatorname{" + escape(name) + "}";
    }
};

} // namespace

std::unique_ptr<Syntax> make_latex_syntax(const NotationOptions& options) {
    return std::make_unique<LatexSyntax>(options);
}

} // namespace ms::sym2::detail
