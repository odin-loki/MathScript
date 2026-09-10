// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief §11.1: the two MathML tables. Presentation says what an expression looks
///   like; Content says what it means.
///
/// They are in one file because they are one format's two halves and the choice between
/// them is the same choice each time -- a glyph or a meaning -- so the pair reads best
/// side by side. Everything structural was already decided by the walker in
/// `notation.cpp`; what is left here is which element a thing is spelled with.
///
/// Two invariants hold throughout and every method below is written to keep them:
///
///  1. **Every method returns exactly one element.** `<mfrac>`, `<msup>`, `<mroot>`,
///     `<mtd>` and `<apply>`'s operand slots are positional, so a fragment that came
///     back as two sibling elements would not merely look wrong, it would shift every
///     following child into the wrong slot -- `<mfrac>a b c</mfrac>` is not a fraction.
///     Anything with more than one part is therefore wrapped, in `<mrow>` for
///     Presentation and in `<apply>` for Content.
///  2. **Nothing that came from a symbol name reaches the output unescaped.** A name is
///     arbitrary text and `<` in it would close nothing and open something.
///
/// Greek letters and the operator signs are written as numeric character references
/// (`&#x3B1;`) rather than as literal UTF-8. Both are correct MathML, and UTF-8 is
/// shorter; the reference is chosen because this output is routinely pasted into a page
/// or a file whose encoding is declared wrong or not declared at all, and a numeric
/// reference is the one spelling that survives being read as ASCII, Latin-1 or UTF-8
/// alike. It is applied without exception, so no string emitted from here carries a byte
/// above 0x7F -- other than one a symbol's own name put there, which is the caller's
/// text and is escaped and passed through rather than reinterpreted.

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ms/core/format.hpp"

#include "notation_syntax.hpp"

namespace ms::sym2::detail {

using bignum::BigInt;

namespace {

/// The `+` or leading zeros `printf`'s `%g` puts in an exponent, removed: `"+06"` is
/// `"6"` and `"-07"` is `"-7"`. Neither MathML half has a use for the padding, and in
/// Content a leading zero would make the exponent look like an octal literal to a
/// reader that has seen one too many C files.
std::string normalised_exponent(const std::string& text) {
    std::size_t at = 0;
    std::string sign;
    if (!text.empty() && (text[0] == '+' || text[0] == '-')) {
        sign = text[0] == '-' ? "-" : "";
        at = 1;
    }
    while (at + 1 < text.size() && text[at] == '0') {
        ++at;
    }
    return sign + text.substr(at);
}

// --- Presentation MathML --------------------------------------------------------------

/// The Greek letter `base` names, as a numeric character reference, or false when it
/// names none.
///
/// The letter-to-code-point pairs follow the TeX spellings that the MathML entity tables
/// were built to match, which is why `epsilon` is the lunate U+03F5 and `varepsilon` the
/// ordinary U+03B5, and likewise `phi` U+03D5 against `varphi` U+03C6. Those two pairs
/// look backwards read as English, and they are the ones worth stating: someone who
/// wrote `varphi` and saw a `phi` on the page would be looking at a different symbol
/// from the one in their expression.
bool greek_reference(const std::string& base, std::string& out) {
    // The capital column repeats the plain letter for every `var` name, because Unicode
    // has no capital variant shapes -- there is no capital `varrho` distinct from a
    // capital `rho`. Falling back to the plain capital is the only spelling available
    // and is what every renderer would show anyway.
    static const struct {
        const char* name;
        const char* lower;
        const char* upper;
    } kLetters[] = {
        {"alpha", "&#x3B1;", "&#x391;"},      {"beta", "&#x3B2;", "&#x392;"},
        {"gamma", "&#x3B3;", "&#x393;"},      {"delta", "&#x3B4;", "&#x394;"},
        {"epsilon", "&#x3F5;", "&#x395;"},    {"zeta", "&#x3B6;", "&#x396;"},
        {"eta", "&#x3B7;", "&#x397;"},        {"theta", "&#x3B8;", "&#x398;"},
        {"iota", "&#x3B9;", "&#x399;"},       {"kappa", "&#x3BA;", "&#x39A;"},
        {"lambda", "&#x3BB;", "&#x39B;"},     {"mu", "&#x3BC;", "&#x39C;"},
        {"nu", "&#x3BD;", "&#x39D;"},         {"xi", "&#x3BE;", "&#x39E;"},
        {"omicron", "&#x3BF;", "&#x39F;"},    {"pi", "&#x3C0;", "&#x3A0;"},
        {"rho", "&#x3C1;", "&#x3A1;"},        {"sigma", "&#x3C3;", "&#x3A3;"},
        {"tau", "&#x3C4;", "&#x3A4;"},        {"upsilon", "&#x3C5;", "&#x3A5;"},
        {"phi", "&#x3D5;", "&#x3A6;"},        {"chi", "&#x3C7;", "&#x3A7;"},
        {"psi", "&#x3C8;", "&#x3A8;"},        {"omega", "&#x3C9;", "&#x3A9;"},
        {"varepsilon", "&#x3B5;", "&#x395;"}, {"vartheta", "&#x3D1;", "&#x398;"},
        {"varpi", "&#x3D6;", "&#x3A0;"},      {"varrho", "&#x3F1;", "&#x3A1;"},
        {"varsigma", "&#x3C2;", "&#x3A3;"},   {"varphi", "&#x3C6;", "&#x3A6;"},
    };
    bool capital = false;
    const std::string letter = greek_letter(base, capital);
    if (letter.empty()) {
        return false;
    }
    for (const auto& entry : kLetters) {
        if (letter == entry.name) {
            out = capital ? entry.upper : entry.lower;
            return true;
        }
    }
    return false;
}

/// One token of a name: `<mn>` when it is a numeral, `<mi>` otherwise.
///
/// The `<mi>` half needs no `mathvariant`, because MathML already draws a one-character
/// `<mi>` in italic and a longer one upright. That default is exactly the print
/// convention -- a variable `x` is sloped and a function name `sin` is not -- so the
/// right output here is the one that says nothing.
std::string identifier_element(const std::string& text) {
    if (!text.empty() && text.find_first_not_of("0123456789") == std::string::npos) {
        return "<mn>" + text + "</mn>";
    }
    std::string reference;
    if (greek_reference(text, reference)) {
        return "<mi>" + reference + "</mi>";
    }
    if (text == "hbar") {
        // Not a Greek letter, so `greek_letter` does not report it, but it is a name
        // anyone doing physics writes and U+210F is the character they mean by it.
        return "<mi>&#x210F;</mi>";
    }
    return "<mi>" + xml_escape(text) + "</mi>";
}

/// A number, given `ms::format_exact`'s spelling of it.
///
/// Two things that spelling contains have to be undone rather than copied. A leading
/// `-` is a hyphen-minus, which is a narrower character than the minus sign it has to
/// line up under a `+`, so the sign becomes an `<mo>` carrying U+2212. And `%g` reaches
/// for exponent notation surprisingly early -- `format_exact(1e6)` is the string
/// `"1e+06"` -- so a plain `<mn>1e+06</mn>` would print the literal characters `1e+06`,
/// which is source code rather than mathematics. Typesetting that exponent as a power of
/// ten is the entire job of a presentation format.
std::string presentation_number(std::string digits) {
    std::string sign;
    if (!digits.empty() && digits.front() == '-') {
        sign = "<mo>&#x2212;</mo>";
        digits.erase(0, 1);
    }
    std::string exponent;
    const std::size_t at = digits.find_first_of("eE");
    if (at != std::string::npos) {
        exponent = normalised_exponent(digits.substr(at + 1));
        digits.erase(at);
    }
    const std::string mantissa = "<mn>" + digits + "</mn>";
    if (exponent.empty()) {
        return sign.empty() ? mantissa : "<mrow>" + sign + mantissa + "</mrow>";
    }
    std::string power_of_ten = "<msup><mn>10</mn>";
    power_of_ten += exponent.front() == '-'
                        ? "<mrow><mo>&#x2212;</mo><mn>" + exponent.substr(1) + "</mn></mrow>"
                        : "<mn>" + exponent + "</mn>";
    power_of_ten += "</msup>";
    return "<mrow>" + sign + mantissa + "<mo>&#xD7;</mo>" + power_of_ten + "</mrow>";
}

class PresentationMathmlSyntax final : public Syntax {
public:
    using Syntax::Syntax;

    std::string integer(const BigInt& value) const override {
        return presentation_number(value.to_string());
    }

    /// The sign of an exact p/q goes outside the fraction bar, not into the numerator.
    /// `-1/3` written with the minus above the bar is a fraction whose numerator happens
    /// to be a negative number, which is a different thing to look at from the negation
    /// of a fraction -- and the walker has already given a Rational with a negative
    /// numerator the loose precedence that a leading minus needs, so this is also the
    /// spelling that matches the grouping it will be given.
    std::string rational(const BigInt& num, const BigInt& den) const override {
        if (!num.negative) {
            return quotient(integer(num), integer(den));
        }
        BigInt magnitude = num;
        magnitude.negative = false;
        return negate(quotient(integer(magnitude), integer(den)));
    }

    std::string real(double value) const override {
        if (!std::isfinite(value)) {
            // `format_exact` falls back to `std::to_string` here, which spells these
            // "inf" and "nan" -- C library words, not mathematics. They denote exactly
            // what the Constant head denotes, so they are spelled the same way.
            if (std::isnan(value)) {
                return constant("nan");
            }
            return constant(value < 0.0 ? "-inf" : "inf");
        }
        // The separator is applied here and only here. It is a preference about how a
        // Real atom's digits look, and `<mn>` renders its content literally, so a comma
        // in it is a comma on the page and nothing more.
        return presentation_number(
            apply_decimal_separator(ms::format_exact(value), options().decimal_separator));
    }

    std::string symbol(const std::string& base, const std::string& subscript) const override {
        const std::string letter = identifier_element(base);
        if (subscript.empty()) {
            return letter;
        }
        return "<msub>" + letter + identifier_element(subscript) + "</msub>";
    }

    std::string constant(const std::string& name) const override {
        if (name == "pi") {
            return "<mi>&#x3C0;</mi>";
        }
        if (name == "e" || name == "i") {
            // Euler's number and the imaginary unit are a sloped `e` and `i` in print,
            // which is what a one-character `<mi>` draws, so that is what goes here.
            // Unicode does have U+2147 and U+2148 for exactly these two meanings, and
            // they belong in the Content output below, where a machine is reading. The
            // consequence is that `symbol("e")` and `constant("e")` are indistinguishable
            // in Presentation MathML -- which is correct, because they are
            // indistinguishable on paper too. A rendering cannot carry that difference;
            // that is what the other table in this file is for.
            return "<mi>" + name + "</mi>";
        }
        if (name == "inf") {
            return "<mi>&#x221E;</mi>";
        }
        if (name == "-inf") {
            return "<mrow><mo>&#x2212;</mo><mi>&#x221E;</mi></mrow>";
        }
        // "nan" is spelled the way it is read. It and "undefined" are both longer than
        // one character, so `<mi>` draws them upright without being told to, which is
        // right: neither is a variable.
        if (name == "nan") {
            return "<mi>NaN</mi>";
        }
        return "<mi>" + xml_escape(name) + "</mi>";
    }

    /// The signs and the terms are siblings inside one `<mrow>` rather than nested
    /// pairs, because MathML reads an `<mo>`'s form -- prefix, infix or postfix -- off
    /// its position among its siblings, and that is what decides the space around it.
    /// A leading U+2212 in the first position is the prefix minus of `-x`; the same
    /// character after a term is the infix minus of `a - b`, set wider. Nest the terms
    /// and every operator becomes a prefix one.
    std::string sum(const std::vector<SumTerm>& terms) const override {
        std::string out = "<mrow>";
        for (std::size_t i = 0; i < terms.size(); ++i) {
            if (terms[i].negative) {
                out += "<mo>&#x2212;</mo>";
            } else if (i > 0) {
                out += "<mo>+</mo>";
            }
            out += terms[i].text;
        }
        return out + "</mrow>";
    }

    std::string product(const std::vector<std::string>& factors) const override {
        const std::string separator = multiplication_operator();
        std::string out = "<mrow>";
        for (std::size_t i = 0; i < factors.size(); ++i) {
            if (i > 0) {
                out += separator;
            }
            out += factors[i];
        }
        return out + "</mrow>";
    }

    std::string quotient(const std::string& numerator,
                         const std::string& denominator) const override {
        return "<mfrac>" + numerator + denominator + "</mfrac>";
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        return "<msup>" + base + exponent + "</msup>";
    }

    /// `<mroot>` takes the radicand first and the degree second, which is the opposite
    /// of the order it is read aloud in. Writing the degree first spells the `x`th root
    /// of three where the cube root of `x` was meant, and both are well-formed markup,
    /// so nothing but a reader will catch it.
    std::string root(const std::string& radicand, const std::string& degree) const override {
        if (degree.empty()) {
            return "<msqrt>" + radicand + "</msqrt>";
        }
        return "<mroot>" + radicand + degree + "</mroot>";
    }

    std::string negate(const std::string& operand) const override {
        return "<mrow><mo>&#x2212;</mo>" + operand + "</mrow>";
    }

    /// U+2061 FUNCTION APPLICATION between the name and the parenthesis is what makes
    /// this an application rather than a product. Without it `f(x+1)` is markup that
    /// says `f` next to a bracketed sum, which is exactly the ambiguity §11.2 lists as
    /// the reason presentation markup cannot be parsed back; a renderer also spaces the
    /// two cases differently and a screen reader says two different sentences.
    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        std::string out = "<mrow>" + identifier_element(name) + "<mo>&#x2061;</mo><mrow><mo>(</mo>";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) {
                out += "<mo>,</mo>";
            }
            out += args[i];
        }
        return out + "<mo>)</mo></mrow></mrow>";
    }

    std::string group(const std::string& inner) const override {
        return "<mrow><mo>(</mo>" + inner + "<mo>)</mo></mrow>";
    }

    /// `d` at every order rather than U+2202, because the node records which variables
    /// are differentiated by and says nothing about whether the body has any others.
    /// Writing a partial sign would assert a distinction the tree does not make.
    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        std::string bottom = "<mrow>";
        for (const std::string& var : vars) {
            bottom += "<mi>d</mi>" + var;
        }
        bottom += "</mrow>";
        std::string top = "<mi>d</mi>";
        if (vars.size() > 1) {
            top = "<msup><mi>d</mi><mn>" + std::to_string(vars.size()) + "</mn></msup>";
        }
        return "<mrow><mfrac>" + top + bottom + "</mfrac>" + body + "</mrow>";
    }

    /// One integral sign per variable, so a double integral is written with two, and
    /// each variable gets its own `d`. The walker hands the variables over in the order
    /// they are integrated in and they stay in it.
    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        std::string out = "<mrow>";
        for (std::size_t i = 0; i < vars.size(); ++i) {
            out += "<mo>&#x222B;</mo>";
        }
        out += body;
        for (const std::string& var : vars) {
            out += "<mi>d</mi>" + var;
        }
        return out + "</mrow>";
    }

    /// `lim` is an `<mo>`, not an `<mi>`. The operator dictionary has an entry for it,
    /// which is where its upright shape, the space after it and its movable limit come
    /// from; as an `<mi>` it would be an identifier named "lim" sitting next to the
    /// body, spaced and read aloud as a product with it.
    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        return "<mrow><munder><mo>lim</mo><mrow>" + var + "<mo>&#x2192;</mo>" + point +
               "</mrow></munder>" + body + "</mrow>";
    }

    /// `matrix_environment` is documented for LaTeX, but what it actually names is the
    /// pair of delimiters, and MathML has no environments -- so it is honoured here as
    /// the fences. A caller who asked for `bmatrix` and got parentheses would have had
    /// the option quietly ignored in one notation out of seven.
    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        std::string table = "<mtable>";
        for (const std::vector<std::string>& row : rows) {
            table += "<mtr>";
            for (const std::string& cell : row) {
                table += "<mtd>" + cell + "</mtd>";
            }
            table += "</mtr>";
        }
        table += "</mtable>";

        const std::string& environment = options().matrix_environment;
        if (environment == "matrix") {
            return table;
        }
        std::string open = "(";
        std::string close = ")";
        if (environment == "bmatrix") {
            open = "[";
            close = "]";
        } else if (environment == "Bmatrix") {
            open = "{";
            close = "}";
        } else if (environment == "vmatrix") {
            open = "|";
            close = "|";
        } else if (environment == "Vmatrix") {
            open = "&#x2016;";
            close = "&#x2016;";
        }
        return "<mrow><mo>" + open + "</mo>" + table + "<mo>" + close + "</mo></mrow>";
    }

    /// True, and it is what makes the layout right: an `<mrow>` around a sum inside a
    /// product is the grouping, and the walker's precedence model is exactly the right
    /// thing to drive it. Presentation MathML is the one MathML half where a
    /// parenthesis is meaningful, because here it is a glyph a reader sees.
    bool needs_grouping() const override { return true; }
    /// `<mfrac>`, `<msqrt>`/`<mroot>` and `<msup>` each take their children directly, so
    /// a grouping inside one of them would print a parenthesis that a reader would take
    /// to be part of the expression.
    bool quotient_is_fenced() const override { return true; }
    bool root_is_fenced() const override { return true; }
    bool exponent_is_fenced() const override { return true; }

private:
    std::string multiplication_operator() const {
        switch (options().multiplication) {
        case NotationOptions::Multiplication::Juxtaposition:
            // Juxtaposition still needs an operator here. U+2062 INVISIBLE TIMES prints
            // nothing, and printing nothing is the point -- but it is what tells the
            // renderer that `2x` is a product, so it gets a product's spacing, and what
            // stops a screen reader from pronouncing the two tokens as one name.
            // Concatenating the factors with no `<mo>` at all loses both.
            return "<mo>&#x2062;</mo>";
        case NotationOptions::Multiplication::Cross:
            return "<mo>&#xD7;</mo>";
        case NotationOptions::Multiplication::Dot:
            break;
        }
        // U+22C5 DOT OPERATOR, not U+00B7 MIDDLE DOT and not a full stop: it is the one
        // of the three that carries operator spacing and sits on the maths axis.
        return "<mo>&#x22C5;</mo>";
    }
};

// --- Content MathML -------------------------------------------------------------------
//
// This is the only output in §11.1 that round-trips exactly by construction, because it
// encodes the tree rather than a rendering of it -- `<apply><plus/>` is the Add node,
// not a picture of one -- which is why §11.2 names it the recommended interchange and
// the escape hatch from parsing LaTeX. Every choice below is made in favour of that
// property when it conflicts with how the result would look.

/// The Content MathML element for a function named `name`, or the empty string when
/// MathML has none for it.
///
/// The entry worth reading twice is `log`. MathML's `<log/>` is the base-ten logarithm
/// unless a `<logbase/>` qualifier says otherwise, and `log` in this tree is the natural
/// one -- `evaluate` in `expr.cpp` sends it to `std::log`. Mapping the name to the
/// same-looking element would silently change the base of every logarithm that went
/// through here, and the output would still be valid MathML that a consumer would
/// happily evaluate to the wrong number.
std::string content_function_element(const std::string& name, std::size_t arity) {
    static const struct {
        const char* name;
        const char* element;
    } kKnown[] = {
        {"sin", "sin"},           {"cos", "cos"},         {"tan", "tan"},
        {"sec", "sec"},           {"csc", "csc"},         {"cot", "cot"},
        {"sinh", "sinh"},         {"cosh", "cosh"},       {"tanh", "tanh"},
        {"sech", "sech"},         {"csch", "csch"},       {"coth", "coth"},
        {"asin", "arcsin"},       {"acos", "arccos"},     {"atan", "arctan"},
        {"arcsin", "arcsin"},     {"arccos", "arccos"},   {"arctan", "arctan"},
        {"asinh", "arcsinh"},     {"acosh", "arccosh"},   {"atanh", "arctanh"},
        {"exp", "exp"},           {"log", "ln"},          {"ln", "ln"},
        {"abs", "abs"},           {"floor", "floor"},     {"ceil", "ceiling"},
        {"ceiling", "ceiling"},   {"factorial", "factorial"},
        {"gcd", "gcd"},           {"lcm", "lcm"},         {"max", "max"},
        {"min", "min"},           {"conjugate", "conjugate"},
        {"arg", "arg"},           {"determinant", "determinant"},
        {"transpose", "transpose"},
    };
    if (name == "sqrt" && arity == 1) {
        // `<root/>` with no `<degree>` qualifier is the square root by definition, so
        // this is the exact same node the walker builds for `x^(1/2)`. Two spellings of
        // one meaning would defeat the point of a semantic format.
        return "root";
    }
    for (const auto& entry : kKnown) {
        if (name == entry.name) {
            return entry.element;
        }
    }
    return {};
}

class ContentMathmlSyntax final : public Syntax {
public:
    using Syntax::Syntax;

    std::string integer(const BigInt& value) const override {
        // A leading `-` is inside the lexical form of a `<cn>`, so a negative integer
        // stays one number. `<apply><minus/><cn>2</cn></apply>` denotes the same value
        // but says "the negation of two", and a consumer looking for a literal would
        // have to evaluate it to find one.
        return "<cn>" + value.to_string() + "</cn>";
    }

    /// An exact p/q as a single number, which is what it is.
    ///
    /// `<apply><divide/><cn>1</cn><cn>3</cn></apply>` is a correct encoding of the same
    /// value, and it is the wrong one: it says a division is to be performed, so a
    /// consumer that performs it in floating point has 0.333... and the exactness this
    /// tree exists to keep (§10.1, "a CAS that cannot represent 1/3 is not a CAS") is
    /// gone at the format boundary. `type="rational"` says the atom itself is the exact
    /// number and there is nothing to evaluate.
    std::string rational(const BigInt& num, const BigInt& den) const override {
        return "<cn type=\"rational\">" + num.to_string() + "<sep/>" + den.to_string() +
               "</cn>";
    }

    std::string real(double value) const override {
        if (!std::isfinite(value)) {
            if (std::isnan(value)) {
                return constant("nan");
            }
            return constant(value < 0.0 ? "-inf" : "inf");
        }
        // `options().decimal_separator` is deliberately not applied. It is documented as
        // a preference that may not alter the value, and here it would: a `<cn>`'s
        // content is a number in MathML's own lexical space, which is written with a
        // point, so `<cn>3,5</cn>` does not denote three and a half -- it does not
        // denote a number at all. The preference is about how digits look and this
        // output is not looked at.
        const std::string digits = ms::format_exact(value);
        const std::size_t at = digits.find_first_of("eE");
        if (at == std::string::npos) {
            return "<cn>" + digits + "</cn>";
        }
        // `%g` reaches exponent notation at 1e6 already, and `<cn>1e+06</cn>` is outside
        // the lexical space of a plain `<cn>` in the same way `3,5` is. MathML spells an
        // exponent with a type of its own and a `<sep/>` between mantissa and exponent.
        return "<cn type=\"e-notation\">" + digits.substr(0, at) + "<sep/>" +
               normalised_exponent(digits.substr(at + 1)) + "</cn>";
    }

    /// The name, put back together, and nothing else.
    ///
    /// This is the one place where Content and Presentation genuinely disagree rather
    /// than merely differing. A symbol called `alpha` is a variable whose name is the
    /// six letters `alpha`; drawing it as U+03B1 is right on a page and wrong here,
    /// because a consumer reading it back would get a variable named with a Greek
    /// character, which is a different variable. The same argument applies to the
    /// subscript: `x_1` is one identifier, and `<ci><msub>...</msub></ci>` -- which the
    /// spec does allow -- turns its name into a little tree that a reader has to flatten
    /// before it can compare two of them.
    std::string symbol(const std::string& base, const std::string& subscript) const override {
        return "<ci>" + xml_escape(subscript.empty() ? base : base + "_" + subscript) +
               "</ci>";
    }

    std::string constant(const std::string& name) const override {
        if (name == "pi") {
            return "<pi/>";
        }
        if (name == "e") {
            return "<exponentiale/>";
        }
        if (name == "i") {
            return "<imaginaryi/>";
        }
        if (name == "inf") {
            return "<infinity/>";
        }
        if (name == "-inf") {
            // MathML has one infinity and no sign on it, so the sign is an operation.
            return "<apply><minus/><infinity/></apply>";
        }
        if (name == "nan") {
            return "<notanumber/>";
        }
        // MathML has no element for "undefined", and `<notanumber/>` is not it: NaN is a
        // floating-point value that arithmetic produces and propagates, where undefined
        // is the absence of a value. A `csymbol` in a content dictionary of this
        // system's own is the honest encoding -- it names the symbol and says where its
        // definition lives, rather than borrowing an element that means something else.
        return "<csymbol cd=\"mathscript\">" + xml_escape(name) + "</csymbol>";
    }

    /// `<plus/>` over every term, with a negative one wrapped in a unary `<minus/>`.
    ///
    /// A two-term sum whose second term is negative could be the binary
    /// `<apply><minus/>a b</apply>`, which is the more idiomatic spelling. It is not
    /// used because it is a special case that fires at exactly one arity: `a - b - c`
    /// cannot take it, so the general form has to exist anyway, and having one form
    /// means a consumer has one shape to match. Both denote the same value.
    std::string sum(const std::vector<SumTerm>& terms) const override {
        std::string out = "<apply><plus/>";
        for (const SumTerm& term : terms) {
            out += term.negative ? negate(term.text) : term.text;
        }
        return out + "</apply>";
    }

    std::string product(const std::vector<std::string>& factors) const override {
        std::string out = "<apply><times/>";
        for (const std::string& factor : factors) {
            out += factor;
        }
        return out + "</apply>";
    }

    std::string quotient(const std::string& numerator,
                         const std::string& denominator) const override {
        return "<apply><divide/>" + numerator + denominator + "</apply>";
    }

    std::string power(const std::string& base, const std::string& exponent) const override {
        return "<apply><power/>" + base + exponent + "</apply>";
    }

    /// The `<degree>` qualifier comes before the radicand, which is the order MathML
    /// puts qualifiers in and the reverse of how `<mroot>` above wants its two children.
    /// A square root omits it entirely rather than saying `<degree><cn>2</cn></degree>`,
    /// because two is what `<root/>` already means.
    std::string root(const std::string& radicand, const std::string& degree) const override {
        if (degree.empty()) {
            return "<apply><root/>" + radicand + "</apply>";
        }
        return "<apply><root/><degree>" + degree + "</degree>" + radicand + "</apply>";
    }

    std::string negate(const std::string& operand) const override {
        return "<apply><minus/>" + operand + "</apply>";
    }

    std::string call(const std::string& name,
                     const std::vector<std::string>& args) const override {
        const std::string element = content_function_element(name, args.size());
        std::string out = "<apply>";
        if (element.empty()) {
            // An unknown name is a `csymbol` rather than a `ci`. `<ci>` means an
            // identifier -- a variable -- and applying one says "whatever this variable
            // holds, applied to these arguments", which is a different claim from "the
            // function that this name denotes". `cd` names where the definition lives,
            // and for a function this tree carries but MathML does not know, that is
            // this system.
            out += "<csymbol cd=\"mathscript\">" + xml_escape(name) + "</csymbol>";
        } else {
            out += "<" + element + "/>";
        }
        for (const std::string& arg : args) {
            out += arg;
        }
        return out + "</apply>";
    }

    /// Unreachable: `needs_grouping()` is false, so the walker never asks for a
    /// grouping. It returns `inner` unchanged rather than the empty string so that if it
    /// ever were reached -- a new call site in the walker, say -- the result would be
    /// markup that still denotes the right expression, instead of a subexpression
    /// silently deleted from the output.
    std::string group(const std::string& inner) const override { return inner; }

    /// `<diff/>` for one variable and `<partialdiff/>` for several, which is the
    /// distinction MathML draws. Repeated variables stay repeated `<bvar>`s rather than
    /// being collapsed into a `<bvar>` with a `<degree>`: collapsing means deciding that
    /// two rendered variables are the same variable, and that is a structural decision
    /// the walker did not make and this table is in no position to.
    std::string derivative(const std::string& body,
                           const std::vector<std::string>& vars) const override {
        std::string out = vars.size() > 1 ? "<apply><partialdiff/>" : "<apply><diff/>";
        for (const std::string& var : vars) {
            out += "<bvar>" + var + "</bvar>";
        }
        return out + body + "</apply>";
    }

    std::string integral(const std::string& body,
                         const std::vector<std::string>& vars) const override {
        std::string out = "<apply><int/>";
        for (const std::string& var : vars) {
            out += "<bvar>" + var + "</bvar>";
        }
        return out + body + "</apply>";
    }

    /// `<lowlimit>` rather than a `<condition>` holding a `<tendsto/>`. Both are in the
    /// spec; the condition form exists to say *how* the variable approaches the point --
    /// from above, from below, along some path -- and this tree records no direction, so
    /// writing one would invent it.
    std::string limit(const std::string& body, const std::string& var,
                      const std::string& point) const override {
        return "<apply><limit/><bvar>" + var + "</bvar><lowlimit>" + point + "</lowlimit>" +
               body + "</apply>";
    }

    std::string matrix(const std::vector<std::vector<std::string>>& rows) const override {
        std::string out = "<matrix>";
        for (const std::vector<std::string>& row : rows) {
            out += "<matrixrow>";
            for (const std::string& cell : row) {
                out += cell;
            }
            out += "</matrixrow>";
        }
        return out + "</matrix>";
    }

    /// False. `<apply>` is an operator followed by its operands in order, so no
    /// arrangement of them is ambiguous and there is nothing for a grouping to
    /// disambiguate. It is not merely unnecessary: Content MathML has no element that
    /// means "these were bracketed", so a parenthesis would have to be written as a
    /// character, and a character in a semantic tree is a value rather than a delimiter.
    bool needs_grouping() const override { return false; }
    /// True for the same reason the others are, and it has to be said out loud: the
    /// walker calls `group()` directly -- not through `place()`, which checks
    /// `needs_grouping()` -- when a multi-factor numerator or denominator goes into an
    /// unfenced quotient. Saying false here would reach that call.
    bool quotient_is_fenced() const override { return true; }
    bool root_is_fenced() const override { return true; }
    bool exponent_is_fenced() const override { return true; }
};

} // namespace

std::unique_ptr<Syntax> make_presentation_mathml_syntax(const NotationOptions& options) {
    return std::make_unique<PresentationMathmlSyntax>(options);
}

std::unique_ptr<Syntax> make_content_mathml_syntax(const NotationOptions& options) {
    return std::make_unique<ContentMathmlSyntax>(options);
}

} // namespace ms::sym2::detail
