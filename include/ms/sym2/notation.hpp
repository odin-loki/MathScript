// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

/// @file
/// @brief §11.1: one tree walk, seven notations.
///
/// The plan is explicit that these are not seven printers: "Build one visitor
/// interface and five tables -- not five printers." The reason is that almost
/// everything a printer does is *structural* rather than lexical, and structure is the
/// same in every notation:
///
///   - which factors of a product are really a denominator (`x*y^-1` is `x/y`),
///   - which terms of a sum are really subtractions (`a + -1*b` is `a - b`),
///   - which powers are really roots (`x^(1/2)` is a square root, `x^(2/3)` is not
///     obviously either and the caller decides),
///   - what display order a human expects, which is not the canonical order the node
///     is stored in,
///   - where a grouping is needed, which follows from precedence rather than from
///     spelling,
///   - how a symbol name decomposes into a base and a subscript, and whether the base
///     names a letter the notation spells specially (`alpha`, `hbar`, `infty`).
///
/// Get one of those wrong in a printer and it prints a different expression. Get it
/// wrong in seven printers and it prints seven different expressions, six of which
/// nobody will ever look at closely enough to notice. So the decisions live in the
/// walker, once, and a notation is a `Syntax` table that only says how to spell what
/// the walker has already decided.
///
/// This is what §10.3's precedence-aware `to_string` unblocked. The old engine
/// parenthesised every node and printed every number with `std::to_string`, so a LaTeX
/// printer built on it would have inherited `\left(\left(2.000000 \cdot x\right) +
/// 1.000000\right)` -- both faults, in a notation where they are harder to see.

#include <cstdint>
#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/core/matrix.hpp"
#include "ms/sym2/expr.hpp"

namespace ms::sym2 {

/// The notations `to_notation` can emit.
enum class Notation : std::uint8_t {
    Latex,               ///< `\frac{x^{2}}{y}`. For documents and for §12's renderer.
    PresentationMathml,  ///< `<mfrac>...</mfrac>`. Layout, for a browser.
    ContentMathml,       ///< `<apply><divide/>...</apply>`. Semantics, for machines.
    Unicode,             ///< `x²·√y`. For a terminal that has more than ASCII.
    Ascii,               ///< `x^2*sqrt(y)`. `to_string`, reached through this API.
    SymPy,               ///< `x**2*sqrt(y)`, pasteable into a Python session.
    Mathematica,         ///< `x^2*Sqrt[y]`, pasteable into a notebook.
    C,                   ///< `pow(x, 2)*sqrt(y)`, compilable.
    Cpp,                 ///< `std::pow(x, 2)*std::sqrt(y)`.
    Python,              ///< `x**2*math.sqrt(y)`.
};

/// Choices that change the output without changing what it denotes. Every one of these
/// is a preference; none of them may alter the value.
struct NotationOptions {
    /// How a product is spelled between two factors that need a visible operator.
    enum class Multiplication : std::uint8_t {
        Juxtaposition, ///< `2x`. LaTeX and Unicode only, where it is unambiguous.
        Dot,           ///< `\cdot`, `·`, `*`.
        Cross,         ///< `\times`, `×`.
    };

    /// LaTeX: `\[ ... \]` and display-style fractions, rather than inline.
    bool display = false;
    /// LaTeX: `\left( ... \right)` rather than bare `(`, so delimiters grow with their
    /// content. Off by default because it doubles the length of every grouping and the
    /// growth only matters around a tall subexpression.
    bool sized_delimiters = false;
    Multiplication multiplication = Multiplication::Dot;
    /// LaTeX/Unicode: the decimal point. A comma is the separator in most of Europe.
    /// It is *only* applied to a Real atom's digits, never to an argument separator,
    /// because `f(1,5, 2,5)` cannot be read back by anything.
    char decimal_separator = '.';
    /// LaTeX: the environment a matrix goes in -- `pmatrix`, `bmatrix`, `vmatrix`,
    /// `Vmatrix`, `matrix`.
    std::string matrix_environment = "pmatrix";
    /// `x^(1/2)` as `\sqrt{x}`. `x^(1/3)` as `\sqrt[3]{x}`. Off gives `x^{1/2}`, which
    /// is the same expression and is what a caller feeding a parser wants.
    bool roots_as_radicals = true;
    /// Source emission: the name every free symbol is read from, e.g. `"v"` gives
    /// `v[0]`... Empty means emit the symbol's own name as an identifier.
    std::string source_variable_array;
};

// --- The seven notations ------------------------------------------------------------
//
// Each is a one-line call into the shared walker with a different table. They are named
// separately because that is how they are used, and because a name is where the
// documentation for a format's quirks belongs.

/// LaTeX, requiring only `amsmath` (for `pmatrix` and friends).
std::string to_latex(const ExprRef& e, const NotationOptions& options = {});

/// Presentation MathML: what it looks like. `<mrow>`, `<mfrac>`, `<msup>`. No
/// namespace declaration and no `<math>` wrapper -- see `mathml_document`.
std::string to_presentation_mathml(const ExprRef& e, const NotationOptions& options = {});

/// Content MathML: what it means. `<apply><plus/>...</apply>`. This is the only output
/// here that round-trips exactly by construction, because it encodes the tree rather
/// than a rendering of it, which is why §11.2 names it the recommended interchange.
std::string to_content_mathml(const ExprRef& e, const NotationOptions& options = {});

/// Plain text with the Unicode a terminal is likely to have: superscript digits, `√`,
/// `·`, `π`, and the Greek letters. Nothing outside the BMP, and nothing that needs a
/// font a terminal will not have.
std::string to_unicode(const ExprRef& e, const NotationOptions& options = {});

/// A Python expression using SymPy's spellings: `**`, `sqrt`, `Rational(1, 3)`,
/// `pi`, `oo`, `Derivative`, `Integral`, `Limit`. Exact atoms stay exact -- emitting
/// `1/3` would be a float in the reader's session and the expression would no longer
/// be the one that was printed.
std::string to_sympy(const ExprRef& e, const NotationOptions& options = {});

/// A Wolfram Language expression: `Sqrt[x]`, `Sin[x]`, `Pi`, `Infinity`, `D[...]`.
/// Square brackets are function application there; parentheses are grouping only.
std::string to_mathematica(const ExprRef& e, const NotationOptions& options = {});

/// The three languages `to_source` emits. Separate from `Notation` on purpose: with a
/// `Notation` parameter, `to_source(e, Notation::Latex)` is a call anyone can write and
/// nothing can answer honestly -- it would have to invent a language or silently pick
/// one. A type that cannot name a non-language makes the question go away.
enum class SourceLanguage : std::uint8_t { C, Cpp, Python };

/// A compilable expression in `language`, over `double`. Every rational becomes a
/// floating-point quotient, because `1/3` is zero in C and the printed expression
/// would then denote a different number.
std::string to_source(const ExprRef& e, SourceLanguage language,
                      const NotationOptions& options = {});

/// Dispatch by enum, for a caller that has the notation as data -- the REPL's
/// `sym_export("x^2", "latex")`, or a GUI menu.
std::string to_notation(const ExprRef& e, Notation notation, const NotationOptions& options = {});

/// The notation named by `text` (case-insensitive: "latex", "mathml",
/// "content-mathml", "unicode", "ascii", "sympy", "mathematica", "c", "c++", "python").
/// Reports rather than defaulting, so a typo does not silently produce the wrong
/// format.
Result<Notation> notation_from_name(const std::string& text);
/// The canonical name of `notation`, which `notation_from_name` accepts.
std::string notation_name(Notation notation);

// --- Matrices -----------------------------------------------------------------------
//
// `ms::sym2` has no matrix head -- a matrix in this tree is `ms::Matrix<double>`, which
// holds numbers rather than expressions. The plan asks for matrix output in §11.1
// anyway, and it is the same tree walk with each entry as a leaf, so it is here.

/// A matrix in `options.matrix_environment`. Entries are printed exactly, by the same
/// rule `Real` atoms use: what comes out has to denote the matrix that went in.
std::string to_latex(const Matrix<double>& m, const NotationOptions& options = {});
std::string to_notation(const Matrix<double>& m, Notation notation,
                        const NotationOptions& options = {});

/// A standalone document around `body`, for a caller writing a `.tex` file.
std::string latex_document(const std::string& body);
/// A `<math>` element with the MathML namespace around `body`.
std::string mathml_document(const std::string& body, bool display = false);

} // namespace ms::sym2
