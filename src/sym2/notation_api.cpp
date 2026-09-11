// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief The public entry points of §11.1. Each is the same call into the same walk
///   with a different table, which is what "one visitor and five tables" comes to in
///   the end.

#include "ms/sym2/notation.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "notation_syntax.hpp"

namespace ms::sym2 {

namespace {

std::unique_ptr<detail::Syntax> table_for(Notation notation, const NotationOptions& options) {
    switch (notation) {
    case Notation::Latex:
        return detail::make_latex_syntax(options);
    case Notation::PresentationMathml:
        return detail::make_presentation_mathml_syntax(options);
    case Notation::ContentMathml:
        return detail::make_content_mathml_syntax(options);
    case Notation::Unicode:
        return detail::make_unicode_syntax(options);
    case Notation::Ascii:
        return detail::make_ascii_syntax(options);
    case Notation::SymPy:
        return detail::make_sympy_syntax(options);
    case Notation::Mathematica:
        return detail::make_mathematica_syntax(options);
    case Notation::C:
    case Notation::Cpp:
    case Notation::Python:
        return detail::make_source_syntax(notation, options);
    }
    return detail::make_ascii_syntax(options);
}

Notation notation_for(SourceLanguage language) {
    switch (language) {
    case SourceLanguage::C: return Notation::C;
    case SourceLanguage::Cpp: return Notation::Cpp;
    case SourceLanguage::Python: return Notation::Python;
    }
    return Notation::Cpp;
}

std::string lowered(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

} // namespace

std::string to_latex(const ExprRef& e, const NotationOptions& options) {
    return detail::render(e, *detail::make_latex_syntax(options));
}

std::string to_presentation_mathml(const ExprRef& e, const NotationOptions& options) {
    return detail::render(e, *detail::make_presentation_mathml_syntax(options));
}

std::string to_content_mathml(const ExprRef& e, const NotationOptions& options) {
    return detail::render(e, *detail::make_content_mathml_syntax(options));
}

std::string to_unicode(const ExprRef& e, const NotationOptions& options) {
    return detail::render(e, *detail::make_unicode_syntax(options));
}

std::string to_sympy(const ExprRef& e, const NotationOptions& options) {
    return detail::render(e, *detail::make_sympy_syntax(options));
}

std::string to_mathematica(const ExprRef& e, const NotationOptions& options) {
    return detail::render(e, *detail::make_mathematica_syntax(options));
}

std::string to_source(const ExprRef& e, SourceLanguage language,
                      const NotationOptions& options) {
    return detail::render(e, *detail::make_source_syntax(notation_for(language), options));
}

std::string to_notation(const ExprRef& e, Notation notation, const NotationOptions& options) {
    return detail::render(e, *table_for(notation, options));
}

std::string to_latex(const Matrix<double>& m, const NotationOptions& options) {
    return detail::render_matrix(m, *detail::make_latex_syntax(options));
}

std::string to_notation(const Matrix<double>& m, Notation notation,
                        const NotationOptions& options) {
    return detail::render_matrix(m, *table_for(notation, options));
}

std::string notation_name(Notation notation) {
    switch (notation) {
    case Notation::Latex: return "latex";
    case Notation::PresentationMathml: return "mathml";
    case Notation::ContentMathml: return "content-mathml";
    case Notation::Unicode: return "unicode";
    case Notation::Ascii: return "ascii";
    case Notation::SymPy: return "sympy";
    case Notation::Mathematica: return "mathematica";
    case Notation::C: return "c";
    case Notation::Cpp: return "c++";
    case Notation::Python: return "python";
    }
    return "ascii";
}

Result<Notation> notation_from_name(const std::string& text) {
    // Aliases are the spellings people actually type. A name that is not here is
    // reported rather than falling back to a default, because a caller that asked for
    // "latex2" and silently received ASCII has no way to find out.
    static const struct {
        const char* name;
        Notation notation;
    } kNames[] = {
        {"latex", Notation::Latex},
        {"tex", Notation::Latex},
        {"mathml", Notation::PresentationMathml},
        {"presentation-mathml", Notation::PresentationMathml},
        {"content-mathml", Notation::ContentMathml},
        {"contentmathml", Notation::ContentMathml},
        {"unicode", Notation::Unicode},
        {"utf8", Notation::Unicode},
        {"ascii", Notation::Ascii},
        {"text", Notation::Ascii},
        {"sympy", Notation::SymPy},
        {"python-sympy", Notation::SymPy},
        {"mathematica", Notation::Mathematica},
        {"wolfram", Notation::Mathematica},
        {"c", Notation::C},
        {"c++", Notation::Cpp},
        {"cpp", Notation::Cpp},
        {"python", Notation::Python},
    };
    const std::string key = lowered(text);
    for (const auto& entry : kNames) {
        if (key == entry.name) {
            return entry.notation;
        }
    }
    std::string known;
    for (const auto& entry : kNames) {
        known += known.empty() ? "" : ", ";
        known += entry.name;
    }
    return std::unexpected(
        DomainError{"notation_from_name", "unknown notation '" + text + "'; known: " + known});
}

std::string latex_document(const std::string& body) {
    return "\\documentclass{article}\n"
           "\\usepackage{amsmath}\n"
           "\\usepackage{amssymb}\n"
           "\\begin{document}\n"
           "\\[\n" +
           body +
           "\n\\]\n"
           "\\end{document}\n";
}

std::string mathml_document(const std::string& body, bool display) {
    return std::string("<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"") +
           (display ? "block" : "inline") + "\">" + body + "</math>";
}

} // namespace ms::sym2
