// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

/// @file
/// @brief §11.2: read back the LaTeX §11.1 writes.
///
/// The plan's assessment of this item is the design: "LaTeX is presentation markup and
/// there is no correct general parser, so the work is to define a subset, parse it
/// strictly, and reject everything outside it with a source position."
///
/// The subset is defined by the printer rather than by taste. `parse_latex` accepts
/// everything `to_latex` can emit -- under every `NotationOptions` combination -- plus
/// the twelve human spellings `docs/LATEX_SUBSET.md` lists by name, and refuses
/// everything else with a line, a column, what was expected and what was found.
///
/// That definition is what makes the property worth asserting:
///
///     parse_latex(to_latex(e, options)) == e
///
/// for every expression and every option set, with `docs/LATEX_SUBSET.md` §4.2 listing
/// exhaustively the shapes where it does not hold. Each entry there is a case where the
/// *printed* form carries less than the node did -- a total and a partial derivative
/// are spelled the same way, an `\mathrm{name}` is both a multi-letter symbol and an
/// unnamed constant -- and the honest answer is to say so rather than to guess.
///
/// A general LaTeX reader has to guess, and each guess is a chance to hand back an
/// expression the author did not write. That is the defect class this project's audits
/// kept finding, and refusing to guess is the only way it does not arrive here too.

#include <string>
#include <string_view>

#include "ms/core/matrix.hpp"
#include "ms/sym2/expr.hpp"
#include "ms/sym2/notation.hpp"

namespace ms::sym2 {

/// The expression `text` denotes, or a `ParseError` carrying a 1-based line and column.
///
/// `options.decimal_separator` selects which spelling of a decimal is read. Every other
/// field is ignored: the grammar accepts the output of all option settings at once,
/// which is what makes the round-trip property independent of how the printer was
/// configured.
///
/// The column counts UTF-8 scalar values rather than bytes, because a symbol name may
/// carry raw non-ASCII and a byte column would point into the middle of a character.
Result<ExprRef> parse_latex(std::string_view text, const NotationOptions& options = {});

/// The matrix `text` denotes: a `\begin{pmatrix}` and its siblings, whose cells are
/// numbers. This is a separate entry point rather than a case of `parse_latex` for the
/// same reason `to_latex(const Matrix<double>&)` is a separate overload -- `Head` has
/// no matrix member, so a matrix is not an expression here and reading one as if it
/// were would need a node that does not exist.
Result<Matrix<double>> parse_latex_matrix(std::string_view text,
                                          const NotationOptions& options = {});

} // namespace ms::sym2
