// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

/// @file
/// @brief Conversion between the new `ms::sym2` core and the existing
///   `ms::SymExpr`, for the period in which both exist.
///
/// §10.5 says not to rewrite in place: the old engine carries Laplace, Mellin, Hankel,
/// Fourier and Z transforms, series, limits, linear solve and separable ODEs, and that
/// is worth keeping while each is ported and differentially tested. These two functions
/// are what lets a caller reach either one, and what lets a test run the same
/// expression through both and compare.
///
/// The conversion is not symmetric, and the asymmetry is the point:
///
/// **`from_legacy` loses nothing.** Every `SymOp` has a `sym2` counterpart. A `Const`
/// becomes an exact `Rational` when the double it holds is exactly a manageable
/// fraction -- which every literal a person types is -- and a `Real` otherwise, so the
/// exactness the old core could not represent is recovered where it is recoverable
/// rather than invented where it is not.
///
/// **`to_legacy` loses what the old core cannot hold.** An exact 1/3 becomes the
/// nearest double; `Function` heads outside the old six become an opaque variable
/// whose name is the printed call, which is not something the old engine can
/// differentiate or transform. `to_legacy` reports rather than silently degrading:
/// the `lossy` flag says whether anything was approximated on the way through.

#include <string>

#include "ms/symbolic/symbolic.hpp"
#include "ms/sym2/expr.hpp"

namespace ms::sym2 {

/// Convert a legacy expression into the new core. Total: every `SymOp` maps.
///
/// The unsupported-result sentinel of the old API -- a `SymOp::Deriv` node standing for
/// "no closed form" -- converts to a `Derivative` head, which is what it always meant.
/// In the new core that is a representable answer rather than a value the caller has to
/// know to test for, which is the §10.2 fix.
ExprRef from_legacy(const SymExpr& expr);

/// Convert back. `lossy` is set when the result is an approximation of the input:
/// an exact rational that had to become a double, an exponent too large to survive the
/// old representation, or a function head the old core has no operator for.
SymExpr to_legacy(const ExprRef& expr, bool* lossy = nullptr);

/// Parse through the legacy parser and convert, which is how `sym2` gets a front end
/// before it has one of its own.
Result<ExprRef> parse(const std::string& text);

} // namespace ms::sym2
