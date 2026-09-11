// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include "ms/interp/repl_engine.hpp"
#include <string>

namespace ms::interp {

/// A compact table for the screen: the first few rows and columns, rounded to four
/// decimals with the padding trimmed. Rounded, but never to nothing -- an entry of 1e-5
/// used to render "0.0000", so the preview asserted a matrix of zeros.
std::string format_plot_preview(const PlotSeries& plot);

/// The whole series or grid, written exactly, for `saveplot`.
///
/// That command used to write the preview: the first ten rows of the first sixteen
/// columns, at four decimals. It said "saved plot preview", which is true, and what
/// landed in the file was a truncated table of rounded numbers where the user had asked
/// for their data. A file is read back, so it carries all of it and carries it exactly.
std::string format_plot_data(const PlotSeries& plot);

} // namespace ms::interp
