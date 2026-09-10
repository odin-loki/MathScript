// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/interp/plot_console.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "ms/core/format.hpp"

namespace ms::interp {

namespace {

constexpr size_t kMaxPreviewRows = 10;
constexpr size_t kMaxPreviewCols = 16;
constexpr size_t kMaxPointPreview = 6;

void append_matrix_preview(std::ostringstream& out, const Matrix<double>& m, const char* label) {
    if (m.rows() == 0 || m.cols() == 0) {
        return;
    }
    double vmin = m(0, 0);
    double vmax = m(0, 0);
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            vmin = std::min(vmin, m(i, j));
            vmax = std::max(vmax, m(i, j));
        }
    }
    // format_preview, not setprecision(4): four decimals renders 1e-5 as "0.0000", so
    // an imshow of a small-valued grid previewed as a matrix of zeros with a zero-width
    // range. The compact spelling is kept wherever it is faithful and only widens where
    // it would otherwise say zero.
    out << label << " (" << m.rows() << "x" << m.cols() << ") range [" << format_preview(vmin, 4, false)
        << ", " << format_preview(vmax, 4, false) << "]\n";

    const size_t rows = std::min(m.rows(), kMaxPreviewRows);
    const size_t cols = std::min(m.cols(), kMaxPreviewCols);
    for (size_t i = 0; i < rows; ++i) {
        out << "  ";
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) {
                out << ' ';
            }
            out << std::setw(8) << format_preview(m(i, j), 4, false);
        }
        if (m.cols() > cols) {
            out << " ...";
        }
        out << '\n';
    }
    if (m.rows() > rows) {
        out << "  ...\n";
    }
}

const char* plot_kind_label(PlotSeries::Kind kind) {
    switch (kind) {
    case PlotSeries::Kind::Line:
        return "line";
    case PlotSeries::Kind::Scatter:
        return "scatter";
    case PlotSeries::Kind::Bar:
        return "bar";
    case PlotSeries::Kind::Heatmap:
        return "heatmap";
    case PlotSeries::Kind::Spy:
        return "spy";
    case PlotSeries::Kind::Surface3D:
        return "surface3d";
    }
    return "line";
}

} // namespace

std::string format_plot_preview(const PlotSeries& plot) {
    if (!plot.valid) {
        return "(no plot)\n";
    }

    std::ostringstream out;
    switch (plot.kind) {
    case PlotSeries::Kind::Heatmap:
    case PlotSeries::Kind::Surface3D:
        if (plot.grid.rows() > 0 && plot.grid.cols() > 0) {
            const char* label = plot.kind == PlotSeries::Kind::Surface3D ? "Z" : "data";
            append_matrix_preview(out, plot.grid, label);
        } else {
            out << "matrix plot (" << plot.matrix_rows << "x" << plot.matrix_cols << ")\n";
        }
        break;
    case PlotSeries::Kind::Spy:
        if (plot.grid.rows() > 0 && plot.grid.cols() > 0) {
            out << "spy pattern (" << plot.grid.rows() << "x" << plot.grid.cols() << ", "
                << plot.nnz << " nnz)\n";
            const size_t rows = std::min(plot.grid.rows(), kMaxPreviewRows);
            const size_t cols = std::min(plot.grid.cols(), kMaxPreviewCols);
            for (size_t i = 0; i < rows; ++i) {
                out << "  ";
                for (size_t j = 0; j < cols; ++j) {
                    out << (plot.grid(i, j) != 0.0 ? '#' : '.');
                }
                if (plot.grid.cols() > cols) {
                    out << "...";
                }
                out << '\n';
            }
            if (plot.grid.rows() > rows) {
                out << "  ...\n";
            }
        } else {
            out << "spy (" << plot.matrix_rows << "x" << plot.matrix_cols << ", " << plot.nnz
                << " nnz)\n";
        }
        break;
    case PlotSeries::Kind::Line:
    case PlotSeries::Kind::Scatter:
    case PlotSeries::Kind::Bar: {
        const char* label = plot.kind == PlotSeries::Kind::Bar       ? "histogram"
                            : plot.kind == PlotSeries::Kind::Scatter ? "scatter"
                                                                     : "plot";
        out << label << " (" << plot.y.size() << " points)\n";
        const size_t n = std::min(plot.y.size(), kMaxPointPreview);
        for (size_t i = 0; i < n; ++i) {
            out << "  ";
            if (!plot.x.empty()) {
                out << format_preview(plot.x[i]) << " -> ";
            }
            out << format_preview(plot.y[i]) << '\n';
        }
        if (plot.y.size() > n) {
            out << "  ...\n";
        }
        break;
    }
    }
    return out.str();
}

std::string format_plot_data(const PlotSeries& plot) {
    if (!plot.valid) {
        return "(no plot)\n";
    }
    std::ostringstream out;
    out << "kind " << plot_kind_label(plot.kind) << "\n";
    if (plot.grid.rows() > 0 && plot.grid.cols() > 0) {
        out << "grid " << plot.grid.rows() << " " << plot.grid.cols() << "\n";
        for (size_t i = 0; i < plot.grid.rows(); ++i) {
            for (size_t j = 0; j < plot.grid.cols(); ++j) {
                if (j > 0) {
                    out << ' ';
                }
                out << format_exact(plot.grid(i, j));
            }
            out << '\n';
        }
        return out.str();
    }
    out << "points " << plot.y.size() << "\n";
    for (size_t i = 0; i < plot.y.size(); ++i) {
        if (i < plot.x.size()) {
            out << format_exact(plot.x[i]) << ' ';
        }
        out << format_exact(plot.y[i]) << '\n';
    }
    return out.str();
}

} // namespace ms::interp
