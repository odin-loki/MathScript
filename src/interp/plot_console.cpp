#include "ms/interp/plot_console.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

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
    out << label << " (" << m.rows() << "x" << m.cols() << ") range [" << std::fixed
        << std::setprecision(4) << vmin << ", " << vmax << "]\n";

    const size_t rows = std::min(m.rows(), kMaxPreviewRows);
    const size_t cols = std::min(m.cols(), kMaxPreviewCols);
    out << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < rows; ++i) {
        out << "  ";
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) {
                out << ' ';
            }
            out << std::setw(8) << m(i, j);
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
                out << plot.x[i] << " -> ";
            }
            out << plot.y[i] << '\n';
        }
        if (plot.y.size() > n) {
            out << "  ...\n";
        }
        break;
    }
    }
    return out.str();
}

} // namespace ms::interp
