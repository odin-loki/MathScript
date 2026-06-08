#include "gui/PlotWidget.hpp"

#include <QPainter>
#include <QPaintEvent>
#include <algorithm>
#include <cmath>

namespace {

QColor heat_color(double value, double vmin, double vmax) {
    if (std::abs(vmax - vmin) < 1e-12) {
        return QColor(64, 156, 255);
    }
    const double t = std::clamp((value - vmin) / (vmax - vmin), 0.0, 1.0);
    const int r = static_cast<int>(255.0 * t);
    const int b = static_cast<int>(255.0 * (1.0 - t));
    return QColor(r, 80, b);
}

} // namespace

PlotWidget::PlotWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(180);
}

void PlotWidget::set_plot(const ms::interp::PlotSeries& plot) {
    plot_ = plot;
    has_plot_ = plot.valid;
    update();
}

void PlotWidget::clear_plot() {
    plot_ = ms::interp::PlotSeries{};
    has_plot_ = false;
    update();
}

void PlotWidget::paintEvent(QPaintEvent* event) {
    QWidget::paintEvent(event);

    QPainter painter(this);
    painter.fillRect(rect(), QColor(30, 30, 30));

    const int pad = 28;
    const QRect plot_rect(pad, pad, width() - 2 * pad, height() - 2 * pad);
    painter.setPen(QColor(80, 80, 80));
    painter.drawRect(plot_rect);

    if (!has_plot_) {
        painter.setPen(QColor(160, 160, 160));
        painter.drawText(rect(), Qt::AlignCenter, "plot([y...]) or surf([z...])");
        return;
    }

    switch (plot_.kind) {
    case ms::interp::PlotSeries::Kind::Heatmap:
    case ms::interp::PlotSeries::Kind::Spy:
    case ms::interp::PlotSeries::Kind::Surface3D:
        paint_matrix(painter, plot_rect);
        return;
    default:
        paint_series(painter, plot_rect);
        return;
    }
}

void PlotWidget::paint_series(QPainter& painter, const QRect& plot_rect) const {
    const auto& x = plot_.x;
    const auto& y = plot_.y;
    if (x.empty() || y.size() != x.size()) {
        painter.setPen(QColor(160, 160, 160));
        painter.drawText(rect(), Qt::AlignCenter, "plot([y...]) or hist([...])");
        return;
    }

    const auto [x_min_it, x_max_it] = std::minmax_element(x.begin(), x.end());
    const auto [y_min_it, y_max_it] = std::minmax_element(y.begin(), y.end());
    double x_min = *x_min_it;
    double x_max = *x_max_it;
    const bool bar_chart = plot_.kind == ms::interp::PlotSeries::Kind::Bar;
    const bool scatter = plot_.kind == ms::interp::PlotSeries::Kind::Scatter;
    double y_min = bar_chart ? 0.0 : *y_min_it;
    double y_max = *y_max_it;
    if (std::abs(x_max - x_min) < 1e-12) {
        x_min -= 1.0;
        x_max += 1.0;
    }
    if (std::abs(y_max - y_min) < 1e-12) {
        y_max = y_min + 1.0;
    }

    auto map_x = [&](double xv) {
        return plot_rect.left() +
               static_cast<int>((xv - x_min) / (x_max - x_min) * plot_rect.width());
    };
    auto map_y = [&](double yv) {
        return plot_rect.bottom() -
               static_cast<int>((yv - y_min) / (y_max - y_min) * plot_rect.height());
    };

    painter.setPen(QColor(180, 180, 180));
    painter.drawText(8, 16, QString("n=%1").arg(static_cast<int>(x.size())));

    if (bar_chart) {
        painter.setBrush(QColor(64, 156, 255));
        painter.setPen(Qt::NoPen);
        const double bar_width = (x_max - x_min) / static_cast<double>(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            const int left = map_x(x[i] - bar_width * 0.4);
            const int right = map_x(x[i] + bar_width * 0.4);
            const int top = map_y(y[i]);
            const int bottom = map_y(0.0);
            painter.drawRect(QRect(left, top, std::max(2, right - left), bottom - top));
        }
        return;
    }

    if (scatter || x.size() == 1) {
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(64, 156, 255));
        for (size_t i = 0; i < x.size(); ++i) {
            painter.drawEllipse(QPoint(map_x(x[i]), map_y(y[i])), 4, 4);
        }
        return;
    }

    painter.setPen(QPen(QColor(64, 156, 255), 2));
    for (size_t i = 1; i < x.size(); ++i) {
        painter.drawLine(map_x(x[i - 1]), map_y(y[i - 1]), map_x(x[i]), map_y(y[i]));
    }
}

void PlotWidget::paint_matrix(QPainter& painter, const QRect& plot_rect) const {
    const auto& grid = plot_.grid;
    if (grid.rows() == 0 || grid.cols() == 0) {
        painter.setPen(QColor(160, 160, 160));
        painter.drawText(rect(), Qt::AlignCenter, "empty matrix plot");
        return;
    }

    double vmin = grid(0, 0);
    double vmax = grid(0, 0);
    for (size_t i = 0; i < grid.rows(); ++i) {
        for (size_t j = 0; j < grid.cols(); ++j) {
            vmin = std::min(vmin, grid(i, j));
            vmax = std::max(vmax, grid(i, j));
        }
    }

    const int cell_w = std::max(2, plot_rect.width() / static_cast<int>(grid.cols()));
    const int cell_h = std::max(2, plot_rect.height() / static_cast<int>(grid.rows()));

    painter.setPen(QColor(180, 180, 180));
    if (plot_.kind == ms::interp::PlotSeries::Kind::Surface3D) {
        painter.drawText(8, 16, QString("surf %1x%2").arg(static_cast<int>(grid.rows())).arg(static_cast<int>(grid.cols())));
    } else if (plot_.kind == ms::interp::PlotSeries::Kind::Spy) {
        painter.drawText(8, 16, QString("spy %1 nnz").arg(static_cast<qulonglong>(plot_.nnz)));
    } else {
        painter.drawText(8, 16, QString("imshow %1x%2").arg(static_cast<int>(grid.rows())).arg(static_cast<int>(grid.cols())));
    }

    painter.setPen(Qt::NoPen);
    for (size_t i = 0; i < grid.rows(); ++i) {
        for (size_t j = 0; j < grid.cols(); ++j) {
            const int x = plot_rect.left() + static_cast<int>(j) * cell_w;
            const int y = plot_rect.top() + static_cast<int>(i) * cell_h;
            if (plot_.kind == ms::interp::PlotSeries::Kind::Spy) {
                painter.setBrush(grid(i, j) != 0.0 ? QColor(64, 156, 255) : QColor(50, 50, 50));
            } else {
                painter.setBrush(heat_color(grid(i, j), vmin, vmax));
            }
            painter.drawRect(x, y, cell_w, cell_h);
        }
    }
}

bool PlotWidget::export_png(const QString& path) const {
    if (!has_plot_) {
        return false;
    }
    return grab().toImage().save(path);
}
