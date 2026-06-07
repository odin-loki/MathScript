#include "gui/PlotWidget.hpp"

#include <QPainter>
#include <QPaintEvent>
#include <algorithm>
#include <cmath>

PlotWidget::PlotWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(180);
}

void PlotWidget::set_series(
    const std::vector<double>& x,
    const std::vector<double>& y,
    bool bar_chart) {
    x_ = x;
    y_ = y;
    bar_chart_ = bar_chart;
    update();
}

void PlotWidget::clear_series() {
    x_.clear();
    y_.clear();
    bar_chart_ = false;
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

    if (x_.empty() || y_.size() != x_.size()) {
        painter.setPen(QColor(160, 160, 160));
        painter.drawText(rect(), Qt::AlignCenter, "plot([y...]) or hist([...])");
        return;
    }

    const auto [x_min_it, x_max_it] = std::minmax_element(x_.begin(), x_.end());
    const auto [y_min_it, y_max_it] = std::minmax_element(y_.begin(), y_.end());
    double x_min = *x_min_it;
    double x_max = *x_max_it;
    double y_min = bar_chart_ ? 0.0 : *y_min_it;
    double y_max = *y_max_it;
    if (std::abs(x_max - x_min) < 1e-12) {
        x_min -= 1.0;
        x_max += 1.0;
    }
    if (std::abs(y_max - y_min) < 1e-12) {
        y_max = y_min + 1.0;
    }

    auto map_x = [&](double x) {
        return plot_rect.left() +
               static_cast<int>((x - x_min) / (x_max - x_min) * plot_rect.width());
    };
    auto map_y = [&](double y) {
        return plot_rect.bottom() -
               static_cast<int>((y - y_min) / (y_max - y_min) * plot_rect.height());
    };

    painter.setPen(QColor(180, 180, 180));
    painter.drawText(8, 16, QString("n=%1").arg(static_cast<int>(x_.size())));

    if (bar_chart_) {
        painter.setBrush(QColor(64, 156, 255));
        painter.setPen(Qt::NoPen);
        const double bar_width = (x_max - x_min) / static_cast<double>(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
            const int left = map_x(x_[i] - bar_width * 0.4);
            const int right = map_x(x_[i] + bar_width * 0.4);
            const int top = map_y(y_[i]);
            const int bottom = map_y(0.0);
            painter.drawRect(QRect(left, top, std::max(2, right - left), bottom - top));
        }
        return;
    }

    painter.setPen(QPen(QColor(64, 156, 255), 2));
    if (x_.size() == 1) {
        painter.setBrush(QColor(64, 156, 255));
        const int px = map_x(x_[0]);
        const int py = map_y(y_[0]);
        painter.drawEllipse(QPoint(px, py), 4, 4);
    } else {
        for (size_t i = 1; i < x_.size(); ++i) {
            painter.drawLine(map_x(x_[i - 1]), map_y(y_[i - 1]), map_x(x_[i]), map_y(y_[i]));
        }
    }
}
