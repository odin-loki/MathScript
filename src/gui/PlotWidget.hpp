#pragma once

#include "ms/interp/repl_engine.hpp"
#include <QString>
#include <QWidget>
#include <vector>

class PlotWidget : public QWidget {
    Q_OBJECT

public:
    explicit PlotWidget(QWidget* parent = nullptr);

    void set_plot(const ms::interp::PlotSeries& plot);
    void clear_plot();
    bool export_png(const QString& path) const;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void paint_series(QPainter& painter, const QRect& plot_rect) const;
    void paint_matrix(QPainter& painter, const QRect& plot_rect) const;

    ms::interp::PlotSeries plot_;
    bool has_plot_ = false;
};
