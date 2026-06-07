#pragma once

#include <QWidget>
#include <vector>

class PlotWidget : public QWidget {
    Q_OBJECT

public:
    explicit PlotWidget(QWidget* parent = nullptr);

    void set_series(
        const std::vector<double>& x,
        const std::vector<double>& y,
        bool bar_chart = false);
    void clear_series();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<double> x_;
    std::vector<double> y_;
    bool bar_chart_ = false;
};
