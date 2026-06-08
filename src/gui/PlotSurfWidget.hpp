#pragma once

#include "ms/interp/repl_engine.hpp"
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPoint>
#include <QString>

class QMouseEvent;
class QWheelEvent;

class PlotSurfWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    explicit PlotSurfWidget(QWidget* parent = nullptr);

    void set_surface(const ms::interp::PlotSeries& plot);
    void clear_surface();
    bool export_png(const QString& path);

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    ms::Matrix<double> grid_;
    bool has_surface_ = false;
    bool dragging_ = false;
    QPoint last_pos_;
    float yaw_ = 35.0f;
    float pitch_ = 28.0f;
    float distance_ = 2.2f;
};
