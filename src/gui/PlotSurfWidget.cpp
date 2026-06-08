#include "gui/PlotSurfWidget.hpp"

#include <QMatrix4x4>
#include <QMouseEvent>
#include <QSurfaceFormat>
#include <QWheelEvent>
#include <QtMath>
#include <algorithm>
#include <cmath>

namespace {

void set_height_color(double z, double zmin, double zmax) {
    const float t = static_cast<float>((z - zmin) / std::max(zmax - zmin, 1e-12));
    glColor3f(0.2f + 0.8f * t, 0.45f + 0.3f * (1.0f - t), 1.0f - 0.6f * t);
}

void set_triangle_normal(float x0, float y0, float z0, float x1, float y1, float z1, float x2,
                         float y2, float z2) {
    const float ux = x1 - x0;
    const float uy = y1 - y0;
    const float uz = z1 - z0;
    const float vx = x2 - x0;
    const float vy = y2 - y0;
    const float vz = z2 - z0;
    const float nx = uy * vz - uz * vy;
    const float ny = uz * vx - ux * vz;
    const float nz = ux * vy - uy * vx;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-8f) {
        glNormal3f(nx / len, ny / len, nz / len);
    } else {
        glNormal3f(0.0f, 0.0f, 1.0f);
    }
}

} // namespace

PlotSurfWidget::PlotSurfWidget(QWidget* parent) : QOpenGLWidget(parent) {
    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    fmt.setVersion(2, 1);
    fmt.setProfile(QSurfaceFormat::CompatibilityProfile);
    setFormat(fmt);
    setMinimumHeight(180);
}

void PlotSurfWidget::set_surface(const ms::interp::PlotSeries& plot) {
    grid_ = plot.grid;
    has_surface_ = plot.valid && grid_.rows() > 0 && grid_.cols() > 0;
    update();
}

void PlotSurfWidget::clear_surface() {
    grid_ = ms::Matrix<double>{};
    has_surface_ = false;
    update();
}

bool PlotSurfWidget::export_png(const QString& path) {
    if (!has_surface_) {
        return false;
    }
    makeCurrent();
    const QImage image = grabFramebuffer();
    doneCurrent();
    return !image.isNull() && image.save(path);
}

void PlotSurfWidget::initializeGL() {
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    const GLfloat ambient[] = {0.18f, 0.18f, 0.22f, 1.0f};
    const GLfloat diffuse[] = {0.82f, 0.82f, 0.88f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
}

void PlotSurfWidget::resizeGL(int width, int height) {
    glViewport(0, 0, width, height);
}

void PlotSurfWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        dragging_ = true;
        last_pos_ = event->pos();
    }
    QOpenGLWidget::mousePressEvent(event);
}

void PlotSurfWidget::mouseMoveEvent(QMouseEvent* event) {
    if (dragging_ && (event->buttons() & Qt::LeftButton)) {
        const int dx = event->pos().x() - last_pos_.x();
        const int dy = event->pos().y() - last_pos_.y();
        yaw_ += static_cast<float>(dx) * 0.4f;
        pitch_ += static_cast<float>(dy) * 0.4f;
        pitch_ = qBound(-80.0f, pitch_, 80.0f);
        last_pos_ = event->pos();
        update();
    }
    QOpenGLWidget::mouseMoveEvent(event);
}

void PlotSurfWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        dragging_ = false;
    }
    QOpenGLWidget::mouseReleaseEvent(event);
}

void PlotSurfWidget::wheelEvent(QWheelEvent* event) {
    distance_ -= static_cast<float>(event->angleDelta().y()) * 0.001f;
    distance_ = qBound(1.0f, distance_, 8.0f);
    update();
    QOpenGLWidget::wheelEvent(event);
}

void PlotSurfWidget::paintGL() {
    glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!has_surface_) {
        return;
    }

    const int rows = static_cast<int>(grid_.rows());
    const int cols = static_cast<int>(grid_.cols());
    if (rows < 2 || cols < 2) {
        return;
    }

    double zmin = grid_(0, 0);
    double zmax = grid_(0, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            zmin = std::min(zmin, grid_(static_cast<size_t>(i), static_cast<size_t>(j)));
            zmax = std::max(zmax, grid_(static_cast<size_t>(i), static_cast<size_t>(j)));
        }
    }
    const double zscale = (zmax - zmin) > 1e-12 ? 0.35 / (zmax - zmin) : 1.0;

    QMatrix4x4 projection;
    projection.perspective(45.0f, static_cast<float>(width()) / static_cast<float>(height()), 0.1f,
                           100.0f);

    QMatrix4x4 view;
    view.translate(0.0f, 0.0f, -distance_);
    view.rotate(pitch_, 1.0f, 0.0f, 0.0f);
    view.rotate(yaw_, 0.0f, 1.0f, 0.0f);
    view.translate(-0.5f * static_cast<float>(cols - 1), -0.5f * static_cast<float>(rows - 1),
                   0.0f);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(projection.constData());
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(view.constData());

    const GLfloat light_pos[] = {0.35f, 0.75f, 1.4f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    auto height_at = [&](int row, int col) -> float {
        const double z = grid_(static_cast<size_t>(row), static_cast<size_t>(col));
        return static_cast<float>((z - zmin) * zscale);
    };

    glBegin(GL_TRIANGLES);
    for (int i = 0; i + 1 < rows; ++i) {
        for (int j = 0; j + 1 < cols; ++j) {
            const double z00 = grid_(static_cast<size_t>(i), static_cast<size_t>(j));
            const double z01 = grid_(static_cast<size_t>(i), static_cast<size_t>(j + 1));
            const double z10 = grid_(static_cast<size_t>(i + 1), static_cast<size_t>(j));
            const double z11 = grid_(static_cast<size_t>(i + 1), static_cast<size_t>(j + 1));

            const float x0 = static_cast<float>(j);
            const float y0 = static_cast<float>(i);
            const float x1 = static_cast<float>(j + 1);
            const float y1 = static_cast<float>(i + 1);
            const float h00 = height_at(i, j);
            const float h01 = height_at(i, j + 1);
            const float h10 = height_at(i + 1, j);
            const float h11 = height_at(i + 1, j + 1);

            set_triangle_normal(x0, y0, h00, x1, y0, h01, x0, y1, h10);
            set_height_color(z00, zmin, zmax);
            glVertex3f(x0, y0, h00);
            set_height_color(z01, zmin, zmax);
            glVertex3f(x1, y0, h01);
            set_height_color(z10, zmin, zmax);
            glVertex3f(x0, y1, h10);

            set_triangle_normal(x1, y0, h01, x1, y1, h11, x0, y1, h10);
            set_height_color(z01, zmin, zmax);
            glVertex3f(x1, y0, h01);
            set_height_color(z11, zmin, zmax);
            glVertex3f(x1, y1, h11);
            set_height_color(z10, zmin, zmax);
            glVertex3f(x0, y1, h10);
        }
    }
    glEnd();

    glDisable(GL_LIGHTING);
    glLineWidth(1.0f);
    glColor3f(0.85f, 0.85f, 0.85f);
    glBegin(GL_LINES);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float x = static_cast<float>(j);
            const float y = static_cast<float>(i);
            const float z = height_at(i, j);
            if (j + 1 < cols) {
                glVertex3f(x, y, z);
                glVertex3f(x + 1.0f, y, height_at(i, j + 1));
            }
            if (i + 1 < rows) {
                glVertex3f(x, y, z);
                glVertex3f(x, y + 1.0f, height_at(i + 1, j));
            }
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}
