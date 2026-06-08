#pragma once

#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTimer>

#include "ms/interp/repl_engine.hpp"

class PlotWidget;
class PlotSurfWidget;
class QDockWidget;
class QTreeView;
class QFileSystemModel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void on_submit();
    void on_file_activated(const QModelIndex& index);
    void refresh_status();

private:
    void append_output(const QString& text);
    void refresh_variables();
    void refresh_plot();
    void export_plot_png();
    void setup_menus();
    void apply_dark_theme();

    QSplitter* main_splitter_ = nullptr;
    QPlainTextEdit* output_ = nullptr;
    QPlainTextEdit* editor_ = nullptr;
    QLineEdit* input_ = nullptr;
    QListWidget* variables_ = nullptr;
    QStackedWidget* plot_stack_ = nullptr;
    PlotWidget* plot_2d_ = nullptr;
    PlotSurfWidget* plot_3d_ = nullptr;
    QDockWidget* files_dock_ = nullptr;
    QTreeView* file_tree_ = nullptr;
    QFileSystemModel* file_model_ = nullptr;
    QTimer* status_timer_ = nullptr;
    ms::interp::Interpreter interp_;
};
