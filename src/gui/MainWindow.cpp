#include "gui/MainWindow.hpp"
#include "gui/PlotSurfWidget.hpp"
#include "gui/PlotWidget.hpp"

#include <QApplication>
#include <QAction>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileSystemModel>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QPushButton>
#include <QTextCursor>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidget>

#include "ms/distributed/mpi_context.hpp"
#include "ms/runtime/topology.hpp"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("MathScript IDE");
    resize(1200, 720);
    apply_dark_theme();
    setup_menus();

    main_splitter_ = new QSplitter(Qt::Horizontal, this);

    auto* center = new QWidget(main_splitter_);
    auto* center_layout = new QVBoxLayout(center);

    editor_ = new QPlainTextEdit(center);
    editor_->setPlaceholderText("Script editor (open a file from the browser)");
    editor_->setMaximumHeight(160);
    center_layout->addWidget(editor_);

    output_ = new QPlainTextEdit(center);
    output_->setReadOnly(true);
    output_->setPlaceholderText("MathScript output");
    center_layout->addWidget(output_, 2);

    plot_stack_ = new QStackedWidget(center);
    plot_2d_ = new PlotWidget(plot_stack_);
    plot_3d_ = new PlotSurfWidget(plot_stack_);
    plot_stack_->addWidget(plot_2d_);
    plot_stack_->addWidget(plot_3d_);
    center_layout->addWidget(plot_stack_);

    auto* row = new QHBoxLayout();
    input_ = new QLineEdit(center);
    input_->setPlaceholderText("REPL: help, plot([1,2,3,4]), save session.ms");
    auto* run = new QPushButton("Run", center);
    auto* run_script = new QPushButton("Run Script", center);
    row->addWidget(input_);
    row->addWidget(run);
    row->addWidget(run_script);
    center_layout->addLayout(row);

    variables_ = new QListWidget(main_splitter_);
    variables_->setMinimumWidth(220);
    main_splitter_->addWidget(center);
    main_splitter_->addWidget(variables_);
    main_splitter_->setStretchFactor(0, 4);
    main_splitter_->setStretchFactor(1, 1);

    setCentralWidget(main_splitter_);

    files_dock_ = new QDockWidget("Files", this);
    file_model_ = new QFileSystemModel(this);
    file_model_->setRootPath(QDir::currentPath());
    file_tree_ = new QTreeView(files_dock_);
    file_tree_->setModel(file_model_);
    file_tree_->setRootIndex(file_model_->index(QDir::currentPath()));
    file_tree_->hideColumn(1);
    file_tree_->hideColumn(2);
    file_tree_->hideColumn(3);
    files_dock_->setWidget(file_tree_);
    addDockWidget(Qt::LeftDockWidgetArea, files_dock_);

    status_timer_ = new QTimer(this);
    connect(status_timer_, &QTimer::timeout, this, &MainWindow::refresh_status);
    status_timer_->start(2000);
    refresh_status();

    connect(run, &QPushButton::clicked, this, &MainWindow::on_submit);
    connect(run_script, &QPushButton::clicked, this, [this]() {
        const QString script = editor_->toPlainText();
        for (const QString& line : script.split('\n', Qt::SkipEmptyParts)) {
            input_->setText(line);
            on_submit();
        }
    });
    connect(input_, &QLineEdit::returnPressed, this, &MainWindow::on_submit);
    connect(file_tree_, &QTreeView::activated, this, &MainWindow::on_file_activated);

    append_output("MathScript IDE ready. Type 'help' and press Run.\n");
    refresh_variables();
}

void MainWindow::apply_dark_theme() {
    QPalette palette;
    palette.setColor(QPalette::Window, QColor(45, 45, 48));
    palette.setColor(QPalette::WindowText, QColor(220, 220, 220));
    palette.setColor(QPalette::Base, QColor(30, 30, 30));
    palette.setColor(QPalette::AlternateBase, QColor(45, 45, 48));
    palette.setColor(QPalette::Text, QColor(220, 220, 220));
    palette.setColor(QPalette::Button, QColor(60, 60, 60));
    palette.setColor(QPalette::ButtonText, QColor(220, 220, 220));
    palette.setColor(QPalette::Highlight, QColor(64, 156, 255));
    palette.setColor(QPalette::HighlightedText, Qt::black);
    qApp->setPalette(palette);
}

void MainWindow::setup_menus() {
    auto* file_menu = menuBar()->addMenu("&File");
    auto* open_action = file_menu->addAction("Open...");
    auto* save_session_action = file_menu->addAction("Save Session...");
    auto* load_session_action = file_menu->addAction("Load Session...");
    auto* export_plot_action = file_menu->addAction("Export Plot as PNG...");
    file_menu->addSeparator();
    auto* quit_action = file_menu->addAction("Quit");

    connect(open_action, &QAction::triggered, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(this, "Open Script");
        if (path.isEmpty()) {
            return;
        }
        QFile file(path);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            editor_->setPlainText(QString::fromUtf8(file.readAll()));
        }
    });
    connect(save_session_action, &QAction::triggered, this, [this]() {
        const QString path = QFileDialog::getSaveFileName(this, "Save Session", {}, "MathScript (*.ms)");
        if (path.isEmpty()) {
            return;
        }
        const auto result = interp_.save_session(path.toStdString());
        if (result) {
            append_output("saved session to " + path + "\n");
            refresh_variables();
        } else {
            append_output("error: " + QString::fromStdString(ms::format_error(result.error())) + "\n");
        }
    });
    connect(load_session_action, &QAction::triggered, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(this, "Load Session", {}, "MathScript (*.ms)");
        if (path.isEmpty()) {
            return;
        }
        const auto result = interp_.load_session(path.toStdString());
        if (result) {
            append_output("loaded session from " + path + "\n");
            refresh_variables();
            refresh_plot();
        } else {
            append_output("error: " + QString::fromStdString(ms::format_error(result.error())) + "\n");
        }
    });
    connect(export_plot_action, &QAction::triggered, this, &MainWindow::export_plot_png);
    connect(quit_action, &QAction::triggered, qApp, &QApplication::quit);
}

void MainWindow::refresh_status() {
    const auto topo = ms::detect_topology();
    static ms::distributed::MPIContext mpi = ms::distributed::init(0, nullptr);
    statusBar()->showMessage(QString("CPU %1 | GPUs %2 | MPI rank %3/%4")
                                 .arg(static_cast<int>(topo.total_threads()))
                                 .arg(topo.total_gpus)
                                 .arg(ms::distributed::rank(mpi))
                                 .arg(ms::distributed::size(mpi)));
}

void MainWindow::append_output(const QString& text) {
    output_->moveCursor(QTextCursor::End);
    output_->insertPlainText(text);
    output_->moveCursor(QTextCursor::End);
}

void MainWindow::refresh_variables() {
    variables_->clear();
    const auto& state = interp_.state();
    for (const auto& [name, value] : state.scalars) {
        variables_->addItem(QString("%1 = %2").arg(QString::fromStdString(name)).arg(value));
    }
    for (const auto& [name, matrix] : state.matrices) {
        variables_->addItem(QString("%1 (%2x%3)")
                                 .arg(QString::fromStdString(name))
                                 .arg(static_cast<int>(matrix.rows()))
                                 .arg(static_cast<int>(matrix.cols())));
    }
    if (variables_->count() == 0) {
        variables_->addItem("(no variables)");
    }
}

void MainWindow::refresh_plot() {
    if (!interp_.has_plot()) {
        plot_2d_->clear_plot();
        plot_3d_->clear_surface();
        plot_stack_->setCurrentWidget(plot_2d_);
        return;
    }

    const auto& plot = interp_.plot();
    if (plot.kind == ms::interp::PlotSeries::Kind::Surface3D) {
        plot_3d_->set_surface(plot);
        plot_stack_->setCurrentWidget(plot_3d_);
        return;
    }

    plot_2d_->set_plot(plot);
    plot_stack_->setCurrentWidget(plot_2d_);
}

void MainWindow::export_plot_png() {
    if (!interp_.has_plot()) {
        append_output("error: no plot to export\n");
        return;
    }

    const QString path =
        QFileDialog::getSaveFileName(this, "Export Plot", {}, "PNG Image (*.png)");
    if (path.isEmpty()) {
        return;
    }

    const bool ok = interp_.plot().kind == ms::interp::PlotSeries::Kind::Surface3D
                          ? plot_3d_->export_png(path)
                          : plot_2d_->export_png(path);
    if (ok) {
        append_output("exported plot to " + path + "\n");
    } else {
        append_output("error: failed to export plot to " + path + "\n");
    }
}

void MainWindow::on_file_activated(const QModelIndex& index) {
    const QString path = file_model_->filePath(index);
    if (file_model_->isDir(index)) {
        return;
    }
    QFile file(path);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        editor_->setPlainText(QString::fromUtf8(file.readAll()));
        append_output("opened " + path + "\n");
    }
}

void MainWindow::on_submit() {
    const QString line = input_->text().trimmed();
    if (line.isEmpty()) {
        return;
    }
    append_output("ms> " + line + "\n");
    input_->clear();

    const auto result = interp_.execute(line.toStdString());
    if (result) {
        if (!result->empty()) {
            append_output(QString::fromStdString(*result));
        }
    } else {
        append_output("error: " + QString::fromStdString(ms::format_error(result.error())) + "\n");
    }
    refresh_variables();
    refresh_plot();
}
