#include "gui/MainWindow.hpp"
#include "gui/PlotSurfWidget.hpp"
#include "gui/PlotWidget.hpp"
#include "gui/ReplWorker.hpp"
#include "gui/ScriptHighlighter.hpp"

#include <QApplication>
#include <QAction>
#include <QKeyEvent>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileSystemModel>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMetaType>
#include <QTextCursor>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidget>

#include "ms/distributed/mpi_context.hpp"
#include "ms/runtime/topology.hpp"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    qRegisterMetaType<ms::interp::SessionState>("ms::interp::SessionState");
    qRegisterMetaType<ms::interp::PlotSeries>("ms::interp::PlotSeries");

    repl_thread_ = new QThread(this);
    repl_worker_ = new ReplWorker();
    repl_worker_->moveToThread(repl_thread_);
    repl_thread_->start();

    connect(repl_worker_, &ReplWorker::finished, this, &MainWindow::on_repl_finished);
    connect(repl_worker_, &ReplWorker::error, this, &MainWindow::on_repl_error);

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
    new ScriptHighlighter(editor_->document());
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
    run_ = new QPushButton("Run", center);
    run_script_ = new QPushButton("Run Script", center);
    row->addWidget(input_);
    row->addWidget(run_);
    row->addWidget(run_script_);
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

    connect(run_, &QPushButton::clicked, this, &MainWindow::on_submit);
    connect(run_script_, &QPushButton::clicked, this, &MainWindow::on_run_script);
    connect(input_, &QLineEdit::returnPressed, this, &MainWindow::on_submit);
    input_->installEventFilter(this);
    connect(file_tree_, &QTreeView::activated, this, &MainWindow::on_file_activated);

    append_output("MathScript IDE ready. Type 'help' and press Run.\n");
    refresh_variables();
}

MainWindow::~MainWindow() {
    repl_thread_->quit();
    repl_thread_->wait();
    delete repl_worker_;
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
        if (repl_busy_) {
            append_output("error: wait for the current REPL command to finish\n");
            return;
        }
        const QString path = QFileDialog::getSaveFileName(this, "Save Session", {}, "MathScript (*.ms)");
        if (path.isEmpty()) {
            return;
        }
        QString err;
        QMetaObject::invokeMethod(
            repl_worker_, "saveSession", Qt::BlockingQueuedConnection, Q_RETURN_ARG(QString, err),
            Q_ARG(QString, path));
        if (err.isEmpty()) {
            append_output("saved session to " + path + "\n");
            refresh_variables();
        } else {
            append_output("error: " + err + "\n");
        }
    });
    connect(load_session_action, &QAction::triggered, this, [this]() {
        if (repl_busy_) {
            append_output("error: wait for the current REPL command to finish\n");
            return;
        }
        const QString path = QFileDialog::getOpenFileName(this, "Load Session", {}, "MathScript (*.ms)");
        if (path.isEmpty()) {
            return;
        }
        QString err;
        QMetaObject::invokeMethod(
            repl_worker_, "loadSession", Qt::BlockingQueuedConnection, Q_RETURN_ARG(QString, err),
            Q_ARG(QString, path));
        if (err.isEmpty()) {
            append_output("loaded session from " + path + "\n");
            refresh_variables();
            refresh_plot();
        } else {
            append_output("error: " + err + "\n");
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
    ms::interp::SessionState state;
    QMetaObject::invokeMethod(repl_worker_, "sessionState", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(ms::interp::SessionState, state));
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
    bool has_plot = false;
    QMetaObject::invokeMethod(repl_worker_, "hasPlot", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(bool, has_plot));
    if (!has_plot) {
        plot_2d_->clear_plot();
        plot_3d_->clear_surface();
        plot_stack_->setCurrentWidget(plot_2d_);
        return;
    }

    ms::interp::PlotSeries plot;
    QMetaObject::invokeMethod(repl_worker_, "plot", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(ms::interp::PlotSeries, plot));
    if (plot.kind == ms::interp::PlotSeries::Kind::Surface3D) {
        plot_3d_->set_surface(plot);
        plot_stack_->setCurrentWidget(plot_3d_);
        return;
    }

    plot_2d_->set_plot(plot);
    plot_stack_->setCurrentWidget(plot_2d_);
}

void MainWindow::export_plot_png() {
    bool has_plot = false;
    QMetaObject::invokeMethod(repl_worker_, "hasPlot", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(bool, has_plot));
    if (!has_plot) {
        append_output("error: no plot to export\n");
        return;
    }

    const QString path =
        QFileDialog::getSaveFileName(this, "Export Plot", {}, "PNG Image (*.png)");
    if (path.isEmpty()) {
        return;
    }

    ms::interp::PlotSeries plot;
    QMetaObject::invokeMethod(repl_worker_, "plot", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(ms::interp::PlotSeries, plot));
    const bool ok = plot.kind == ms::interp::PlotSeries::Kind::Surface3D
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

void MainWindow::start_eval(const QString& line) {
    repl_busy_ = true;
    run_->setEnabled(false);
    run_script_->setEnabled(false);
    input_->setEnabled(false);
    append_output("ms> " + line + "\n");
    QMetaObject::invokeMethod(repl_worker_, "evaluate", Qt::QueuedConnection, Q_ARG(QString, line));
}

void MainWindow::finish_repl_op() {
    refresh_variables();
    refresh_plot();

    if (!script_queue_.isEmpty()) {
        start_eval(script_queue_.takeFirst());
        return;
    }

    repl_busy_ = false;
    run_->setEnabled(true);
    run_script_->setEnabled(true);
    input_->setEnabled(true);
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
    if (obj != input_ || event->type() != QEvent::KeyPress) {
        return QMainWindow::eventFilter(obj, event);
    }
    auto* key = static_cast<QKeyEvent*>(event);
    if (key->key() == Qt::Key_Up) {
        if (repl_history_.isEmpty()) {
            return true;
        }
        if (repl_history_pos_ < 0) {
            repl_history_draft_ = input_->text();
            repl_history_pos_ = repl_history_.size() - 1;
        } else if (repl_history_pos_ > 0) {
            --repl_history_pos_;
        }
        input_->setText(repl_history_.at(repl_history_pos_));
        return true;
    }
    if (key->key() == Qt::Key_Down) {
        if (repl_history_.isEmpty() || repl_history_pos_ < 0) {
            return true;
        }
        if (repl_history_pos_ + 1 >= repl_history_.size()) {
            repl_history_pos_ = -1;
            input_->setText(repl_history_draft_);
        } else {
            ++repl_history_pos_;
            input_->setText(repl_history_.at(repl_history_pos_));
        }
        return true;
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::on_submit() {
    if (repl_busy_) {
        return;
    }
    const QString line = input_->text().trimmed();
    if (line.isEmpty()) {
        return;
    }
    if (repl_history_.isEmpty() || repl_history_.last() != line) {
        repl_history_.append(line);
    }
    repl_history_pos_ = -1;
    repl_history_draft_.clear();
    input_->clear();
    script_queue_.clear();
    start_eval(line);
}

void MainWindow::on_run_script() {
    if (repl_busy_) {
        return;
    }
    script_queue_.clear();
    for (const QString& line : editor_->toPlainText().split('\n', Qt::SkipEmptyParts)) {
        script_queue_.append(line.trimmed());
    }
    script_queue_.removeAll(QString{});
    if (script_queue_.isEmpty()) {
        return;
    }
    start_eval(script_queue_.takeFirst());
}

void MainWindow::on_repl_finished(const QString& output) {
    if (!output.isEmpty()) {
        append_output(output);
        if (!output.endsWith('\n')) {
            append_output("\n");
        }
    }
    finish_repl_op();
}

void MainWindow::on_repl_error(const QString& message) {
    append_output("error: " + message + "\n");
    finish_repl_op();
}
