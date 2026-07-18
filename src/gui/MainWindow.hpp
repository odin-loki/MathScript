#pragma once

#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>
#include <QThread>
#include <QTimer>

class PlotWidget;
class PlotSurfWidget;
class QDockWidget;
class QTreeView;
class QFileSystemModel;
class ReplWorker;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void on_submit();
    void on_run_script();
    void on_repl_finished(const QString& output);
    void on_repl_error(const QString& message);
    void on_file_activated(const QModelIndex& index);
    void on_variable_double_clicked(QListWidgetItem* item);
    void refresh_status();

private:
    void append_output(const QString& text);
    void refresh_variables();
    void refresh_plot();
    void export_plot_png();
    void setup_menus();
    void apply_dark_theme();
    void start_eval(const QString& line);
    void finish_repl_op();

    QSplitter* main_splitter_ = nullptr;
    QPlainTextEdit* output_ = nullptr;
    QPlainTextEdit* editor_ = nullptr;
    QLineEdit* input_ = nullptr;
    QPushButton* run_ = nullptr;
    QPushButton* run_script_ = nullptr;
    QListWidget* variables_ = nullptr;
    QStackedWidget* plot_stack_ = nullptr;
    PlotWidget* plot_2d_ = nullptr;
    PlotSurfWidget* plot_3d_ = nullptr;
    QDockWidget* files_dock_ = nullptr;
    QTreeView* file_tree_ = nullptr;
    QFileSystemModel* file_model_ = nullptr;
    QTimer* status_timer_ = nullptr;
    QThread* repl_thread_ = nullptr;
    ReplWorker* repl_worker_ = nullptr;
    bool repl_busy_ = false;
    QStringList script_queue_;
    QStringList repl_history_;
    int repl_history_pos_ = -1;
    QString repl_history_draft_;
};
