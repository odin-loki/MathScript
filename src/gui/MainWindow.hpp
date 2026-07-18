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
class LineNumberArea;
class QCloseEvent;
class QDockWidget;
class QMenu;
class QAction;
class QLabel;
class QTreeView;
class QFileSystemModel;
class ReplWorker;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void on_submit();
    void on_run_script();
    void on_repl_finished(const QString& output);
    void on_repl_error(const QString& message);
    void on_repl_cancelled();
    void on_stop();
    void on_file_activated(const QModelIndex& index);
    void on_variable_double_clicked(QListWidgetItem* item);
    void refresh_status();
    void update_cursor_status();

private:
    void append_output(const QString& text, bool is_error = false);
    void refresh_variables();
    void refresh_plot();
    void export_plot_png();
    void export_command_history();
    void export_last_result_latex();
    void setup_menus();
    void apply_dark_theme();
    void apply_light_theme();
    void restore_layout();
    void save_layout();
    void start_eval(const QString& line);
    void finish_repl_op();
    void clear_output();
    void show_about();
    void open_script_file(const QString& path);
    void add_recent_file(const QString& path);
    void refresh_recent_menu();
    void adjust_font_size(int delta);
    void show_welcome_banner();
    void find_in_output();
    void find_next_in_output();
    void find_prev_in_output();
    void find_in_script();
    void find_next_in_script();
    void find_prev_in_script();
    void replace_in_script();
    void replace_next_in_script();
    void go_to_line();
    void duplicate_line();
    void toggle_comment();
    void indent_lines();
    void unindent_lines();
    void set_plot_panel_visible(bool visible);
    void set_dark_theme(bool dark);
    void set_word_wrap(bool wrap);

    QSplitter* main_splitter_ = nullptr;
    QPlainTextEdit* output_ = nullptr;
    QPlainTextEdit* editor_ = nullptr;
    LineNumberArea* line_number_area_ = nullptr;
    QLineEdit* input_ = nullptr;
    QPushButton* run_ = nullptr;
    QPushButton* stop_ = nullptr;
    QPushButton* run_script_ = nullptr;
    QPushButton* clear_output_ = nullptr;
    QMenu* recent_menu_ = nullptr;
    QListWidget* variables_ = nullptr;
    QStackedWidget* plot_stack_ = nullptr;
    PlotWidget* plot_2d_ = nullptr;
    PlotSurfWidget* plot_3d_ = nullptr;
    QDockWidget* files_dock_ = nullptr;
    QTreeView* file_tree_ = nullptr;
    QFileSystemModel* file_model_ = nullptr;
    QTimer* status_timer_ = nullptr;
    QLabel* status_runtime_label_ = nullptr;
    QLabel* status_cursor_label_ = nullptr;
    QThread* repl_thread_ = nullptr;
    ReplWorker* repl_worker_ = nullptr;
    bool repl_busy_ = false;
    QStringList script_queue_;
    QStringList repl_history_;
    int repl_history_pos_ = -1;
    QString repl_history_draft_;
    QStringList recent_files_;
    int mono_font_size_ = 11;
    QString find_output_text_;
    QString find_script_text_;
    QString replace_script_text_;
    QString last_result_;
    QAction* show_plot_panel_action_ = nullptr;
    QAction* dark_theme_action_ = nullptr;
    QAction* word_wrap_action_ = nullptr;
    bool dark_theme_ = true;
    bool word_wrap_ = false;
};
