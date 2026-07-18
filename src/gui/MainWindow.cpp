#include "gui/MainWindow.hpp"
#include "gui/LineNumberArea.hpp"
#include "gui/PlotSurfWidget.hpp"
#include "gui/PlotWidget.hpp"
#include "gui/ReplCompleter.hpp"
#include "gui/ReplWorker.hpp"
#include "gui/ScriptHighlighter.hpp"

#include <QApplication>
#include <QAction>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QFont>
#include <QKeyEvent>
#include <QMessageBox>
#include <QShortcut>
#include <QColor>
#include <QSettings>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QInputDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenuBar>
#include <QMetaType>
#include <QPalette>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextDocument>
#include <QTextEdit>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidget>

#include "ms/cuda/nvml.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/simd/simd.hpp"

#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace {

constexpr int kVarNameRole = Qt::UserRole;
constexpr int kVarKindRole = Qt::UserRole + 1;
constexpr char kVarKindMatrix[] = "matrix";
constexpr char kVarKindScalar[] = "scalar";
constexpr int kMaxRecentFiles = 8;
constexpr int kDefaultMonoFontSize = 11;

bool replace_one_in_editor(QPlainTextEdit* editor, const QString& find_text, const QString& replace_text) {
    if (editor == nullptr || find_text.isEmpty()) {
        return false;
    }

    QTextCursor cursor = editor->textCursor();
    if (cursor.hasSelection() && cursor.selectedText() == find_text) {
        cursor.insertText(replace_text);
        return true;
    }

    if (cursor.hasSelection()) {
        cursor.setPosition(cursor.selectionEnd());
        editor->setTextCursor(cursor);
    }

    if (editor->find(find_text)) {
        cursor = editor->textCursor();
        cursor.insertText(replace_text);
        return true;
    }

    editor->moveCursor(QTextCursor::Start);
    if (editor->find(find_text)) {
        cursor = editor->textCursor();
        cursor.insertText(replace_text);
        return true;
    }

    return false;
}

int replace_all_in_editor(QPlainTextEdit* editor, const QString& find_text, const QString& replace_text) {
    if (editor == nullptr || find_text.isEmpty()) {
        return 0;
    }

    editor->moveCursor(QTextCursor::Start);
    int count = 0;
    while (editor->find(find_text)) {
        QTextCursor cursor = editor->textCursor();
        cursor.insertText(replace_text);
        ++count;
    }
    return count;
}

void duplicate_lines_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    if (cursor.hasSelection()) {
        const int anchor = cursor.anchor();
        const int position = cursor.position();
        const int anchor_block = editor->document()->findBlock(anchor).blockNumber();
        const int position_block = editor->document()->findBlock(position).blockNumber();
        start_block = std::min(anchor_block, position_block);
        end_block = std::max(anchor_block, position_block);

        const int end_pos = std::max(anchor, position);
        QTextCursor end_cursor = cursor;
        end_cursor.setPosition(end_pos, QTextCursor::MoveAnchor);
        if (end_cursor.positionInBlock() == 0 && end_block > start_block) {
            end_block--;
        }
    } else {
        start_block = end_block = cursor.blockNumber();
    }

    QString duplicated;
    for (int block = start_block; block <= end_block; ++block) {
        if (block > start_block) {
            duplicated += '\n';
        }
        duplicated += editor->document()->findBlockByNumber(block).text();
    }

    cursor.setPosition(editor->document()->findBlockByNumber(end_block).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::MoveAnchor);
    cursor.insertText('\n' + duplicated);

    const int duplicate_start_block = end_block + 1;
    const int duplicate_end_block = end_block + (end_block - start_block + 1);
    cursor.setPosition(editor->document()->findBlockByNumber(duplicate_start_block).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    for (int block = duplicate_start_block + 1; block <= duplicate_end_block; ++block) {
        cursor.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor);
    }
    editor->setTextCursor(cursor);

    cursor.endEditBlock();
}

void get_selected_block_range(QPlainTextEdit* editor, int& start_block, int& end_block);

void select_block_range_in_editor(QPlainTextEdit* editor, int start_block, int end_block) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.setPosition(editor->document()->findBlockByNumber(start_block).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    for (int block = start_block + 1; block <= end_block; ++block) {
        cursor.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor);
    }
    editor->setTextCursor(cursor);
}

void move_lines_up_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    get_selected_block_range(editor, start_block, end_block);
    if (start_block == 0) {
        cursor.endEditBlock();
        return;
    }

    const QString above_line = editor->document()->findBlockByNumber(start_block - 1).text();
    QStringList selected_lines;
    for (int block = start_block; block <= end_block; ++block) {
        selected_lines.append(editor->document()->findBlockByNumber(block).text());
    }

    cursor.setPosition(editor->document()->findBlockByNumber(start_block - 1).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    for (int block = start_block; block <= end_block; ++block) {
        cursor.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor);
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    }
    cursor.insertText(selected_lines.join('\n') + '\n' + above_line);

    select_block_range_in_editor(editor, start_block - 1, end_block - 1);
    cursor.endEditBlock();
}

void move_lines_down_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    get_selected_block_range(editor, start_block, end_block);
    if (end_block >= editor->document()->blockCount() - 1) {
        cursor.endEditBlock();
        return;
    }

    const QString below_line = editor->document()->findBlockByNumber(end_block + 1).text();
    QStringList selected_lines;
    for (int block = start_block; block <= end_block; ++block) {
        selected_lines.append(editor->document()->findBlockByNumber(block).text());
    }

    cursor.setPosition(editor->document()->findBlockByNumber(start_block).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    for (int block = start_block + 1; block <= end_block + 1; ++block) {
        cursor.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor);
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    }
    cursor.insertText(below_line + '\n' + selected_lines.join('\n'));

    select_block_range_in_editor(editor, start_block + 1, end_block + 1);
    cursor.endEditBlock();
}

void delete_lines_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    get_selected_block_range(editor, start_block, end_block);

    const int last_block = editor->document()->blockCount() - 1;
    if (end_block >= last_block) {
        if (start_block == 0) {
            cursor.select(QTextCursor::Document);
            cursor.removeSelectedText();
            editor->setTextCursor(cursor);
            cursor.endEditBlock();
            return;
        }
        cursor.setPosition(editor->document()->findBlockByNumber(start_block - 1).position());
        cursor.movePosition(QTextCursor::EndOfBlock);
        cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
        cursor.removeSelectedText();
        cursor.setPosition(editor->document()->findBlockByNumber(start_block - 1).position());
        cursor.movePosition(QTextCursor::EndOfBlock);
    } else {
        cursor.setPosition(editor->document()->findBlockByNumber(start_block).position());
        const int end_pos = editor->document()->findBlockByNumber(end_block + 1).position();
        cursor.setPosition(end_pos, QTextCursor::KeepAnchor);
        cursor.removeSelectedText();
        cursor.setPosition(editor->document()->findBlockByNumber(start_block).position());
    }

    editor->setTextCursor(cursor);
    cursor.endEditBlock();
}

bool line_has_line_comment(const QString& line) {
    int i = 0;
    while (i < line.size() && line[i].isSpace()) {
        ++i;
    }
    return line.mid(i).startsWith("//");
}

QString comment_line(const QString& line) {
    int i = 0;
    while (i < line.size() && line[i].isSpace()) {
        ++i;
    }
    return line.left(i) + "//" + line.mid(i);
}

QString uncomment_line(const QString& line) {
    int i = 0;
    while (i < line.size() && line[i].isSpace()) {
        ++i;
    }
    const QString prefix = line.left(i);
    QString rest = line.mid(i);
    if (!rest.startsWith("//")) {
        return line;
    }
    rest = rest.mid(2);
    if (!rest.isEmpty() && rest[0].isSpace()) {
        rest = rest.mid(1);
    }
    return prefix + rest;
}

void toggle_comments_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    if (cursor.hasSelection()) {
        const int anchor = cursor.anchor();
        const int position = cursor.position();
        const int anchor_block = editor->document()->findBlock(anchor).blockNumber();
        const int position_block = editor->document()->findBlock(position).blockNumber();
        start_block = std::min(anchor_block, position_block);
        end_block = std::max(anchor_block, position_block);

        const int end_pos = std::max(anchor, position);
        QTextCursor end_cursor = cursor;
        end_cursor.setPosition(end_pos, QTextCursor::MoveAnchor);
        if (end_cursor.positionInBlock() == 0 && end_block > start_block) {
            end_block--;
        }
    } else {
        start_block = end_block = cursor.blockNumber();
    }

    bool all_commented = true;
    for (int block = start_block; block <= end_block; ++block) {
        if (!line_has_line_comment(editor->document()->findBlockByNumber(block).text())) {
            all_commented = false;
            break;
        }
    }

    for (int block = start_block; block <= end_block; ++block) {
        const QTextBlock text_block = editor->document()->findBlockByNumber(block);
        const QString original = text_block.text();
        const QString updated = all_commented ? uncomment_line(original) : comment_line(original);
        if (updated == original) {
            continue;
        }

        QTextCursor block_cursor(text_block);
        block_cursor.movePosition(QTextCursor::StartOfBlock);
        block_cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
        block_cursor.insertText(updated);
    }

    cursor.setPosition(editor->document()->findBlockByNumber(start_block).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::MoveAnchor);
    for (int block = start_block + 1; block <= end_block; ++block) {
        cursor.movePosition(QTextCursor::Down, QTextCursor::MoveAnchor);
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::MoveAnchor);
    }
    editor->setTextCursor(cursor);

    cursor.endEditBlock();
}

constexpr char kIndentText[] = "    ";

void get_selected_block_range(QPlainTextEdit* editor, int& start_block, int& end_block) {
    start_block = 0;
    end_block = 0;
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    if (cursor.hasSelection()) {
        const int anchor = cursor.anchor();
        const int position = cursor.position();
        const int anchor_block = editor->document()->findBlock(anchor).blockNumber();
        const int position_block = editor->document()->findBlock(position).blockNumber();
        start_block = std::min(anchor_block, position_block);
        end_block = std::max(anchor_block, position_block);

        const int end_pos = std::max(anchor, position);
        QTextCursor end_cursor = cursor;
        end_cursor.setPosition(end_pos, QTextCursor::MoveAnchor);
        if (end_cursor.positionInBlock() == 0 && end_block > start_block) {
            end_block--;
        }
    } else {
        start_block = end_block = cursor.blockNumber();
    }
}

void restore_block_cursor(QPlainTextEdit* editor, int start_block, int end_block) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.setPosition(editor->document()->findBlockByNumber(start_block).position());
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::MoveAnchor);
    for (int block = start_block + 1; block <= end_block; ++block) {
        cursor.movePosition(QTextCursor::Down, QTextCursor::MoveAnchor);
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::MoveAnchor);
    }
    editor->setTextCursor(cursor);
}

int trim_trailing_whitespace_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return 0;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    get_selected_block_range(editor, start_block, end_block);

    int trimmed_count = 0;
    for (int block_num = start_block; block_num <= end_block; ++block_num) {
        const QTextBlock text_block = editor->document()->findBlockByNumber(block_num);
        const QString original = text_block.text();
        int end = original.size();
        while (end > 0 && (original[end - 1] == QLatin1Char(' ') || original[end - 1] == QLatin1Char('\t'))) {
            --end;
        }
        if (end == original.size()) {
            continue;
        }

        const QString updated = original.left(end);
        QTextCursor block_cursor(text_block);
        block_cursor.movePosition(QTextCursor::StartOfBlock);
        block_cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
        block_cursor.insertText(updated);
        ++trimmed_count;
    }

    restore_block_cursor(editor, start_block, end_block);
    cursor.endEditBlock();
    return trimmed_count;
}

int remove_blank_lines_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return 0;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = editor->document()->blockCount() - 1;
    if (cursor.hasSelection()) {
        get_selected_block_range(editor, start_block, end_block);
    }

    int removed_count = 0;
    for (int block_num = end_block; block_num >= start_block; --block_num) {
        const QTextBlock text_block = editor->document()->findBlockByNumber(block_num);
        if (!text_block.text().trimmed().isEmpty()) {
            continue;
        }

        const int last_block = editor->document()->blockCount() - 1;
        if (block_num >= last_block) {
            if (block_num == 0) {
                cursor.select(QTextCursor::Document);
                cursor.removeSelectedText();
            } else {
                cursor.setPosition(editor->document()->findBlockByNumber(block_num - 1).position());
                cursor.movePosition(QTextCursor::EndOfBlock);
                cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
                cursor.removeSelectedText();
            }
        } else {
            cursor.setPosition(editor->document()->findBlockByNumber(block_num).position());
            const int end_pos = editor->document()->findBlockByNumber(block_num + 1).position();
            cursor.setPosition(end_pos, QTextCursor::KeepAnchor);
            cursor.removeSelectedText();
        }
        ++removed_count;
    }

    if (removed_count > 0) {
        const int new_last = editor->document()->blockCount() - 1;
        const int target_block = std::min(start_block, new_last);
        cursor.setPosition(editor->document()->findBlockByNumber(target_block).position());
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::MoveAnchor);
        editor->setTextCursor(cursor);
    }

    cursor.endEditBlock();
    return removed_count;
}

void indent_lines_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    get_selected_block_range(editor, start_block, end_block);

    for (int block = start_block; block <= end_block; ++block) {
        const QTextBlock text_block = editor->document()->findBlockByNumber(block);
        QTextCursor block_cursor(text_block);
        block_cursor.movePosition(QTextCursor::StartOfBlock);
        block_cursor.insertText(QString::fromLatin1(kIndentText));
    }

    restore_block_cursor(editor, start_block, end_block);
    cursor.endEditBlock();
}

void unindent_lines_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return;
    }

    QTextCursor cursor = editor->textCursor();
    cursor.beginEditBlock();

    int start_block = 0;
    int end_block = 0;
    get_selected_block_range(editor, start_block, end_block);

    for (int block = start_block; block <= end_block; ++block) {
        const QTextBlock text_block = editor->document()->findBlockByNumber(block);
        const QString line = text_block.text();
        int remove = 0;
        if (!line.isEmpty() && line[0] == QLatin1Char('\t')) {
            remove = 1;
        } else {
            while (remove < 4 && remove < line.size() && line[remove] == QLatin1Char(' ')) {
                ++remove;
            }
        }
        if (remove == 0) {
            continue;
        }

        QTextCursor block_cursor(text_block);
        block_cursor.movePosition(QTextCursor::StartOfBlock);
        block_cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, remove);
        block_cursor.removeSelectedText();
    }

    restore_block_cursor(editor, start_block, end_block);
    cursor.endEditBlock();
}

bool is_open_brace(QChar ch) {
    return ch == QLatin1Char('(') || ch == QLatin1Char('[') || ch == QLatin1Char('{');
}

bool is_close_brace(QChar ch) {
    return ch == QLatin1Char(')') || ch == QLatin1Char(']') || ch == QLatin1Char('}');
}

QChar matching_brace_pair(QChar ch) {
    switch (ch.unicode()) {
    case '(':
        return QLatin1Char(')');
    case ')':
        return QLatin1Char('(');
    case '[':
        return QLatin1Char(']');
    case ']':
        return QLatin1Char('[');
    case '{':
        return QLatin1Char('}');
    case '}':
        return QLatin1Char('{');
    default:
        return QChar();
    }
}

bool go_to_matching_brace_in_editor(QPlainTextEdit* editor) {
    if (editor == nullptr) {
        return false;
    }

    const QString text = editor->toPlainText();
    if (text.isEmpty()) {
        return false;
    }

    QTextCursor cursor = editor->textCursor();
    int pos = cursor.position();

    // Prefer brace under cursor; otherwise try the character immediately before.
    QChar brace;
    if (pos < text.size() && (is_open_brace(text[pos]) || is_close_brace(text[pos]))) {
        brace = text[pos];
    } else if (pos > 0 && (is_open_brace(text[pos - 1]) || is_close_brace(text[pos - 1]))) {
        --pos;
        brace = text[pos];
    } else {
        return false;
    }

    const QChar match = matching_brace_pair(brace);
    if (match.isNull()) {
        return false;
    }

    int depth = 0;
    if (is_open_brace(brace)) {
        for (int i = pos; i < text.size(); ++i) {
            const QChar ch = text[i];
            if (ch == brace) {
                ++depth;
            } else if (ch == match) {
                --depth;
                if (depth == 0) {
                    cursor.setPosition(i);
                    editor->setTextCursor(cursor);
                    editor->centerCursor();
                    return true;
                }
            }
        }
    } else {
        for (int i = pos; i >= 0; --i) {
            const QChar ch = text[i];
            if (ch == brace) {
                ++depth;
            } else if (ch == match) {
                --depth;
                if (depth == 0) {
                    cursor.setPosition(i);
                    editor->setTextCursor(cursor);
                    editor->centerCursor();
                    return true;
                }
            }
        }
    }

    return false;
}

QTextEdit::ExtraSelection make_add_selection_extra(QPlainTextEdit* editor, const QTextCursor& cursor) {
    QTextEdit::ExtraSelection extra;
    extra.cursor = cursor;
    extra.format.setBackground(editor->palette().color(QPalette::Highlight));
    extra.format.setForeground(editor->palette().color(QPalette::HighlightedText));
    return extra;
}

bool range_overlaps_existing(const QList<QTextEdit::ExtraSelection>& extras, int start, int end) {
    for (const auto& extra : extras) {
        const int a = extra.cursor.selectionStart();
        const int b = extra.cursor.selectionEnd();
        if (start == a && end == b) {
            return true;
        }
    }
    return false;
}

// Select word under cursor, or find the next match of the current selection and keep prior
// matches highlighted via ExtraSelection (QPlainTextEdit has a single primary cursor).
bool add_selection_for_next_occurrence_in_editor(QPlainTextEdit* editor, QString& needle,
                                                 QList<QTextEdit::ExtraSelection>& extras) {
    if (editor == nullptr) {
        return false;
    }

    QTextCursor cursor = editor->textCursor();
    if (!cursor.hasSelection()) {
        cursor.select(QTextCursor::WordUnderCursor);
        if (!cursor.hasSelection()) {
            extras.clear();
            needle.clear();
            editor->setExtraSelections({});
            return false;
        }
        needle = cursor.selectedText();
        extras.clear();
        editor->setExtraSelections({});
        editor->setTextCursor(cursor);
        editor->centerCursor();
        return true;
    }

    const QString selected = cursor.selectedText();
    if (selected.isEmpty()) {
        return false;
    }

    if (selected != needle) {
        needle = selected;
        extras.clear();
    }

    const int current_start = cursor.selectionStart();
    const int current_end = cursor.selectionEnd();
    bool demoted_current = false;
    if (!range_overlaps_existing(extras, current_start, current_end)) {
        QTextCursor keep = cursor;
        keep.setPosition(current_start);
        keep.setPosition(current_end, QTextCursor::KeepAnchor);
        extras.append(make_add_selection_extra(editor, keep));
        demoted_current = true;
    }

    QTextDocument* doc = editor->document();
    int search_from = current_end;
    for (int pass = 0; pass < 2; ++pass) {
        QTextCursor found = doc->find(needle, search_from);
        while (!found.isNull()) {
            const int start = found.selectionStart();
            const int end = found.selectionEnd();
            if (!range_overlaps_existing(extras, start, end)) {
                editor->setExtraSelections(extras);
                editor->setTextCursor(found);
                editor->centerCursor();
                return true;
            }
            found = doc->find(needle, end);
        }
        search_from = 0;
    }

    // No new match: restore extras without demoting the current primary selection.
    if (demoted_current) {
        extras.removeLast();
    }
    editor->setExtraSelections(extras);
    return false;
}

constexpr size_t kListPreviewMaxRows = 3;
constexpr size_t kListPreviewMaxCols = 4;
constexpr size_t kTooltipMaxRows = 6;
constexpr size_t kTooltipMaxCols = 6;
constexpr size_t kDumpMaxRows = 8;
constexpr size_t kDumpMaxCols = 8;

std::string format_matrix_cell(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << value;
    std::string text = out.str();
    while (!text.empty() && text.back() == '0') {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.') {
        text.pop_back();
    }
    return text.empty() ? "0" : text;
}

void append_matrix_cells(std::ostringstream& out, const ms::Matrix<double>& matrix, size_t max_rows,
                         size_t max_cols) {
    const size_t rows = std::min(matrix.rows(), max_rows);
    const size_t cols = std::min(matrix.cols(), max_cols);
    for (size_t i = 0; i < rows; ++i) {
        if (i > 0) {
            out << "; ";
        }
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << format_matrix_cell(matrix(i, j));
        }
        if (matrix.cols() > cols) {
            out << ", ...";
        }
    }
    if (matrix.rows() > rows) {
        out << "; ...";
    }
}

QString matrix_inline_preview(const ms::Matrix<double>& matrix, size_t max_rows, size_t max_cols) {
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        return QStringLiteral("[]");
    }
    std::ostringstream out;
    out << "[";
    append_matrix_cells(out, matrix, max_rows, max_cols);
    out << "]";
    return QString::fromStdString(out.str());
}

QString matrix_tooltip_text(const std::string& name, const ms::Matrix<double>& matrix) {
    QString text = QString("%1 (%2x%3)\n")
                       .arg(QString::fromStdString(name))
                       .arg(static_cast<int>(matrix.rows()))
                       .arg(static_cast<int>(matrix.cols()));
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        return text.trimmed();
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    const size_t rows = std::min(matrix.rows(), kTooltipMaxRows);
    const size_t cols = std::min(matrix.cols(), kTooltipMaxCols);
    for (size_t i = 0; i < rows; ++i) {
        out << "  [";
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << format_matrix_cell(matrix(i, j));
        }
        if (matrix.cols() > cols) {
            out << ", ...";
        }
        out << "]\n";
    }
    if (matrix.rows() > rows) {
        out << "  ...\n";
    }
    text += QString::fromStdString(out.str()).trimmed();
    return text;
}

QString matrix_dump_text(const std::string& name, const ms::Matrix<double>& matrix) {
    const bool truncated = matrix.rows() > kDumpMaxRows || matrix.cols() > kDumpMaxCols;
    QString header =
        truncated ? QString("%1 (%2x%3, showing %4x%5):\n")
                        .arg(QString::fromStdString(name))
                        .arg(static_cast<int>(matrix.rows()))
                        .arg(static_cast<int>(matrix.cols()))
                        .arg(static_cast<int>(std::min(matrix.rows(), kDumpMaxRows)))
                        .arg(static_cast<int>(std::min(matrix.cols(), kDumpMaxCols)))
                  : QString("%1 (%2x%3):\n")
                        .arg(QString::fromStdString(name))
                        .arg(static_cast<int>(matrix.rows()))
                        .arg(static_cast<int>(matrix.cols()));

    std::ostringstream out;
    if (truncated) {
        const size_t rows = std::min(matrix.rows(), kDumpMaxRows);
        const size_t cols = std::min(matrix.cols(), kDumpMaxCols);
        out << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < rows; ++i) {
            out << "  [";
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0) {
                    out << ", ";
                }
                out << matrix(i, j);
            }
            if (matrix.cols() > cols) {
                out << ", ...";
            }
            out << "]\n";
        }
        if (matrix.rows() > rows) {
            out << "  ...\n";
        }
    } else {
        ms::interp::print_matrix(out, matrix);
    }
    return header + QString::fromStdString(out.str());
}

bool is_small_matrix(const ms::Matrix<double>& matrix) {
    return matrix.rows() <= kListPreviewMaxRows && matrix.cols() <= kListPreviewMaxCols;
}

QString escape_latex_text(const QString& text) {
    QString out;
    out.reserve(text.size() + 8);
    for (const QChar ch : text) {
        switch (ch.unicode()) {
            case '\\':
                out += QStringLiteral("\\textbackslash{}");
                break;
            case '{':
                out += QStringLiteral("\\{");
                break;
            case '}':
                out += QStringLiteral("\\}");
                break;
            case '$':
                out += QStringLiteral("\\$");
                break;
            case '%':
                out += QStringLiteral("\\%");
                break;
            case '&':
                out += QStringLiteral("\\&");
                break;
            case '#':
                out += QStringLiteral("\\#");
                break;
            case '_':
                out += QStringLiteral("\\_");
                break;
            case '^':
                out += QStringLiteral("\\^{}");
                break;
            case '~':
                out += QStringLiteral("\\~{}");
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}

bool parse_scalar_result(const QString& text) {
    if (text.isEmpty()) {
        return false;
    }
    bool ok = false;
    const double parsed = text.toDouble(&ok);
    return ok && std::isfinite(parsed);
}

QString matrix_row_to_latex(const QString& row_text) {
    QStringList cells;
    for (const QString& cell : row_text.split(',', Qt::SkipEmptyParts)) {
        const QString trimmed = cell.trimmed();
        if (trimmed == QStringLiteral("...")) {
            continue;
        }
        cells.append(escape_latex_text(trimmed));
    }
    return cells.join(QStringLiteral(" & "));
}

QString result_to_latex_body(const QString& result) {
    const QString text = result.trimmed();
    if (text.isEmpty()) {
        return QStringLiteral("\\[\\]");
    }

    if (parse_scalar_result(text)) {
        return QStringLiteral("\\[%1\\]").arg(escape_latex_text(text));
    }

    if (text.startsWith('[') && text.endsWith(']')) {
        const QString inner = text.mid(1, text.size() - 2).trimmed();
        const QStringList rows = inner.split(';', Qt::SkipEmptyParts);
        if (!rows.isEmpty()) {
            QStringList latex_rows;
            for (const QString& row : rows) {
                const QString trimmed = row.trimmed();
                if (trimmed == QStringLiteral("...")) {
                    continue;
                }
                latex_rows.append(matrix_row_to_latex(trimmed));
            }
            if (!latex_rows.isEmpty()) {
                return QStringLiteral("\\[\n\\begin{bmatrix}\n%1\n\\end{bmatrix}\n\\]")
                    .arg(latex_rows.join(QStringLiteral(" \\\\\n")));
            }
        }
    }

    return QStringLiteral("\\begin{verbatim}\n%1\n\\end{verbatim}").arg(escape_latex_text(text));
}

QString wrap_latex_document(const QString& body) {
    return QStringLiteral("\\documentclass{article}\n"
                          "\\usepackage{amsmath}\n"
                          "\\begin{document}\n\n"
                          "%1\n\n"
                          "\\end{document}\n")
        .arg(body);
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    qRegisterMetaType<ms::interp::SessionState>("ms::interp::SessionState");
    qRegisterMetaType<ms::interp::PlotSeries>("ms::interp::PlotSeries");

    repl_thread_ = new QThread(this);
    repl_worker_ = new ReplWorker();
    repl_worker_->moveToThread(repl_thread_);
    repl_thread_->start();

    connect(repl_worker_, &ReplWorker::finished, this, &MainWindow::on_repl_finished);
    connect(repl_worker_, &ReplWorker::error, this, &MainWindow::on_repl_error);
    connect(repl_worker_, &ReplWorker::cancelled, this, &MainWindow::on_repl_cancelled);

    setWindowTitle("MathScript IDE");
    resize(1200, 720);

    QSettings startup_settings;
    dark_theme_ = startup_settings.value("gui/dark_theme", true).toBool();
    word_wrap_ = startup_settings.value("gui/word_wrap", false).toBool();
    if (dark_theme_) {
        apply_dark_theme();
    } else {
        apply_light_theme();
    }
    setup_menus();

    main_splitter_ = new QSplitter(Qt::Horizontal, this);

    auto* center = new QWidget(main_splitter_);
    auto* center_layout = new QVBoxLayout(center);

    editor_ = new QPlainTextEdit(center);
    editor_->setPlaceholderText("Script editor (open a file from the browser)");
    editor_->setMinimumHeight(200);
    new ScriptHighlighter(editor_->document());
    line_number_area_ = new LineNumberArea(editor_);
    center_layout->addWidget(editor_);

    output_ = new QPlainTextEdit(center);
    output_->setReadOnly(true);
    output_->setPlaceholderText("MathScript output");
    center_layout->addWidget(output_, 2);
    set_word_wrap(word_wrap_);

    QSettings settings;
    mono_font_size_ = settings.value("gui/mono_font_size", kDefaultMonoFontSize).toInt();
    adjust_font_size(0);
    recent_files_ = settings.value("gui/recent_files").toStringList();

    plot_stack_ = new QStackedWidget(center);
    plot_2d_ = new PlotWidget(plot_stack_);
    plot_3d_ = new PlotSurfWidget(plot_stack_);
    plot_stack_->addWidget(plot_2d_);
    plot_stack_->addWidget(plot_3d_);
    center_layout->addWidget(plot_stack_);

    auto* row = new QHBoxLayout();
    input_ = new QLineEdit(center);
    input_->setPlaceholderText("REPL: help, plot([1,2,3,4]), save session.ms (Ctrl+Space to complete)");
    setup_repl_completer(input_);
    run_ = new QPushButton("Run", center);
    stop_ = new QPushButton("Stop", center);
    stop_->setEnabled(false);
    stop_->setStatusTip("Cancel the running REPL command or queued script lines");
    stop_->setToolTip("Stop (cooperative cancel)");
    run_script_ = new QPushButton("Run Script", center);
    clear_output_ = new QPushButton("Clear Output", center);
    clear_output_->setStatusTip("Clear the REPL output pane");
    row->addWidget(input_);
    row->addWidget(run_);
    row->addWidget(stop_);
    row->addWidget(run_script_);
    row->addWidget(clear_output_);
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
    status_runtime_label_ = new QLabel(this);
    status_runtime_label_->setMinimumWidth(320);
    statusBar()->addWidget(status_runtime_label_, 1);
    status_cursor_label_ = new QLabel(this);
    statusBar()->addPermanentWidget(status_cursor_label_);
    status_timer_->start(2000);
    refresh_status();
    update_cursor_status();
    connect(editor_, &QPlainTextEdit::cursorPositionChanged, this, &MainWindow::update_cursor_status);

    connect(run_, &QPushButton::clicked, this, &MainWindow::on_submit);
    auto* run_shortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Return), this);
    connect(run_shortcut, &QShortcut::activated, this, &MainWindow::on_submit);
    connect(stop_, &QPushButton::clicked, this, &MainWindow::on_stop);
    connect(run_script_, &QPushButton::clicked, this, &MainWindow::on_run_script);
    connect(clear_output_, &QPushButton::clicked, this, &MainWindow::clear_output);
    connect(input_, &QLineEdit::returnPressed, this, &MainWindow::on_submit);
    input_->installEventFilter(this);
    editor_->installEventFilter(this);
    connect(file_tree_, &QTreeView::activated, this, &MainWindow::on_file_activated);
    connect(variables_, &QListWidget::itemDoubleClicked, this, &MainWindow::on_variable_double_clicked);

    restore_layout();
    refresh_recent_menu();
    show_welcome_banner();
    refresh_variables();
}

MainWindow::~MainWindow() {
    repl_thread_->quit();
    repl_thread_->wait();
    delete repl_worker_;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    save_layout();
    QMainWindow::closeEvent(event);
}

void MainWindow::restore_layout() {
    QSettings settings;
    const QByteArray window_state = settings.value("mainwindow/state").toByteArray();
    if (!window_state.isEmpty()) {
        restoreState(window_state);
    }

    const QByteArray splitter_state = settings.value("mainwindow/splitter").toByteArray();
    if (!splitter_state.isEmpty() && main_splitter_ != nullptr) {
        main_splitter_->restoreState(splitter_state);
    }

    if (plot_stack_ != nullptr) {
        const bool plot_visible = settings.value("gui/plot_panel_visible", true).toBool();
        set_plot_panel_visible(plot_visible);
    }

    dark_theme_ = settings.value("gui/dark_theme", true).toBool();
    set_dark_theme(dark_theme_);

    word_wrap_ = settings.value("gui/word_wrap", false).toBool();
    set_word_wrap(word_wrap_);

    find_script_text_ = settings.value("gui/find_script_text").toString();
    replace_script_text_ = settings.value("gui/replace_script_text").toString();
}

void MainWindow::save_layout() {
    QSettings settings;
    settings.setValue("mainwindow/state", saveState());
    if (main_splitter_ != nullptr) {
        settings.setValue("mainwindow/splitter", main_splitter_->saveState());
    }
    settings.setValue("gui/mono_font_size", mono_font_size_);
    settings.setValue("gui/recent_files", recent_files_);
    if (plot_stack_ != nullptr) {
        settings.setValue("gui/plot_panel_visible", plot_stack_->isVisible());
    }
    settings.setValue("gui/dark_theme", dark_theme_);
    settings.setValue("gui/word_wrap", word_wrap_);
    settings.setValue("gui/find_script_text", find_script_text_);
    settings.setValue("gui/replace_script_text", replace_script_text_);
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

void MainWindow::apply_light_theme() {
    QPalette palette;
    palette.setColor(QPalette::Window, QColor(243, 243, 243));
    palette.setColor(QPalette::WindowText, QColor(30, 30, 30));
    palette.setColor(QPalette::Base, QColor(255, 255, 255));
    palette.setColor(QPalette::AlternateBase, QColor(240, 240, 240));
    palette.setColor(QPalette::Text, QColor(30, 30, 30));
    palette.setColor(QPalette::Button, QColor(225, 225, 225));
    palette.setColor(QPalette::ButtonText, QColor(30, 30, 30));
    palette.setColor(QPalette::Highlight, QColor(0, 102, 204));
    palette.setColor(QPalette::HighlightedText, Qt::white);
    qApp->setPalette(palette);
}

void MainWindow::set_dark_theme(bool dark) {
    dark_theme_ = dark;
    if (dark) {
        apply_dark_theme();
    } else {
        apply_light_theme();
    }
    if (dark_theme_action_ != nullptr && dark_theme_action_->isChecked() != dark) {
        dark_theme_action_->blockSignals(true);
        dark_theme_action_->setChecked(dark);
        dark_theme_action_->blockSignals(false);
    }
}

void MainWindow::set_word_wrap(bool wrap) {
    word_wrap_ = wrap;
    const QPlainTextEdit::LineWrapMode mode =
        wrap ? QPlainTextEdit::WidgetWidth : QPlainTextEdit::NoWrap;
    if (editor_ != nullptr) {
        editor_->setLineWrapMode(mode);
    }
    if (output_ != nullptr) {
        output_->setLineWrapMode(mode);
    }
    if (word_wrap_action_ != nullptr && word_wrap_action_->isChecked() != wrap) {
        word_wrap_action_->blockSignals(true);
        word_wrap_action_->setChecked(wrap);
        word_wrap_action_->blockSignals(false);
    }
}

void MainWindow::setup_menus() {
    auto* file_menu = menuBar()->addMenu("&File");
    auto* open_action = file_menu->addAction("Open...");
    recent_menu_ = file_menu->addMenu("Open &Recent");
    auto* save_session_action = file_menu->addAction("Save Session...");
    auto* load_session_action = file_menu->addAction("Load Session...");
    auto* export_history_action = file_menu->addAction("Export Command History...");
    auto* export_latex_action = file_menu->addAction("Export Last Result as LaTeX...");
    auto* export_plot_action = file_menu->addAction("Export Plot as PNG...");
    file_menu->addSeparator();
    auto* quit_action = file_menu->addAction("Quit");

    auto* edit_menu = menuBar()->addMenu("&Edit");
    auto* find_script_action = edit_menu->addAction("Find in Script");
    find_script_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_F));
    auto* find_next_script_action = edit_menu->addAction("Find Next in Script");
    find_next_script_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F3));
    auto* find_prev_script_action = edit_menu->addAction("Find Previous in Script");
    find_prev_script_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_F3));
    auto* replace_script_action = edit_menu->addAction("Replace in Script...");
    replace_script_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_H));
    auto* replace_next_script_action = edit_menu->addAction("Replace Next in Script");
    replace_next_script_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R));
    auto* replace_all_script_action = edit_menu->addAction("Replace All in Script");
    replace_all_script_action->setShortcut(QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_H));
    auto* go_to_line_action = edit_menu->addAction("Go to Line...");
    go_to_line_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_G));
    auto* go_to_matching_brace_action = edit_menu->addAction("Go to Matching Brace");
    go_to_matching_brace_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_BracketRight));
    auto* add_selection_action = edit_menu->addAction("Add Selection for Next Occurrence");
    add_selection_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_D));
    auto* indent_action = edit_menu->addAction("Indent");
    indent_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_BracketRight));
    auto* unindent_action = edit_menu->addAction("Unindent");
    unindent_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_BracketLeft));
    auto* toggle_comment_action = edit_menu->addAction("Toggle Comment");
    toggle_comment_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Slash));
    auto* duplicate_line_action = edit_menu->addAction("Duplicate Line");
    duplicate_line_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_D));
    auto* trim_trailing_whitespace_action = edit_menu->addAction("Trim Trailing Whitespace");
    trim_trailing_whitespace_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_W));
    auto* remove_blank_lines_action = edit_menu->addAction("Remove Blank Lines");
    remove_blank_lines_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Backspace));
    auto* delete_line_action = edit_menu->addAction("Delete Line");
    delete_line_action->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_K));
    auto* move_line_up_action = edit_menu->addAction("Move Line Up");
    move_line_up_action->setShortcut(QKeySequence(Qt::ALT | Qt::Key_Up));
    auto* move_line_down_action = edit_menu->addAction("Move Line Down");
    move_line_down_action->setShortcut(QKeySequence(Qt::ALT | Qt::Key_Down));
    auto* select_all_action = edit_menu->addAction("Select All");
    select_all_action->setShortcut(QKeySequence::SelectAll);
    auto* undo_action = edit_menu->addAction("Undo");
    undo_action->setShortcut(QKeySequence::Undo);
    auto* redo_action = edit_menu->addAction("Redo");
    redo_action->setShortcuts({QKeySequence::Redo, QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Z)});
    edit_menu->addSeparator();
    auto* find_output_action = edit_menu->addAction("Find in Output");
    find_output_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F));
    auto* find_next_output_action = edit_menu->addAction("Find Next in Output");
    find_next_output_action->setShortcut(QKeySequence(Qt::Key_F3));
    auto* find_prev_output_action = edit_menu->addAction("Find Previous in Output");
    find_prev_output_action->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_F3));

    auto* view_menu = menuBar()->addMenu("&View");
    dark_theme_action_ = view_menu->addAction("Dark Theme");
    dark_theme_action_->setCheckable(true);
    dark_theme_action_->setChecked(dark_theme_);
    word_wrap_action_ = view_menu->addAction("Word Wrap");
    word_wrap_action_->setCheckable(true);
    word_wrap_action_->setChecked(word_wrap_);
    view_menu->addSeparator();
    show_plot_panel_action_ = view_menu->addAction("Show Plot Panel");
    show_plot_panel_action_->setCheckable(true);
    show_plot_panel_action_->setChecked(true);
    view_menu->addSeparator();
    auto* clear_output_action = view_menu->addAction("Clear Output");
    clear_output_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_L));
    auto* font_larger_action = view_menu->addAction("Increase Font Size");
    font_larger_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Plus));
    auto* font_smaller_action = view_menu->addAction("Decrease Font Size");
    font_smaller_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Minus));
    auto* font_reset_action = view_menu->addAction("Reset Font Size");
    font_reset_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_0));

    auto* help_menu = menuBar()->addMenu("&Help");
    auto* about_action = help_menu->addAction("About MathScript...");

    connect(open_action, &QAction::triggered, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(this, "Open Script");
        if (path.isEmpty()) {
            return;
        }
        open_script_file(path);
    });
    connect(find_script_action, &QAction::triggered, this, &MainWindow::find_in_script);
    connect(find_next_script_action, &QAction::triggered, this, &MainWindow::find_next_in_script);
    connect(find_prev_script_action, &QAction::triggered, this, &MainWindow::find_prev_in_script);
    connect(replace_script_action, &QAction::triggered, this, &MainWindow::replace_in_script);
    connect(replace_next_script_action, &QAction::triggered, this, &MainWindow::replace_next_in_script);
    connect(replace_all_script_action, &QAction::triggered, this, &MainWindow::replace_all_in_script);
    connect(go_to_line_action, &QAction::triggered, this, &MainWindow::go_to_line);
    connect(go_to_matching_brace_action, &QAction::triggered, this, &MainWindow::go_to_matching_brace);
    connect(add_selection_action, &QAction::triggered, this, &MainWindow::add_selection_for_next_occurrence);
    connect(indent_action, &QAction::triggered, this, &MainWindow::indent_lines);
    connect(unindent_action, &QAction::triggered, this, &MainWindow::unindent_lines);
    connect(toggle_comment_action, &QAction::triggered, this, &MainWindow::toggle_comment);
    connect(duplicate_line_action, &QAction::triggered, this, &MainWindow::duplicate_line);
    connect(trim_trailing_whitespace_action, &QAction::triggered, this,
            &MainWindow::trim_trailing_whitespace);
    connect(remove_blank_lines_action, &QAction::triggered, this, &MainWindow::remove_blank_lines);
    connect(delete_line_action, &QAction::triggered, this, &MainWindow::delete_line);
    connect(move_line_up_action, &QAction::triggered, this, &MainWindow::move_line_up);
    connect(move_line_down_action, &QAction::triggered, this, &MainWindow::move_line_down);
    connect(select_all_action, &QAction::triggered, this, &MainWindow::select_all);
    connect(undo_action, &QAction::triggered, this, &MainWindow::undo_edit);
    connect(redo_action, &QAction::triggered, this, &MainWindow::redo_edit);
    connect(find_output_action, &QAction::triggered, this, &MainWindow::find_in_output);
    connect(find_next_output_action, &QAction::triggered, this, &MainWindow::find_next_in_output);
    connect(find_prev_output_action, &QAction::triggered, this, &MainWindow::find_prev_in_output);
    connect(dark_theme_action_, &QAction::toggled, this, &MainWindow::set_dark_theme);
    connect(word_wrap_action_, &QAction::toggled, this, &MainWindow::set_word_wrap);
    connect(show_plot_panel_action_, &QAction::toggled, this, &MainWindow::set_plot_panel_visible);
    connect(clear_output_action, &QAction::triggered, this, &MainWindow::clear_output);
    connect(font_larger_action, &QAction::triggered, this, [this]() { adjust_font_size(1); });
    connect(font_smaller_action, &QAction::triggered, this, [this]() { adjust_font_size(-1); });
    connect(font_reset_action, &QAction::triggered, this, [this]() {
        mono_font_size_ = kDefaultMonoFontSize;
        adjust_font_size(0);
    });
    connect(about_action, &QAction::triggered, this, &MainWindow::show_about);
    connect(save_session_action, &QAction::triggered, this, [this]() {
        if (repl_busy_) {
            append_output("error: wait for the current REPL command to finish\n", true);
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
            append_output("error: " + err + "\n", true);
        }
    });
    connect(load_session_action, &QAction::triggered, this, [this]() {
        if (repl_busy_) {
            append_output("error: wait for the current REPL command to finish\n", true);
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
            append_output("error: " + err + "\n", true);
        }
    });
    connect(export_plot_action, &QAction::triggered, this, &MainWindow::export_plot_png);
    connect(export_history_action, &QAction::triggered, this, &MainWindow::export_command_history);
    connect(export_latex_action, &QAction::triggered, this, &MainWindow::export_last_result_latex);
    connect(quit_action, &QAction::triggered, qApp, &QApplication::quit);
}

void MainWindow::refresh_status() {
    const auto topo = ms::detect_topology();
    static ms::distributed::MPIContext mpi = ms::distributed::init(0, nullptr);

    QString isa =
        QString::fromStdString(ms::simd::isa_summary(ms::simd::dispatch_info().isa)).trimmed();
    if (isa.isEmpty()) {
        isa = QStringLiteral("scalar");
    }

    QString gpu_part = QStringLiteral("GPU n/a");
    if (ms::has_cuda() && topo.total_gpus > 0) {
        const auto stats = ms::cuda::device_stats(0);
        const size_t free_bytes = ms::cuda::device_memory_free(0);
        const auto free_mib = static_cast<qulonglong>(free_bytes / (1024 * 1024));
        const auto total_mib =
            static_cast<qulonglong>(stats.memory_total_bytes / (1024 * 1024));
        const QString model =
            stats.name.empty() ? QString::fromStdString(ms::get_gpu_model(0))
                               : QString::fromStdString(stats.name);
        gpu_part = QString("%1 %2/%3 MiB").arg(model).arg(free_mib).arg(total_mib);
    }

    status_runtime_label_->setText(QString("SIMD %1 | %2 | MPI %3/%4")
                                       .arg(isa)
                                       .arg(gpu_part)
                                       .arg(ms::distributed::rank(mpi))
                                       .arg(ms::distributed::size(mpi)));
}

void MainWindow::update_cursor_status() {
    if (editor_ == nullptr || status_cursor_label_ == nullptr) {
        return;
    }
    const QTextCursor cursor = editor_->textCursor();
    status_cursor_label_->setText(
        QString("Ln %1, Col %2").arg(cursor.blockNumber() + 1).arg(cursor.columnNumber() + 1));
}

void MainWindow::clear_output() {
    output_->clear();
}

void MainWindow::find_in_output() {
    bool ok = false;
    const QString text =
        QInputDialog::getText(this, "Find in Output", "Find:", QLineEdit::Normal, find_output_text_, &ok);
    if (!ok || text.isEmpty()) {
        return;
    }
    find_output_text_ = text;
    output_->moveCursor(QTextCursor::Start);
    if (!output_->find(find_output_text_)) {
        statusBar()->showMessage("Find in Output: no matches", 3000);
    }
}

void MainWindow::find_next_in_output() {
    if (find_output_text_.isEmpty()) {
        find_in_output();
        return;
    }

    QTextCursor cursor = output_->textCursor();
    if (cursor.hasSelection()) {
        cursor.setPosition(cursor.selectionEnd());
        output_->setTextCursor(cursor);
    }

    if (!output_->find(find_output_text_)) {
        output_->moveCursor(QTextCursor::Start);
        if (!output_->find(find_output_text_)) {
            statusBar()->showMessage("Find in Output: no matches", 3000);
        }
    }
}

void MainWindow::find_prev_in_output() {
    if (find_output_text_.isEmpty()) {
        find_in_output();
        return;
    }

    QTextCursor cursor = output_->textCursor();
    if (cursor.hasSelection()) {
        cursor.setPosition(cursor.selectionStart());
        output_->setTextCursor(cursor);
    }

    if (!output_->find(find_output_text_, QTextDocument::FindBackward)) {
        output_->moveCursor(QTextCursor::End);
        if (!output_->find(find_output_text_, QTextDocument::FindBackward)) {
            statusBar()->showMessage("Find in Output: no matches", 3000);
        }
    }
}

void MainWindow::find_in_script() {
    editor_->setFocus();
    bool ok = false;
    const QString text =
        QInputDialog::getText(this, "Find in Script", "Find:", QLineEdit::Normal, find_script_text_, &ok);
    if (!ok || text.isEmpty()) {
        return;
    }
    find_script_text_ = text;
    editor_->moveCursor(QTextCursor::Start);
    if (!editor_->find(find_script_text_)) {
        statusBar()->showMessage("Find in Script: no matches", 3000);
    }
}

void MainWindow::find_next_in_script() {
    editor_->setFocus();
    if (find_script_text_.isEmpty()) {
        find_in_script();
        return;
    }

    QTextCursor cursor = editor_->textCursor();
    if (cursor.hasSelection()) {
        cursor.setPosition(cursor.selectionEnd());
        editor_->setTextCursor(cursor);
    }

    if (!editor_->find(find_script_text_)) {
        editor_->moveCursor(QTextCursor::Start);
        if (!editor_->find(find_script_text_)) {
            statusBar()->showMessage("Find in Script: no matches", 3000);
        }
    }
}

void MainWindow::find_prev_in_script() {
    editor_->setFocus();
    if (find_script_text_.isEmpty()) {
        find_in_script();
        return;
    }

    QTextCursor cursor = editor_->textCursor();
    if (cursor.hasSelection()) {
        cursor.setPosition(cursor.selectionStart());
        editor_->setTextCursor(cursor);
    }

    if (!editor_->find(find_script_text_, QTextDocument::FindBackward)) {
        editor_->moveCursor(QTextCursor::End);
        if (!editor_->find(find_script_text_, QTextDocument::FindBackward)) {
            statusBar()->showMessage("Find in Script: no matches", 3000);
        }
    }
}

void MainWindow::replace_in_script() {
    editor_->setFocus();

    QDialog dialog(this);
    dialog.setWindowTitle("Replace in Script");

    auto* find_edit = new QLineEdit(find_script_text_, &dialog);
    auto* replace_edit = new QLineEdit(replace_script_text_, &dialog);

    auto* form = new QFormLayout(&dialog);
    form->addRow("Find:", find_edit);
    form->addRow("Replace with:", replace_edit);

    auto* buttons = new QDialogButtonBox(&dialog);
    auto* replace_button = buttons->addButton("Replace", QDialogButtonBox::ActionRole);
    auto* replace_all_button = buttons->addButton("Replace All", QDialogButtonBox::ActionRole);
    buttons->addButton(QDialogButtonBox::Cancel);
    form->addRow(buttons);

    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(replace_button, &QPushButton::clicked, &dialog, [&dialog]() { dialog.done(1); });
    connect(replace_all_button, &QPushButton::clicked, &dialog, [&dialog]() { dialog.done(2); });

    const int result = dialog.exec();
    if (result == QDialog::Rejected) {
        return;
    }

    const QString find_text = find_edit->text();
    if (find_text.isEmpty()) {
        return;
    }

    find_script_text_ = find_text;
    replace_script_text_ = replace_edit->text();

    if (result == 1) {
        if (!replace_one_in_editor(editor_, find_script_text_, replace_script_text_)) {
            statusBar()->showMessage("Replace in Script: no matches", 3000);
        } else {
            statusBar()->showMessage("Replace in Script: replaced 1 occurrence", 3000);
        }
        return;
    }

    const int count = replace_all_in_editor(editor_, find_script_text_, replace_script_text_);
    if (count == 0) {
        statusBar()->showMessage("Replace in Script: no matches", 3000);
    } else if (count == 1) {
        statusBar()->showMessage("Replace in Script: replaced 1 occurrence", 3000);
    } else {
        statusBar()->showMessage(QString("Replace in Script: replaced %1 occurrences").arg(count), 3000);
    }
}

void MainWindow::replace_next_in_script() {
    editor_->setFocus();
    if (find_script_text_.isEmpty()) {
        replace_in_script();
        return;
    }

    if (!replace_one_in_editor(editor_, find_script_text_, replace_script_text_)) {
        statusBar()->showMessage("Replace in Script: no matches", 3000);
    } else {
        statusBar()->showMessage("Replace in Script: replaced 1 occurrence", 3000);
    }
}

void MainWindow::replace_all_in_script() {
    editor_->setFocus();
    if (find_script_text_.isEmpty()) {
        replace_in_script();
        return;
    }

    const int count = replace_all_in_editor(editor_, find_script_text_, replace_script_text_);
    if (count == 0) {
        statusBar()->showMessage("Replace All in Script: no matches", 3000);
    } else if (count == 1) {
        statusBar()->showMessage("Replace All in Script: replaced 1 occurrence", 3000);
    } else {
        statusBar()->showMessage(QString("Replace All in Script: replaced %1 occurrences").arg(count), 3000);
    }
}

void MainWindow::go_to_line() {
    editor_->setFocus();
    const int max_line = std::max(1, editor_->blockCount());
    bool ok = false;
    const int line =
        QInputDialog::getInt(this, "Go to Line", "Line number:", editor_->textCursor().blockNumber() + 1, 1,
                             max_line, 1, &ok);
    if (!ok) {
        return;
    }
    QTextCursor cursor = editor_->textCursor();
    cursor.movePosition(QTextCursor::Start);
    cursor.movePosition(QTextCursor::Down, QTextCursor::MoveAnchor, line - 1);
    editor_->setTextCursor(cursor);
    editor_->centerCursor();
}

void MainWindow::go_to_matching_brace() {
    editor_->setFocus();
    if (!go_to_matching_brace_in_editor(editor_)) {
        statusBar()->showMessage("Go to Matching Brace: no match", 3000);
    }
}

void MainWindow::add_selection_for_next_occurrence() {
    editor_->setFocus();
    if (!add_selection_for_next_occurrence_in_editor(editor_, add_selection_needle_,
                                                     add_selection_extras_)) {
        statusBar()->showMessage("Add Selection: no more matches", 3000);
    }
}

void MainWindow::duplicate_line() {
    editor_->setFocus();
    duplicate_lines_in_editor(editor_);
}

void MainWindow::trim_trailing_whitespace() {
    editor_->setFocus();
    const int count = trim_trailing_whitespace_in_editor(editor_);
    if (count == 0) {
        statusBar()->showMessage("No trailing whitespace", 3000);
    } else if (count == 1) {
        statusBar()->showMessage("Trimmed 1 line", 3000);
    } else {
        statusBar()->showMessage(QString("Trimmed %1 lines").arg(count), 3000);
    }
}

void MainWindow::remove_blank_lines() {
    editor_->setFocus();
    const int count = remove_blank_lines_in_editor(editor_);
    if (count == 0) {
        statusBar()->showMessage("No blank lines", 3000);
    } else if (count == 1) {
        statusBar()->showMessage("Removed 1 blank line", 3000);
    } else {
        statusBar()->showMessage(QString("Removed %1 blank lines").arg(count), 3000);
    }
}

void MainWindow::delete_line() {
    editor_->setFocus();
    delete_lines_in_editor(editor_);
}

void MainWindow::move_line_up() {
    editor_->setFocus();
    move_lines_up_in_editor(editor_);
}

void MainWindow::move_line_down() {
    editor_->setFocus();
    move_lines_down_in_editor(editor_);
}

void MainWindow::select_all() {
    editor_->setFocus();
    editor_->selectAll();
}

void MainWindow::undo_edit() {
    editor_->setFocus();
    editor_->undo();
}

void MainWindow::redo_edit() {
    editor_->setFocus();
    editor_->redo();
}

void MainWindow::toggle_comment() {
    editor_->setFocus();
    toggle_comments_in_editor(editor_);
}

void MainWindow::indent_lines() {
    editor_->setFocus();
    indent_lines_in_editor(editor_);
}

void MainWindow::unindent_lines() {
    editor_->setFocus();
    unindent_lines_in_editor(editor_);
}

void MainWindow::set_plot_panel_visible(bool visible) {
    if (plot_stack_ == nullptr) {
        return;
    }
    plot_stack_->setVisible(visible);
    if (show_plot_panel_action_ != nullptr && show_plot_panel_action_->isChecked() != visible) {
        show_plot_panel_action_->blockSignals(true);
        show_plot_panel_action_->setChecked(visible);
        show_plot_panel_action_->blockSignals(false);
    }
}

void MainWindow::show_about() {
    const QString version = QCoreApplication::applicationVersion();
    QMessageBox::about(
        this, "About MathScript",
        QString("<h3>MathScript IDE</h3>"
                "<p>Version %1</p>"
                "<p>Interactive numerical computing with a REPL, script editor, "
                "variable inspector, and plotting.</p>"
                "<p>Run commands with <b>Run</b> or <b>Ctrl+Enter</b>. "
                "Type <code>help</code> in the REPL for built-in commands.</p>")
            .arg(version.isEmpty() ? QStringLiteral("1.0.0") : version));
}

void MainWindow::open_script_file(const QString& path) {
    if (path.isEmpty()) {
        return;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        append_output("error: failed to open " + path + "\n", true);
        return;
    }
    editor_->setPlainText(QString::fromUtf8(file.readAll()));
    add_recent_file(path);
    append_output("opened " + path + "\n");
}

void MainWindow::add_recent_file(const QString& path) {
    const QString canonical = QFileInfo(path).canonicalFilePath();
    if (canonical.isEmpty()) {
        return;
    }
    recent_files_.removeAll(canonical);
    recent_files_.prepend(canonical);
    while (recent_files_.size() > kMaxRecentFiles) {
        recent_files_.removeLast();
    }
    refresh_recent_menu();
}

void MainWindow::refresh_recent_menu() {
    if (recent_menu_ == nullptr) {
        return;
    }
    recent_menu_->clear();
    if (recent_files_.isEmpty()) {
        auto* empty_action = recent_menu_->addAction("(none)");
        empty_action->setEnabled(false);
        return;
    }
    for (const QString& path : recent_files_) {
        auto* action = recent_menu_->addAction(QFileInfo(path).fileName());
        action->setStatusTip(path);
        action->setToolTip(path);
        connect(action, &QAction::triggered, this, [this, path]() { open_script_file(path); });
    }
}

void MainWindow::adjust_font_size(int delta) {
    if (delta != 0) {
        mono_font_size_ = std::clamp(mono_font_size_ + delta, 8, 24);
    }
    QFont font = editor_->font();
    font.setFamily(QStringLiteral("Consolas"));
    font.setStyleHint(QFont::Monospace);
    font.setPointSize(mono_font_size_);
    editor_->setFont(font);
    output_->setFont(font);
    input_->setFont(font);
    if (line_number_area_ != nullptr) {
        line_number_area_->refresh_layout();
    }
}

void MainWindow::show_welcome_banner() {
    const QString version = QCoreApplication::applicationVersion();
    const QString version_text = version.isEmpty() ? QStringLiteral("1.0.0") : version;
    append_output(QString("MathScript IDE v%1\n").arg(version_text));
    append_output("────────────────────────────────────────\n");
    append_output("Type 'help' and press Run (Ctrl+Enter).\n");
    append_output("Stop cancels long REPL work; Clear Output resets this pane.\n\n");
}

void MainWindow::append_output(const QString& text, bool is_error) {
    output_->moveCursor(QTextCursor::End);
    QTextCharFormat fmt;
    if (is_error) {
        fmt.setForeground(QColor(255, 100, 100));
    } else {
        fmt.setForeground(output_->palette().color(QPalette::Text));
    }
    output_->setCurrentCharFormat(fmt);
    output_->insertPlainText(text);
    output_->moveCursor(QTextCursor::End);
}

void MainWindow::refresh_variables() {
    variables_->clear();
    ms::interp::SessionState state;
    QMetaObject::invokeMethod(repl_worker_, "sessionState", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(ms::interp::SessionState, state));
    for (const auto& [name, value] : state.scalars) {
        auto* item = new QListWidgetItem(QString("%1 = %2").arg(QString::fromStdString(name)).arg(value));
        item->setData(kVarNameRole, QString::fromStdString(name));
        item->setData(kVarKindRole, QString::fromLatin1(kVarKindScalar));
        item->setToolTip(QString("%1 = %2").arg(QString::fromStdString(name)).arg(value));
        variables_->addItem(item);
    }
    for (const auto& [name, matrix] : state.matrices) {
        QString label = QString("%1 (%2x%3)")
                            .arg(QString::fromStdString(name))
                            .arg(static_cast<int>(matrix.rows()))
                            .arg(static_cast<int>(matrix.cols()));
        if (is_small_matrix(matrix)) {
            label += " " + matrix_inline_preview(matrix, kListPreviewMaxRows, kListPreviewMaxCols);
        }
        auto* item = new QListWidgetItem(label);
        item->setData(kVarNameRole, QString::fromStdString(name));
        item->setData(kVarKindRole, QString::fromLatin1(kVarKindMatrix));
        item->setToolTip(matrix_tooltip_text(name, matrix));
        variables_->addItem(item);
    }
    if (variables_->count() == 0) {
        variables_->addItem("(no variables)");
    }
}

void MainWindow::on_variable_double_clicked(QListWidgetItem* item) {
    if (item == nullptr) {
        return;
    }
    const QString kind = item->data(kVarKindRole).toString();
    const QString name = item->data(kVarNameRole).toString();
    if (name.isEmpty() || kind.isEmpty()) {
        return;
    }

    ms::interp::SessionState state;
    QMetaObject::invokeMethod(repl_worker_, "sessionState", Qt::BlockingQueuedConnection,
                              Q_RETURN_ARG(ms::interp::SessionState, state));

    if (kind == QLatin1String(kVarKindScalar)) {
        const auto it = state.scalars.find(name.toStdString());
        if (it != state.scalars.end()) {
            append_output(QString("%1 = %2\n").arg(name).arg(it->second));
        }
        return;
    }

    if (kind != QLatin1String(kVarKindMatrix)) {
        return;
    }

    const auto it = state.matrices.find(name.toStdString());
    if (it == state.matrices.end()) {
        return;
    }
    append_output(matrix_dump_text(it->first, it->second));
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
        append_output("error: no plot to export\n", true);
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
        append_output("error: failed to export plot to " + path + "\n", true);
    }
}

void MainWindow::export_last_result_latex() {
    if (last_result_.trimmed().isEmpty()) {
        append_output("error: no REPL result to export\n", true);
        return;
    }

    const QString path =
        QFileDialog::getSaveFileName(this, "Export Last Result as LaTeX", {}, "LaTeX (*.tex);;All Files (*)");
    if (path.isEmpty()) {
        return;
    }

    const QString latex = wrap_latex_document(result_to_latex_body(last_result_));
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        append_output("error: failed to write " + path + "\n", true);
        return;
    }
    file.write(latex.toUtf8());
    append_output("exported last result to " + path + "\n");
}

void MainWindow::export_command_history() {
    if (repl_busy_) {
        append_output("error: wait for the current REPL command to finish\n", true);
        return;
    }
    const QString path =
        QFileDialog::getSaveFileName(this, "Export Command History", {}, "Text Files (*.txt);;All Files (*)");
    if (path.isEmpty()) {
        return;
    }
    QString err;
    QMetaObject::invokeMethod(
        repl_worker_, "exportHistory", Qt::BlockingQueuedConnection, Q_RETURN_ARG(QString, err),
        Q_ARG(QString, path));
    if (err.isEmpty()) {
        append_output("exported history to " + path + "\n");
    } else {
        append_output("error: " + err + "\n", true);
    }
}

void MainWindow::on_file_activated(const QModelIndex& index) {
    const QString path = file_model_->filePath(index);
    if (file_model_->isDir(index)) {
        return;
    }
    open_script_file(path);
}

void MainWindow::start_eval(const QString& line) {
    repl_busy_ = true;
    run_->setEnabled(false);
    run_script_->setEnabled(false);
    stop_->setEnabled(true);
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
    stop_->setEnabled(false);
    input_->setEnabled(true);
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
    if (event->type() != QEvent::KeyPress) {
        return QMainWindow::eventFilter(obj, event);
    }

    auto* key = static_cast<QKeyEvent*>(event);
    if (obj == editor_) {
        if (key->key() == Qt::Key_Tab && !(key->modifiers() & Qt::ControlModifier)) {
            if (key->modifiers() & Qt::ShiftModifier) {
                unindent_lines();
            } else {
                indent_lines();
            }
            return true;
        }
        return QMainWindow::eventFilter(obj, event);
    }

    if (obj != input_) {
        return QMainWindow::eventFilter(obj, event);
    }
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
        last_result_ = output.trimmed();
        append_output(output);
        if (!output.endsWith('\n')) {
            append_output("\n");
        }
    }
    finish_repl_op();
}

void MainWindow::on_repl_error(const QString& message) {
    append_output("error: " + message + "\n", true);
    finish_repl_op();
}

void MainWindow::on_repl_cancelled() {
    script_queue_.clear();
    append_output("cancelled\n");
    finish_repl_op();
}

void MainWindow::on_stop() {
    if (!repl_busy_) {
        return;
    }
    stop_->setEnabled(false);
    QMetaObject::invokeMethod(repl_worker_, "requestCancel", Qt::QueuedConnection);
}
