#include "gui/LineNumberArea.hpp"

#include <QEvent>
#include <QFontMetrics>
#include <QPainter>
#include <QPlainTextEdit>
#include <QScrollBar>
#include <QTextBlock>

LineNumberArea::LineNumberArea(QPlainTextEdit* editor) : QWidget(editor), editor_(editor) {
    setAutoFillBackground(true);
    editor_->installEventFilter(this);
    reposition();

    connect(editor_, &QPlainTextEdit::blockCountChanged, this, &LineNumberArea::update_area_width);
    connect(editor_, &QPlainTextEdit::updateRequest, this, &LineNumberArea::update_area);
    connect(editor_, &QPlainTextEdit::cursorPositionChanged, this, [this]() {
        if (editor_ != nullptr) {
            update();
        }
    });

    update_area_width();
}

int LineNumberArea::area_width() const {
    if (editor_ == nullptr) {
        return 0;
    }
    int digits = 1;
    int max_lines = std::max(1, editor_->blockCount());
    while (max_lines >= 10) {
        max_lines /= 10;
        ++digits;
    }
    const int space = 3 + editor_->fontMetrics().horizontalAdvance(QLatin1Char('9')) * digits;
    return space;
}

void LineNumberArea::refresh_layout() {
    update_area_width();
    update();
}

void LineNumberArea::reposition() {
    if (editor_ == nullptr) {
        return;
    }
    const QRect cr = editor_->contentsRect();
    setGeometry(QRect(cr.left(), cr.top(), area_width(), cr.height()));
}

void LineNumberArea::update_area_width() {
    if (editor_ == nullptr) {
        return;
    }
    editor_->setViewportMargins(area_width(), 0, 0, 0);
    reposition();
}

bool LineNumberArea::eventFilter(QObject* obj, QEvent* event) {
    if (obj == editor_ && event->type() == QEvent::Resize) {
        reposition();
    }
    return QWidget::eventFilter(obj, event);
}

void LineNumberArea::update_area(const QRect& rect, int dy) {
    if (dy != 0) {
        scroll(0, dy);
    } else {
        update(0, rect.y(), width(), rect.height());
    }
    if (rect.contains(editor_->viewport()->rect())) {
        update_area_width();
    }
}

void LineNumberArea::paintEvent(QPaintEvent* event) {
    if (editor_ == nullptr) {
        return;
    }

    QPainter painter(this);
    painter.fillRect(event->rect(), editor_->palette().color(QPalette::AlternateBase));

    QTextBlock block = editor_->firstVisibleBlock();
    int block_number = block.blockNumber();
    int top =
        qRound(editor_->blockBoundingGeometry(block).translated(editor_->contentOffset()).top());
    int bottom = top + qRound(editor_->blockBoundingRect(block).height());

    const int current_line = editor_->textCursor().blockNumber();
    const QColor normal = editor_->palette().color(QPalette::Text).darker(130);
    const QColor current = editor_->palette().color(QPalette::Highlight);

    while (block.isValid() && top <= event->rect().bottom()) {
        if (block.isVisible() && bottom >= event->rect().top()) {
            const QString number = QString::number(block_number + 1);
            painter.setPen(block_number == current_line ? current : normal);
            painter.drawText(0, top, width() - 2, editor_->fontMetrics().height(),
                             Qt::AlignRight, number);
        }

        block = block.next();
        top = bottom;
        bottom = top + qRound(editor_->blockBoundingRect(block).height());
        ++block_number;
    }
}
