#pragma once

#include <QWidget>

class QPlainTextEdit;
class QResizeEvent;

class LineNumberArea : public QWidget {
    Q_OBJECT

public:
    explicit LineNumberArea(QPlainTextEdit* editor);

    int area_width() const;
    void refresh_layout();

protected:
    void paintEvent(QPaintEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

private:
    void reposition();
    void update_area_width();
    void update_area(const QRect& rect, int dy);

    QPlainTextEdit* editor_ = nullptr;
};
