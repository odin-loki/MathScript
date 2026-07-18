#pragma once

#include <atomic>

#include <QObject>
#include <QString>

#include "ms/interp/repl_engine.hpp"

// Wave 220 — REPL eval on a background QThread (see MainWindow.cpp).
// ReplWorker owns the ms::interp::Interpreter and runs execute() off the UI
// thread. MainWindow posts lines via evaluate(); results arrive on finished/error
// (Qt::QueuedConnection). Build verification: ms_gui compiles when
// MS_BUILD_GUI=ON and Qt6 is found (see src/gui/CMakeLists.txt).
class ReplWorker : public QObject {
    Q_OBJECT

public:
    explicit ReplWorker(QObject* parent = nullptr);

public slots:
    void evaluate(const QString& line);
    void requestCancel();
    QString saveSession(const QString& path);
    QString loadSession(const QString& path);
    QString exportHistory(const QString& path);
    ms::interp::SessionState sessionState() const;
    bool hasPlot() const;
    ms::interp::PlotSeries plot() const;

signals:
    void finished(QString output);
    void error(QString message);
    void cancelled();

private:
    ms::interp::Interpreter interp_;
    std::atomic<bool> cancel_requested_{false};
};
