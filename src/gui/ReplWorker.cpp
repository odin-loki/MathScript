#include "gui/ReplWorker.hpp"

#include "ms/error/error_types.hpp"

ReplWorker::ReplWorker(QObject* parent) : QObject(parent) {}

void ReplWorker::evaluate(const QString& line) {
    const auto result = interp_.execute(line.toStdString());
    if (result) {
        emit finished(result->empty() ? QString{} : QString::fromStdString(*result));
    } else {
        emit error(QString::fromStdString(ms::format_error(result.error())));
    }
}

QString ReplWorker::saveSession(const QString& path) {
    const auto result = interp_.save_session(path.toStdString());
    if (result) {
        return {};
    }
    return QString::fromStdString(ms::format_error(result.error()));
}

QString ReplWorker::loadSession(const QString& path) {
    const auto result = interp_.load_session(path.toStdString());
    if (result) {
        return {};
    }
    return QString::fromStdString(ms::format_error(result.error()));
}

ms::interp::SessionState ReplWorker::sessionState() const {
    return interp_.state();
}

bool ReplWorker::hasPlot() const {
    return interp_.has_plot();
}

ms::interp::PlotSeries ReplWorker::plot() const {
    return interp_.plot();
}
