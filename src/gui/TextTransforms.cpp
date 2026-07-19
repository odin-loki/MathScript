#include "gui/TextTransforms.hpp"

namespace ms::gui {

namespace {

QString snake_case_text(const QString& text) {
    QString result;
    result.reserve(text.size() + 8);
    bool prev_was_lower_or_digit = false;
    for (int i = 0; i < text.size(); ++i) {
        const QChar ch = text.at(i);
        if (ch.isSpace() || ch == '-' || ch == '_') {
            if (!result.isEmpty() && result.back() != '_') {
                result += '_';
            }
            prev_was_lower_or_digit = false;
            continue;
        }
        if (ch.isUpper()) {
            if (prev_was_lower_or_digit ||
                (i + 1 < text.size() && text.at(i + 1).isLower())) {
                if (!result.isEmpty() && result.back() != '_') {
                    result += '_';
                }
            }
            result += ch.toLower();
            prev_was_lower_or_digit = true;
        } else if (ch.isLower() || ch.isDigit()) {
            result += ch;
            prev_was_lower_or_digit = true;
        } else if (!result.isEmpty() && result.back() != '_') {
            result += '_';
            prev_was_lower_or_digit = false;
        }
    }
    while (!result.isEmpty() && result.back() == '_') {
        result.chop(1);
    }
    return result;
}

}  // namespace

QString camel_case_text(const QString& text) {
    const QString snake = snake_case_text(text);
    const QStringList parts = snake.split(QLatin1Char('_'), Qt::SkipEmptyParts);
    if (parts.isEmpty()) {
        return QString();
    }

    QString result = parts.front();
    for (int i = 1; i < parts.size(); ++i) {
        const QString& part = parts.at(i);
        if (part.isEmpty()) {
            continue;
        }
        result += part.front().toUpper();
        if (part.size() > 1) {
            result += part.mid(1);
        }
    }
    return result;
}

QString screaming_snake_case_text(const QString& text) {
    return snake_case_text(text).toUpper();
}

QString trim_leading_whitespace_line(const QString& line) {
    int start = 0;
    while (start < line.size()) {
        const QChar ch = line.at(start);
        if (ch != QLatin1Char(' ') && ch != QLatin1Char('\t')) {
            break;
        }
        ++start;
    }
    return line.mid(start);
}

}  // namespace ms::gui
