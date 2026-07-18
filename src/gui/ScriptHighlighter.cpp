#include "gui/ScriptHighlighter.hpp"

#include <QColor>
#include <QFont>
#include <QTextDocument>

ScriptHighlighter::ScriptHighlighter(QTextDocument* parent) : QSyntaxHighlighter(parent) {
    comment_format_.setForeground(QColor(106, 153, 85));
    comment_format_.setFontItalic(true);

    QTextCharFormat string_format;
    string_format.setForeground(QColor(206, 145, 120));

    QTextCharFormat number_format;
    number_format.setForeground(QColor(181, 206, 168));

    QTextCharFormat keyword_format;
    keyword_format.setForeground(QColor(86, 156, 214));
    keyword_format.setFontWeight(QFont::Bold);

    QTextCharFormat prefix_format;
    prefix_format.setForeground(QColor(78, 201, 176));

    rules_.append({QRegularExpression(QStringLiteral(R"("([^"\\]|\\.)*")")),
                   string_format});
    rules_.append({QRegularExpression(QStringLiteral(R"('([^'\\]|\\.)*')")),
                   string_format});
    rules_.append({QRegularExpression(QStringLiteral(R"(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)")),
                   number_format});
    rules_.append({QRegularExpression(
                       QStringLiteral(R"(\b(sym_|ode_|finance_|crypto_|fem_|cfd_)[A-Za-z_]\w*)")),
                   prefix_format});
    rules_.append({QRegularExpression(QStringLiteral(R"(\b(help|plot|clear|exit|vars)\b)")),
                   keyword_format});
}

int ScriptHighlighter::comment_start(const QString& text) const {
    bool in_string = false;
    QChar string_delim;

    for (int i = 0; i < text.length(); ++i) {
        const QChar c = text.at(i);
        if (in_string) {
            if (c == QLatin1Char('\\') && i + 1 < text.length()) {
                ++i;
                continue;
            }
            if (c == string_delim) {
                in_string = false;
            }
            continue;
        }
        if (c == QLatin1Char('"') || c == QLatin1Char('\'')) {
            in_string = true;
            string_delim = c;
            continue;
        }
        if (c == QLatin1Char('#')) {
            return i;
        }
        if (c == QLatin1Char('/') && i + 1 < text.length() && text.at(i + 1) == QLatin1Char('/')) {
            return i;
        }
    }
    return -1;
}

void ScriptHighlighter::apply_rules(const QString& text, int length) {
    const QString code = text.left(length);
    for (const Rule& rule : rules_) {
        auto it = rule.pattern.globalMatch(code);
        while (it.hasNext()) {
            const QRegularExpressionMatch match = it.next();
            setFormat(match.capturedStart(), match.capturedLength(), rule.format);
        }
    }
}

void ScriptHighlighter::highlightBlock(const QString& text) {
    const int comment_at = comment_start(text);
    const int code_length = comment_at >= 0 ? comment_at : text.length();

    apply_rules(text, code_length);

    if (comment_at >= 0) {
        setFormat(comment_at, text.length() - comment_at, comment_format_);
    }
}
