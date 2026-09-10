// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include <QString>

namespace ms::gui {

QString camel_case_text(const QString& text);
QString screaming_snake_case_text(const QString& text);
QString trim_leading_whitespace_line(const QString& line);

}  // namespace ms::gui
