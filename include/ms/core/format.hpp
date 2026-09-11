// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include <cmath>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace ms {

/// How MathScript renders a number for a human to read.
///
/// std::to_string(double) is printf("%f"): six decimal places and nothing else. That
/// is a perfectly good spelling for the magnitudes anyone types, and it is the one
/// the test corpus and the documentation are written in -- but it cannot represent
/// everything. Below about 5e-7 it prints 0.000000, so a value does not merely lose
/// precision, it disappears; at 1e16 and above it emits a long integer with a
/// spurious ".000000" tail. Either way the session prints something it cannot read
/// back.
///
/// So: keep the six-decimal spelling wherever it is faithful, and fall back to the
/// shortest form that reads back as the same double everywhere else. The fallback is
/// found by asking for successively more significant digits until strtod returns the
/// value it started from, which is what makes it a round trip rather than a guess.
///
/// "Faithful" here is a display's standard, not a file's: six decimals may round a
/// value, they may not erase it. So the lower bound is not a constant -- an early
/// version guessed 5e-7 and was wrong, because printf("%f", 5e-7) is "0.000000" -- it
/// is the question itself, asked of the spelling that was actually produced.
inline std::string format_scalar(double value) {
    if (!std::isfinite(value)) {
        return std::to_string(value);
    }
    if (std::abs(value) < 1e16) {
        std::string text = std::to_string(value);
        if (value == 0.0 || std::strtod(text.c_str(), nullptr) != 0.0) {
            return text;
        }
    }
    char buffer[64];
    for (int precision = 6; precision <= 17; ++precision) {
        std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
        if (std::strtod(buffer, nullptr) == value) {
            break;
        }
    }
    return buffer;
}

/// Integers keep std::to_string exactly. This overload is what makes it safe to route
/// every numeric result through one name: a factorial or a prime index must not
/// acquire exponent notation because it happened to be large, and combo_factorial(20)
/// is 2432902008176640000, not 2.43290200817664e+18.
template <std::integral T>
inline std::string format_scalar(T value) {
    return std::to_string(value);
}

/// A compact spelling for a preview: a cell in a variable list, a tooltip, a truncated
/// matrix dump. It rounds to `decimals` places and, unless `trim_zeros` is false, drops
/// the trailing zeros, which is what a preview wants -- but a preview may abbreviate a value, not erase it. At four
/// decimals 1e-9 renders "0.0000", which trims to "0", and the reader is then looking
/// at a cell that says the entry is zero when it is not. Where the rounded form would
/// claim that, or where the magnitude is past what %f spells readably, this falls back
/// to the round-tripping spelling instead.
inline std::string format_preview(double value, int decimals = 4, bool trim_zeros = true) {
    if (!std::isfinite(value) || std::abs(value) >= 1e16) {
        return format_scalar(value);
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    std::string text(buffer);
    if (!trim_zeros) {
        // A column of numbers reads better with its decimals aligned, so a table keeps
        // the padding. The rule about never printing a non-zero as zero still applies.
        if (value != 0.0 && std::strtod(text.c_str(), nullptr) == 0.0) {
            return format_scalar(value);
        }
        return text;
    }
    // Only the digits after the point are padding. Trimming unconditionally would turn
    // a whole number into a different one -- "100" with decimals = 0 has no point to
    // stop at, and comes out "1".
    if (text.find('.') != std::string::npos) {
        while (!text.empty() && text.back() == '0') {
            text.pop_back();
        }
        if (!text.empty() && text.back() == '.') {
            text.pop_back();
        }
    }
    if (text.empty()) {
        text = "0";
    }
    if (value != 0.0 && std::strtod(text.c_str(), nullptr) == 0.0) {
        return format_scalar(value);
    }
    return text;
}

/// Full precision, for a file rather than a screen. Saving a session is not a
/// display: a value written out and read back has to be the same value, so this
/// always emits enough digits to round-trip and never rounds for looks.
inline std::string format_exact(double value) {
    if (!std::isfinite(value)) {
        return std::to_string(value);
    }
    char buffer[64];
    for (int precision = 6; precision <= 17; ++precision) {
        std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
        if (std::strtod(buffer, nullptr) == value) {
            break;
        }
    }
    return buffer;
}

} // namespace ms
