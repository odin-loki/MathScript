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
inline std::string format_scalar(double value) {
    if (!std::isfinite(value)) {
        return std::to_string(value);
    }
    const double magnitude = std::abs(value);
    if (value == 0.0 || (magnitude >= 5e-7 && magnitude < 1e16)) {
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

/// Integers keep std::to_string exactly. This overload is what makes it safe to route
/// every numeric result through one name: a factorial or a prime index must not
/// acquire exponent notation because it happened to be large, and combo_factorial(20)
/// is 2432902008176640000, not 2.43290200817664e+18.
template <std::integral T>
inline std::string format_scalar(T value) {
    return std::to_string(value);
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
