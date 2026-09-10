// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MathScript Error Types
// All error types for std::expected<T, Error>

#pragma once

#include "ms/unsafe/unsafe.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <variant>

namespace ms {

struct DimensionMismatch {
    size_t got_rows, got_cols;
    size_t expected_rows, expected_cols;

    DimensionMismatch(size_t a, size_t b, size_t c = 0, size_t d = 0)
        : got_rows(a), got_cols(b), expected_rows(c), expected_cols(d) {}
};

struct SingularMatrix {
    double condition_number;

    explicit SingularMatrix(double condition = 0.0) : condition_number(condition) {}
};

struct DeviceError {
    int code;
    std::string_view msg;
};

struct AllocationFailure {
    size_t requested_bytes;
};

struct ConvergenceFail {
    size_t iterations;
    double residual;
};

struct DomainError {
    std::string function;
    std::string reason;
};

struct DistributedError {
    int rank;
    int mpi_code;
};

struct SymbolicError {
    std::string_view reason;
};

struct IOError {
    std::string_view path;
    std::string_view reason;
};

/// A position and what went wrong there. `msg` owns its text, unlike the other error
/// types here: every one of those names a fixed condition and can point at a literal,
/// while a parse diagnostic has to say what it found, which is a substring of the input
/// and so is built at run time. A `string_view` here would be a dangling view at every
/// site worth writing.
///
/// `line` and `col` are 1-based; both zero means the position is not known.
struct ParseError {
    size_t line, col;
    std::string msg;
};

struct OverflowError {
    std::string_view op;
};

struct PluginViolation {
    std::string_view rule;
    std::string_view location;
};

struct ValueOutOfRange {
    std::string param;
    double value;
    double lo;
    double hi;
};

using Error = std::variant<
    DimensionMismatch, SingularMatrix, DeviceError,
    AllocationFailure, ConvergenceFail, DomainError,
    DistributedError, SymbolicError, IOError,
    ParseError, OverflowError, PluginViolation,
    ValueOutOfRange
>;

template<typename T>
using Result = std::expected<T, Error>;

std::string format_error(const Error& error);

inline Result<double> check_range(
    std::string param, double value, double lo, double hi) {
    if (value < lo || value > hi) {
        return std::unexpected(
            Error{ValueOutOfRange{std::move(param), value, lo, hi}});
    }
    return value;
}

} // namespace ms