// MathScript Error Types
// All error types for std::expected<T, Error>

#pragma once

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

struct ParseError {
    size_t line, col;
    std::string_view msg;
};

struct OverflowError {
    std::string_view op;
};

struct PluginViolation {
    std::string_view rule;
    std::string_view location;
};

using Error = std::variant<
    DimensionMismatch, SingularMatrix, DeviceError,
    AllocationFailure, ConvergenceFail, DomainError,
    DistributedError, SymbolicError, IOError,
    ParseError, OverflowError, PluginViolation
>;

template<typename T>
using Result = std::expected<T, Error>;

std::string format_error(const Error& error);

} // namespace ms