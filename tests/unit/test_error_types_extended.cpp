// MathScript Extended Error Types Tests
// Comprehensive coverage of all error type construction and format_error()

#include <gtest/gtest.h>
#include <string>
#include <variant>

#include "ms/error/error_types.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// DimensionMismatch
// ---------------------------------------------------------------------------

TEST(ErrorTypes, DimensionMismatch_2Arg) {
    const DimensionMismatch e(3, 4);
    EXPECT_EQ(e.got_rows, 3u);
    EXPECT_EQ(e.got_cols, 4u);
    EXPECT_EQ(e.expected_rows, 0u);
    EXPECT_EQ(e.expected_cols, 0u);
}

TEST(ErrorTypes, DimensionMismatch_4Arg) {
    const DimensionMismatch e(2, 3, 4, 5);
    EXPECT_EQ(e.got_rows, 2u);
    EXPECT_EQ(e.got_cols, 3u);
    EXPECT_EQ(e.expected_rows, 4u);
    EXPECT_EQ(e.expected_cols, 5u);
}

TEST(ErrorTypes, DimensionMismatch_InError) {
    const Error err = DimensionMismatch(1, 2, 3, 4);
    EXPECT_TRUE(std::holds_alternative<DimensionMismatch>(err));
    const auto& e = std::get<DimensionMismatch>(err);
    EXPECT_EQ(e.got_rows, 1u);
}

TEST(ErrorTypes, DimensionMismatch_FormatError) {
    const Error err = DimensionMismatch(3, 4, 5, 6);
    const std::string msg = format_error(err);
    EXPECT_FALSE(msg.empty());
}

// ---------------------------------------------------------------------------
// SingularMatrix
// ---------------------------------------------------------------------------

TEST(ErrorTypes, SingularMatrix_Default) {
    const SingularMatrix e;
    EXPECT_EQ(e.condition_number, 0.0);
}

TEST(ErrorTypes, SingularMatrix_WithValue) {
    const SingularMatrix e(1e15);
    EXPECT_EQ(e.condition_number, 1e15);
}

TEST(ErrorTypes, SingularMatrix_FormatError) {
    const Error err = SingularMatrix(1e12);
    const std::string msg = format_error(err);
    EXPECT_FALSE(msg.empty());
}

// ---------------------------------------------------------------------------
// ConvergenceFail
// ---------------------------------------------------------------------------

TEST(ErrorTypes, ConvergenceFail_Fields) {
    const ConvergenceFail e{100, 1e-3};
    EXPECT_EQ(e.iterations, 100u);
    EXPECT_DOUBLE_EQ(e.residual, 1e-3);
}

TEST(ErrorTypes, ConvergenceFail_InError) {
    const Error err = ConvergenceFail{50, 0.5};
    EXPECT_TRUE(std::holds_alternative<ConvergenceFail>(err));
    const auto& e = std::get<ConvergenceFail>(err);
    EXPECT_EQ(e.iterations, 50u);
}

TEST(ErrorTypes, ConvergenceFail_FormatError) {
    const Error err = ConvergenceFail{1000, 1e-5};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// DomainError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, DomainError_Fields) {
    const DomainError e{"sqrt", "negative argument"};
    EXPECT_EQ(e.function, "sqrt");
    EXPECT_EQ(e.reason, "negative argument");
}

TEST(ErrorTypes, DomainError_FormatError) {
    const Error err = DomainError{"log", "zero argument"};
    const std::string msg = format_error(err);
    EXPECT_FALSE(msg.empty());
}

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, ParseError_Fields) {
    const ParseError e{10, 5, "unexpected token"};
    EXPECT_EQ(e.line, 10u);
    EXPECT_EQ(e.col, 5u);
    EXPECT_EQ(e.msg, "unexpected token");
}

TEST(ErrorTypes, ParseError_FormatError) {
    const Error err = ParseError{1, 1, "syntax error"};
    const std::string msg = format_error(err);
    EXPECT_FALSE(msg.empty());
}

// ---------------------------------------------------------------------------
// OverflowError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, OverflowError_Fields) {
    const OverflowError e{"exp"};
    EXPECT_EQ(e.op, "exp");
}

TEST(ErrorTypes, OverflowError_FormatError) {
    const Error err = OverflowError{"factorial"};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// AllocationFailure
// ---------------------------------------------------------------------------

TEST(ErrorTypes, AllocationFailure_Fields) {
    const AllocationFailure e{1024};
    EXPECT_EQ(e.requested_bytes, 1024u);
}

TEST(ErrorTypes, AllocationFailure_FormatError) {
    const Error err = AllocationFailure{4096};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// DeviceError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, DeviceError_Fields) {
    const DeviceError e{-1, "GPU not found"};
    EXPECT_EQ(e.code, -1);
    EXPECT_EQ(e.msg, "GPU not found");
}

TEST(ErrorTypes, DeviceError_FormatError) {
    const Error err = DeviceError{1, "device busy"};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// SymbolicError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, SymbolicError_Fields) {
    const SymbolicError e{"undeclared variable"};
    EXPECT_EQ(e.reason, "undeclared variable");
}

TEST(ErrorTypes, SymbolicError_FormatError) {
    const Error err = SymbolicError{"undefined symbol"};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// IOError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, IOError_Fields) {
    const IOError e{"data.csv", "file not found"};
    EXPECT_EQ(e.path, "data.csv");
    EXPECT_EQ(e.reason, "file not found");
}

TEST(ErrorTypes, IOError_FormatError) {
    const Error err = IOError{"matrix.txt", "permission denied"};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// PluginViolation
// ---------------------------------------------------------------------------

TEST(ErrorTypes, PluginViolation_Fields) {
    const PluginViolation e{"no_raw_ptr", "main.cpp:42"};
    EXPECT_EQ(e.rule, "no_raw_ptr");
    EXPECT_EQ(e.location, "main.cpp:42");
}

TEST(ErrorTypes, PluginViolation_FormatError) {
    const Error err = PluginViolation{"unsafe_cast", "src/foo.cpp:10"};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// DistributedError
// ---------------------------------------------------------------------------

TEST(ErrorTypes, DistributedError_Fields) {
    const DistributedError e{2, -3};
    EXPECT_EQ(e.rank, 2);
    EXPECT_EQ(e.mpi_code, -3);
}

TEST(ErrorTypes, DistributedError_FormatError) {
    const Error err = DistributedError{0, 1};
    EXPECT_FALSE(format_error(err).empty());
}

// ---------------------------------------------------------------------------
// Result<T> wrapper behavior
// ---------------------------------------------------------------------------

TEST(ErrorTypes, Result_Success_HasValue) {
    const Result<double> r = 3.14;
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(*r, 3.14);
}

TEST(ErrorTypes, Result_Error_NoValue) {
    const Result<double> r = std::unexpected(DimensionMismatch(1, 2));
    EXPECT_FALSE(r.has_value());
}

TEST(ErrorTypes, Result_IntSuccess) {
    const Result<int> r = 42;
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 42);
}

TEST(ErrorTypes, Result_Error_CanExtractError) {
    const Result<int> r = std::unexpected(SingularMatrix(1e8));
    EXPECT_FALSE(r.has_value());
    const auto& err = r.error();
    EXPECT_TRUE(std::holds_alternative<SingularMatrix>(err));
}
