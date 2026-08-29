#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"

using namespace ms;

TEST(ErrorTest, dimension_mismatch_formats) {
    const std::string message = format_error(DimensionMismatch{2, 3, 4, 5});
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message.find("expected 4x5"), std::string::npos);
}

TEST(ErrorTest, singular_matrix_formats) {
    const std::string message = format_error(SingularMatrix{1e15});
    EXPECT_FALSE(message.empty());
    EXPECT_EQ(message, "singular matrix");
}

TEST(ErrorTest, convergence_fail_formats) {
    const std::string message = format_error(ConvergenceFail{100, 1e-3});
    EXPECT_FALSE(message.empty());
    EXPECT_EQ(message, "convergence failed");
}

TEST(ErrorTest, domain_error_formats) {
    const std::string message = format_error(DomainError{"asin", "argument > 1"});
    EXPECT_FALSE(message.empty());
    EXPECT_EQ(message, "asin: argument > 1");
}

TEST(ErrorTest, all_variants_format_without_crash) {
    const Error variants[] = {
        Error{DimensionMismatch{1, 2, 3, 4}},
        Error{SingularMatrix{42.0}},
        Error{DeviceError{1, "device"}},
        Error{AllocationFailure{1024}},
        Error{ConvergenceFail{50, 1e-6}},
        Error{DomainError{"fn", "reason"}},
        Error{DistributedError{0, 1}},
        Error{SymbolicError{"symbolic"}},
        Error{IOError{"path", "reason"}},
        Error{ParseError{1, 2, "token"}},
        Error{OverflowError{"mul"}},
        Error{PluginViolation{"rule", "loc"}},
        Error{ValueOutOfRange{"param", 2.0, 0.0, 1.0}},
    };

    for (const auto& err : variants) {
        const std::string message = format_error(err);
        EXPECT_FALSE(message.empty());
    }
}

TEST(ErrorTest, expected_chain) {
    auto sqrt_or_zero = [](double x) -> Result<double> {
        return Result<double>(x)
            .and_then([](double v) -> Result<double> {
                if (v < 0.0) {
                    return std::unexpected(DomainError{"sqrt", "negative"});
                }
                return std::sqrt(v);
            })
            .or_else([](const Error&) -> Result<double> { return 0.0; });
    };

    const auto ok = sqrt_or_zero(9.0);
    ASSERT_TRUE(ok.has_value());
    EXPECT_DOUBLE_EQ(*ok, 3.0);

    const auto recovered = sqrt_or_zero(-4.0);
    ASSERT_TRUE(recovered.has_value());
    EXPECT_DOUBLE_EQ(*recovered, 0.0);
}
