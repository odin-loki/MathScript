// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The three spellings in ms/core/format.hpp, and the boundaries between them.
//
// They exist because std::to_string(double) is printf("%f") and nothing else: six
// decimal places, so anything below about 5e-7 prints as 0.000000 and anything from
// 1e16 up grows a spurious ".000000" tail. A value did not lose precision at those
// magnitudes, it disappeared -- and the REPL, the session file and the GUI each showed
// it disappearing in a slightly different way.
//
// format_scalar is for a screen: keep the six-decimal spelling wherever it round-trips,
// because the whole test corpus and every documented example is written in it, and fall
// back to the shortest form that reads back as the same double elsewhere.
// format_exact is for a file: always round-trip, never round for looks.
// format_preview is for a cell in a list: round hard, but never round a non-zero to 0.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "ms/core/format.hpp"

using ms::format_exact;
using ms::format_preview;
using ms::format_scalar;

namespace {

double read_back(const std::string& text) {
    return std::strtod(text.c_str(), nullptr);
}

TEST(Format, ScalarKeepsTheSixDecimalSpellingWhereItIsFaithful) {
    EXPECT_EQ(format_scalar(0.0), "0.000000");
    EXPECT_EQ(format_scalar(1.0), "1.000000");
    EXPECT_EQ(format_scalar(2.5), "2.500000");
    EXPECT_EQ(format_scalar(-3.25), "-3.250000");
    EXPECT_EQ(format_scalar(1234.5), "1234.500000");
}

TEST(Format, ScalarSwitchesRatherThanPrintingZero) {
    // The defect: every one of these used to print 0.000000.
    EXPECT_EQ(read_back(format_scalar(1e-9)), 1e-9);
    EXPECT_EQ(read_back(format_scalar(-1e-9)), -1e-9);
    EXPECT_EQ(read_back(format_scalar(4.9e-324)), 4.9e-324);
    EXPECT_NE(format_scalar(1e-9), "0.000000");
}

TEST(Format, ScalarSwitchesRatherThanGrowingATail) {
    EXPECT_EQ(read_back(format_scalar(1e16)), 1e16);
    EXPECT_EQ(read_back(format_scalar(1e300)), 1e300);
    EXPECT_EQ(format_scalar(1e16).find(".000000"), std::string::npos);
}

// The rule at the small end is not a threshold, it is a question: does the six-decimal
// spelling still say something other than zero? An early version guessed the boundary
// at 5e-7 and was wrong by exactly one value, because printf("%f", 5e-7) is "0.000000"
// and 5e-7 is not zero. Nothing that is not zero may print as zero, at any magnitude.
TEST(Format, ScalarNeverPrintsANonZeroAsZero) {
    for (const double value : {5e-7, 4.9e-7, 1e-7, 1e-30, 4.9e-324,
                               -5e-7, -1e-7, -1e-30}) {
        const std::string text = format_scalar(value);
        EXPECT_NE(read_back(text), 0.0) << value << " printed as " << text;
    }
    // Just above, the plain spelling is kept.
    EXPECT_EQ(format_scalar(1e-6), "0.000001");
    EXPECT_EQ(format_scalar(0.5), "0.500000");
}

// The upper boundary is 1e16, where %f stops being a readable spelling and starts
// being a long integer with a decimal tail.
TEST(Format, ScalarBoundariesAreWhereTheyAreDocumented) {
    EXPECT_EQ(format_scalar(9999999999999998.0), "9999999999999998.000000");
    EXPECT_EQ(format_scalar(1e16).find(".000000"), std::string::npos);
    // Zero is zero, with the sign it was given.
    EXPECT_EQ(format_scalar(0.0), "0.000000");
    EXPECT_EQ(format_scalar(-0.0), std::to_string(-0.0));
}

TEST(Format, IntegralValuesKeepTheirExactSpelling) {
    // A count must not acquire exponent notation for being large: 20! is
    // 2432902008176640000, not 2.43290200817664e+18.
    EXPECT_EQ(format_scalar(static_cast<std::uint64_t>(2432902008176640000ULL)),
              "2432902008176640000");
    EXPECT_EQ(format_scalar(UINT64_MAX), "18446744073709551615");
    EXPECT_EQ(format_scalar(0), "0");
    EXPECT_EQ(format_scalar(-7), "-7");
}

TEST(Format, ScalarSaysWhatItCannotSpell) {
    EXPECT_EQ(format_scalar(std::numeric_limits<double>::quiet_NaN()),
              std::to_string(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_EQ(format_scalar(std::numeric_limits<double>::infinity()),
              std::to_string(std::numeric_limits<double>::infinity()));
}

// A file is read back, so this one has no readable-magnitude exception at all.
TEST(Format, ExactRoundTripsEveryFiniteDouble) {
    const std::vector<double> values{
        0.0,
        -0.0,
        1.0,
        0.1,
        1.0 / 3.0,
        1.23456789,
        std::nextafter(1.0, 2.0),
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        1e-300,
        1e300,
        12345678901234567890.0,
    };
    for (const double value : values) {
        const std::string text = format_exact(value);
        const double back = read_back(text);
        EXPECT_EQ(std::memcmp(&back, &value, sizeof(double)), 0)
            << value << " wrote " << text;
    }
}

TEST(Format, ExactStaysShortForShortValues) {
    EXPECT_EQ(format_exact(1.0), "1");
    EXPECT_EQ(format_exact(2.5), "2.5");
    EXPECT_EQ(format_exact(0.0), "0");
    EXPECT_EQ(format_exact(-4.0), "-4");
}

TEST(Format, PreviewRoundsHard) {
    EXPECT_EQ(format_preview(1.0), "1");
    EXPECT_EQ(format_preview(2.5), "2.5");
    EXPECT_EQ(format_preview(0.0), "0");
    EXPECT_EQ(format_preview(1.23456789), "1.2346");
    EXPECT_EQ(format_preview(-1.23456789), "-1.2346");
    EXPECT_EQ(format_preview(100.0), "100");
    EXPECT_EQ(format_preview(1.23456789, 2), "1.23");
    // With no decimals there is no point to stop the trim at, so a whole number must
    // not have its own zeros eaten: 100 is not 1.
    EXPECT_EQ(format_preview(100.0, 0), "100");
    EXPECT_EQ(format_preview(1200.0, 0), "1200");
    EXPECT_EQ(format_preview(0.0, 0), "0");
}

// The reason this function exists rather than a bare "%.4f".
TEST(Format, PreviewNeverRoundsANonZeroToZero) {
    EXPECT_NE(format_preview(1e-9), "0");
    EXPECT_NE(format_preview(-1e-9), "0");
    EXPECT_NE(format_preview(0.00001), "0");
    EXPECT_EQ(read_back(format_preview(1e-9)), 1e-9);
    EXPECT_EQ(read_back(format_preview(-1e-9)), -1e-9);
    // Large magnitudes go to the round-tripping form rather than a wall of digits.
    EXPECT_EQ(read_back(format_preview(1e300)), 1e300);
    // Zero really is zero.
    EXPECT_EQ(format_preview(0.0), "0");
}

} // namespace
