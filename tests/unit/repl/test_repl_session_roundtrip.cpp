// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// A saved session has to reload as the session that was saved.
//
// It did not. save_session wrote every double with the default ostream format, which
// is %g at six significant digits, so `x = 1.23456789` was written out as `1.23457`
// and came back about 2.1e-06 wrong. Nothing reported the loss: the save succeeded,
// the load succeeded, and the number was simply a different number. The same six
// digits truncated every matrix entry and every plot sample.
//
// Six digits is a reasonable thing to show a person. It is not a reasonable thing to
// write to a file, because a file is read back. Saving now uses ms::format_exact,
// which emits the fewest digits that strtod maps to the same double -- so the text
// stays short for values that are short, and grows only where it must.
//
// These tests compare bit patterns, not printed forms, because printed forms are
// exactly what was hiding the defect.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

// Each test gets its own file; the name is fixed so a failure leaves something to look
// at rather than a random path.
class SessionFile {
public:
    explicit SessionFile(const char* stem)
        : path_((std::filesystem::temp_directory_path() / (std::string("ms_session_") + stem + ".mss")).string()) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    ~SessionFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    const std::string& path() const { return path_; }
    std::string text() const {
        std::ifstream in(path_);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

private:
    std::string path_;
};

TEST(SessionRoundTrip, ScalarsComeBackBitForBit) {
    SessionFile file("scalars");
    Interpreter saver;
    // A value with more significant digits than %g shows, one below the six-decimal
    // display floor, one above the range where %f stays readable, and one whose
    // shortest round-trip spelling needs all seventeen digits.
    expect_ok(saver, "x = 1.23456789");
    expect_ok(saver, "tiny = 0.000000001");
    expect_ok(saver, "huge = 12345678901234567890");
    expect_ok(saver, "third = 0.1");
    expect_ok(saver, "save " + file.path());

    Interpreter loader;
    expect_ok(loader, "load " + file.path());
    const auto& out = loader.state().scalars;
    ASSERT_EQ(out.count("x"), 1U);
    EXPECT_EQ(out.at("x"), 1.23456789);
    EXPECT_EQ(out.at("tiny"), 1e-09);
    EXPECT_EQ(out.at("huge"), 12345678901234567890.0);
    EXPECT_EQ(out.at("third"), 0.1);

    // The saved text is the record, so check it directly too: 1.23457 is what the
    // defect wrote, and it must not be what is in the file.
    const std::string saved = file.text();
    EXPECT_NE(saved.find("1.23456789"), std::string::npos) << saved;
    EXPECT_EQ(saved.find("1.23457\n"), std::string::npos) << saved;
}

// Short values must stay short. A file full of 1.0000000000000000 for every 1 would
// round-trip correctly and still be a regression in everything else.
TEST(SessionRoundTrip, RoundValuesStaySpelledPlainly) {
    SessionFile file("plain");
    Interpreter saver;
    expect_ok(saver, "a = 1");
    expect_ok(saver, "b = 2.5");
    expect_ok(saver, "save " + file.path());
    const std::string saved = file.text();
    EXPECT_NE(saved.find("scalar a = 1\n"), std::string::npos) << saved;
    EXPECT_NE(saved.find("scalar b = 2.5\n"), std::string::npos) << saved;
}

TEST(SessionRoundTrip, MatrixEntriesComeBackBitForBit) {
    SessionFile file("matrix");
    Interpreter saver;
    expect_ok(saver, "A = [1.23456789, 0.000000001; 98765.4321987, 0.1]");
    expect_ok(saver, "save " + file.path());

    Interpreter loader;
    expect_ok(loader, "load " + file.path());
    ASSERT_EQ(loader.state().matrices.count("A"), 1U);
    const auto& a = loader.state().matrices.at("A");
    ASSERT_EQ(a.rows(), 2U);
    ASSERT_EQ(a.cols(), 2U);
    EXPECT_EQ(a(0, 0), 1.23456789);
    EXPECT_EQ(a(0, 1), 1e-09);
    EXPECT_EQ(a(1, 0), 98765.4321987);
    EXPECT_EQ(a(1, 1), 0.1);
}

TEST(SessionRoundTrip, PlotSamplesComeBackBitForBit) {
    SessionFile file("plot");
    Interpreter saver;
    expect_ok(saver, "plot([0.000000001, 1.23456789, 2.5])");
    ASSERT_TRUE(saver.state().plot.valid);
    const std::vector<double> before = saver.state().plot.y;
    expect_ok(saver, "save " + file.path());

    Interpreter loader;
    expect_ok(loader, "load " + file.path());
    ASSERT_TRUE(loader.state().plot.valid);
    EXPECT_EQ(loader.state().plot.y, before);
}

// Every double a session can hold, not only the ones that look like data. Saving is
// the one place where "we never see values like that" is not an argument, because the
// values came from the user in the first place. Each literal below is the shortest
// decimal that reads back as the named double, so the assignment sets the exact bit
// pattern the C++ expression names -- the test asserts that, then asserts the file
// preserves it.
TEST(SessionRoundTrip, AwkwardMagnitudesSurvive) {
    struct Case {
        const char* literal;
        double value;
    };
    const std::vector<Case> cases{
        {"0", 0.0},
        {"-0.0", -0.0},
        {"0.3333333333333333", 1.0 / 3.0},
        {"1.0000000000000002", std::nextafter(1.0, 2.0)},
        {"2.2250738585072014e-308", std::numeric_limits<double>::min()},
        {"5e-324", std::numeric_limits<double>::denorm_min()},
        {"1.7976931348623157e+308", std::numeric_limits<double>::max()},
        {"-1.7976931348623157e+308", -std::numeric_limits<double>::max()},
        {"1e-300", 1e-300},
        {"1e300", 1e300},
    };
    for (std::size_t i = 0; i < cases.size(); ++i) {
        SessionFile file(("awkward" + std::to_string(i)).c_str());
        Interpreter saver;
        expect_ok(saver, std::string("v = ") + cases[i].literal);
        ASSERT_EQ(saver.state().scalars.count("v"), 1U) << cases[i].literal;
        // The literal really is the double we mean, sign of zero included.
        const double set = saver.state().scalars.at("v");
        ASSERT_EQ(std::memcmp(&set, &cases[i].value, sizeof(double)), 0) << cases[i].literal;
        expect_ok(saver, "save " + file.path());

        Interpreter loader;
        expect_ok(loader, "load " + file.path());
        ASSERT_EQ(loader.state().scalars.count("v"), 1U) << cases[i].literal;
        const double got = loader.state().scalars.at("v");
        EXPECT_EQ(std::memcmp(&got, &cases[i].value, sizeof(double)), 0)
            << cases[i].literal << " saved as: " << file.text();
    }
}

// A matrix with no elements still has a shape, and the bracket form cannot carry one.
// A 3x0 matrix used to be written "[; ; ]", which parse_matrix rejects outright -- so
// saving a session containing one produced a file that would not load. A 0x3 matrix was
// written "[]" and came back 0x0, silently a different matrix.
//
// mat_submatrix is how such a matrix arises in practice: a zero row or column count is
// a legitimate block request, and it is what the audit reached this through.
TEST(SessionRoundTrip, EmptyMatricesKeepTheirShape) {
    struct Shape {
        const char* block;
        std::size_t rows;
        std::size_t cols;
    };
    const std::vector<Shape> shapes{
        {"mat_submatrix(A, 0, 0, 3, 0)", 3, 0},
        {"mat_submatrix(A, 0, 0, 0, 3)", 0, 3},
        {"mat_submatrix(A, 0, 0, 0, 0)", 0, 0},
        {"mat_submatrix(A, 0, 0, 1, 0)", 1, 0},
    };
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        SessionFile file(("empty" + std::to_string(i)).c_str());
        Interpreter saver;
        expect_ok(saver, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
        expect_ok(saver, std::string("E = ") + shapes[i].block);
        ASSERT_EQ(saver.state().matrices.count("E"), 1U) << shapes[i].block;
        ASSERT_EQ(saver.state().matrices.at("E").rows(), shapes[i].rows) << shapes[i].block;
        ASSERT_EQ(saver.state().matrices.at("E").cols(), shapes[i].cols) << shapes[i].block;
        expect_ok(saver, "save " + file.path());

        Interpreter loader;
        const auto loaded = loader.execute("load " + file.path());
        ASSERT_TRUE(loaded.has_value())
            << shapes[i].block << " wrote: " << file.text();
        ASSERT_EQ(loader.state().matrices.count("E"), 1U) << shapes[i].block;
        EXPECT_EQ(loader.state().matrices.at("E").rows(), shapes[i].rows) << file.text();
        EXPECT_EQ(loader.state().matrices.at("E").cols(), shapes[i].cols) << file.text();
    }
}

// The shape-only spelling is readable back from a user as well as from a file: it is
// the only way to write down an empty matrix that has columns.
TEST(SessionRoundTrip, TheShapeOnlySpellingIsAcceptedFromAUser) {
    Interpreter interp;
    expect_ok(interp, "E = [0x3]");
    ASSERT_EQ(interp.state().matrices.count("E"), 1U);
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 0U);
    EXPECT_EQ(interp.state().matrices.at("E").cols(), 3U);
    expect_ok(interp, "F = [4x0]");
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 4U);
    EXPECT_EQ(interp.state().matrices.at("F").cols(), 0U);
    // It is only for empty shapes: [2x2] would name four elements it does not carry.
    expect_error(interp, "G = [2x2]");
}

} // namespace
