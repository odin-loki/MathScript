// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The argument spellings nothing ever passed.
//
// A coverage sweep over the two interpreter translation units found that of 1,384
// distinct string literals the dispatcher compares against, exactly eight had never
// been reached by any test: the plot kinds `bar`, `heatmap`, `surface` and
// `surface3d`, and the filter types `lowpass`, `low`, `highpass` and `high`.
//
// None of them is an obscure corner. They are the documented word forms of
// arguments the tests only ever passed in their other form -- filter type as 0 or 1,
// plot kind never round-tripped through a session file at all. A user following the
// documentation would have been the first to execute this code.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

std::filesystem::path temp_session(const char* tag) {
    std::error_code ec;
    auto dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        dir = std::filesystem::path(".");
    }
    return dir / (std::string("ms_string_arg_") + tag + ".mss");
}

// Saves a real session, rewrites the recorded plot kind, and loads it back. Patching
// the file rather than hand-writing one keeps every other field in the format the
// writer actually produces, so the test cannot drift out of step with it.
PlotSeries::Kind round_trip_kind(const char* tag, const std::string& kind_word,
                                 bool* loaded) {
    const auto path = temp_session(tag);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    Interpreter writer;
    EXPECT_TRUE(writer.execute("plot([1, 2, 3])").has_value());
    EXPECT_TRUE(writer.save_session(path.string()).has_value());

    std::string text;
    {
        std::ifstream in(path);
        text.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    const auto pos = text.find("kind=");
    EXPECT_NE(pos, std::string::npos) << "session file records no plot kind";
    if (pos != std::string::npos) {
        const auto end = text.find_first_of(" \n", pos + 5);
        text = text.substr(0, pos + 5) + kind_word + text.substr(end);
        std::ofstream out(path, std::ios::trunc);
        out << text;
    }

    Interpreter reader;
    const auto r = reader.load_session(path.string());
    *loaded = r.has_value();
    const PlotSeries::Kind kind = reader.plot().kind;
    std::filesystem::remove(path, ec);
    return kind;
}

} // namespace

TEST(ReplStringArgForms, SessionLoadUnderstandsEveryPlotKindItCanWrite) {
    struct Case {
        const char* tag;
        const char* word;
        PlotSeries::Kind expected;
    };
    // Every spelling parse_plot_kind accepts, including the two names for the same
    // kind. A kind the writer can emit but the reader cannot parse would silently
    // downgrade a saved plot to a line chart on reload.
    const Case cases[] = {
        {"line", "line", PlotSeries::Kind::Line},
        {"bar", "bar", PlotSeries::Kind::Bar},
        {"scatter", "scatter", PlotSeries::Kind::Scatter},
        {"heatmap", "heatmap", PlotSeries::Kind::Heatmap},
        {"spy", "spy", PlotSeries::Kind::Spy},
        {"surface3d", "surface3d", PlotSeries::Kind::Surface3D},
        {"surface", "surface", PlotSeries::Kind::Surface3D},
    };
    for (const auto& c : cases) {
        bool loaded = false;
        const auto got = round_trip_kind(c.tag, c.word, &loaded);
        EXPECT_TRUE(loaded) << "load_session failed for kind=" << c.word;
        EXPECT_EQ(static_cast<int>(got), static_cast<int>(c.expected))
            << "kind=" << c.word << " did not survive the round trip";
    }
}

TEST(ReplStringArgForms, SessionLoadIgnoresAKindItDoesNotRecognise) {
    // parse_plot_kind returns nullopt rather than a default, and the loader leaves
    // the pending kind alone. The contract worth pinning is that an unknown word is
    // not silently promoted to some other chart type.
    bool loaded = false;
    const auto got = round_trip_kind("unknown", "definitely-not-a-kind", &loaded);
    EXPECT_TRUE(loaded) << "an unrecognised plot kind should not fail the whole load";
    EXPECT_EQ(static_cast<int>(got), static_cast<int>(PlotSeries::Kind::Line));
}

TEST(ReplStringArgForms, ChebyshevFiltersAcceptTheWordFormsOfTheirType) {
    // signal_cheby1/2 take the filter type as 0/1 or as a word. Every test in the
    // tree passed a number, so the word branch had never run -- while the error
    // message the same function produces names only the words.
    Interpreter interp;
    for (const char* fn : {"signal_cheby1", "signal_cheby2"}) {
        for (const char* type : {"lowpass", "low", "highpass", "high"}) {
            const std::string cmd =
                std::string("ba = ") + fn + "(2, 40.0, 0.25, 2.0, " + type + ")";
            const auto r = interp.execute(cmd);
            EXPECT_TRUE(r.has_value())
                << cmd << " -> " << (r ? std::string{} : ms::format_error(r.error()));
        }
    }
}

TEST(ReplStringArgForms, ChebyshevFiltersRejectAnUnknownTypeWord) {
    Interpreter interp;
    for (const char* fn : {"signal_cheby1", "signal_cheby2"}) {
        const std::string cmd =
            std::string("ba = ") + fn + "(2, 40.0, 0.25, 2.0, bandpass)";
        const auto r = interp.execute(cmd);
        ASSERT_FALSE(r.has_value()) << cmd << " was accepted";
        const std::string message = ms::format_error(r.error());
        EXPECT_NE(message.find("lowpass"), std::string::npos)
            << "the rejection should name the accepted forms: " << message;
    }
}

TEST(ReplStringArgForms, NumericAndWordFilterTypesAgreeAndActuallyDiffer) {
    // Two properties, and the second is the one that matters. The word and numeric
    // spellings must select the same filter, or a script rewritten from 1 to
    // "highpass" would silently change its output. And lowpass must not equal
    // highpass, because if it did, both spellings would agree by virtue of the type
    // argument being ignored entirely -- which is exactly the bug this pair of
    // assertions is here to rule out.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("lo   = signal_cheby1(2, 40.0, 0.25, 2.0, lowpass)").has_value());
    ASSERT_TRUE(interp.execute("lon  = signal_cheby1(2, 40.0, 0.25, 2.0, 0)").has_value());
    ASSERT_TRUE(interp.execute("hi   = signal_cheby1(2, 40.0, 0.25, 2.0, highpass)").has_value());
    ASSERT_TRUE(interp.execute("hin  = signal_cheby1(2, 40.0, 0.25, 2.0, 1)").has_value());

    // The printed form opens with "<name> =", so compare from the first newline on:
    // otherwise the assertion fails on the variable names rather than the numbers.
    const auto body = [](const std::string& printed) {
        const auto nl = printed.find('\n');
        return nl == std::string::npos ? printed : printed.substr(nl);
    };

    const auto lo = interp.execute("lo");
    const auto lon = interp.execute("lon");
    const auto hi = interp.execute("hi");
    const auto hin = interp.execute("hin");
    ASSERT_TRUE(lo.has_value() && lon.has_value() && hi.has_value() && hin.has_value());

    EXPECT_EQ(body(*lo), body(*lon)) << "lowpass and 0 produced different coefficients";
    EXPECT_EQ(body(*hi), body(*hin)) << "highpass and 1 produced different coefficients";
    EXPECT_NE(body(*lo), body(*hi))
        << "lowpass and highpass produced identical coefficients, so the type "
           "argument is being ignored";
}
