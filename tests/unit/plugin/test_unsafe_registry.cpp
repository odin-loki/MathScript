// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Tests for the unsafe-site registry.
//
// `src/plugin` was 1,189 lines with no unit tests. The compliance suite exercises
// the AST rules end to end by compiling files that must fail, which is the right
// test for a rule -- but it says nothing about this file, and this file is the one
// that decides which unsafe sites are on the record. A site that silently fails to
// register is a site nobody reviews.
//
// It also had a structural reason for going untested: it was built only when
// MS_BUILD_PLUGIN was on, so any test would have run in one optional CI job. It
// includes no Clang or LLVM header, so that gate has been removed and these run
// everywhere.
//
// The interesting surface is the round trip. Sites are appended to a JSONL file,
// re-parsed on the next call, merged, deduplicated and written out as JSON -- so a
// justification has to survive being escaped and unescaped, and the parser has to
// be unfooled by a justification that contains text looking like the fields around
// it.

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "unsafe_registry.hpp"

namespace {

using ms::plugin::UnsafeRegistry;
using ms::plugin::UnsafeSite;

// A unique directory per test, removed afterwards. The error_code overloads are
// deliberate: this tree is built with -fno-exceptions, so the throwing
// std::filesystem overloads would terminate rather than report.
class TempDir {
public:
    explicit TempDir(const std::string& tag) {
        std::error_code ec;
        auto base = std::filesystem::temp_directory_path(ec);
        if (ec) {
            base = std::filesystem::path(".");
        }
        path_ = base / ("ms_unsafe_registry_" + tag);
        std::filesystem::remove_all(path_, ec);
        std::filesystem::create_directories(path_, ec);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::string str() const { return path_.string(); }
    std::string file(const char* name) const { return (path_ / name).string(); }

private:
    std::filesystem::path path_;
};

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

UnsafeSite site(std::string file, int line, std::string why, std::string cat = "") {
    return UnsafeSite{std::move(file), line, std::move(why), std::move(cat)};
}

} // namespace

TEST(UnsafeRegistry, RefusesToRecordASiteItCouldNotReportOn) {
    auto& reg = UnsafeRegistry::instance();
    const int before = reg.count();

    // Each of these would produce an audit entry that cannot be looked up: no file
    // to open, no line to jump to, or no stated reason -- which is the entire point
    // of the registry.
    reg.record_unsafe("", 12, "SIMD: masked load", "");
    reg.record_unsafe("a.cpp", 0, "SIMD: masked load", "");
    reg.record_unsafe("a.cpp", -1, "SIMD: masked load", "");
    reg.record_unsafe("a.cpp", 12, "", "");

    EXPECT_EQ(reg.count(), before);
}

TEST(UnsafeRegistry, RecordsASiteOnceHoweverManyTimesItIsSeen) {
    auto& reg = UnsafeRegistry::instance();
    const int before = reg.count();

    // A header included by two hundred translation units reports its unsafe sites
    // once per inclusion. Counting those separately would inflate the audit by two
    // orders of magnitude.
    for (int i = 0; i < 5; ++i) {
        reg.record_unsafe("dedupe_me.cpp", 42, "CUDA interop: device pointer", "");
    }
    EXPECT_EQ(reg.count(), before + 1);
}

TEST(UnsafeRegistry, CategoryComesFromTheJustificationPrefixWhenNotGiven) {
    auto& reg = UnsafeRegistry::instance();
    reg.record_unsafe("cat_infer.cpp", 1, "SIMD intrinsics: unaligned load", "");
    reg.record_unsafe("cat_infer.cpp", 2, "no colon here at all", "");
    reg.record_unsafe("cat_infer.cpp", 3, ": leading colon", "");
    reg.record_unsafe("cat_infer.cpp", 4, "explicit", "Given");

    std::string simd, none, leading, given;
    for (const auto& s : reg.get_sites()) {
        if (s.file != "cat_infer.cpp") {
            continue;
        }
        if (s.line == 1) simd = s.category;
        if (s.line == 2) none = s.category;
        if (s.line == 3) leading = s.category;
        if (s.line == 4) given = s.category;
    }
    EXPECT_EQ(simd, "SIMD intrinsics");
    EXPECT_EQ(none, "general");
    // A justification that opens with a colon has no prefix to take.
    EXPECT_EQ(leading, "general");
    EXPECT_EQ(given, "Given") << "an explicit category must not be overwritten";
}

TEST(UnsafeRegistry, SitesComeBackOrderedByFileThenLine) {
    auto& reg = UnsafeRegistry::instance();
    reg.record_unsafe("zz_order.cpp", 30, "x: three", "");
    reg.record_unsafe("zz_order.cpp", 10, "x: one", "");
    reg.record_unsafe("zz_order.cpp", 20, "x: two", "");

    std::vector<int> lines;
    for (const auto& s : reg.get_sites()) {
        if (s.file == "zz_order.cpp") {
            lines.push_back(s.line);
        }
    }
    ASSERT_EQ(lines.size(), 3U);
    EXPECT_EQ(lines[0], 10);
    EXPECT_EQ(lines[1], 20);
    EXPECT_EQ(lines[2], 30);
}

TEST(UnsafeRegistry, EmitDeclinesWhenThereIsNowhereOrNothingToWrite) {
    const TempDir dir("empty_inputs");
    EXPECT_FALSE(UnsafeRegistry::emit_audit_report("", {site("a.cpp", 1, "x: y")}));
    EXPECT_FALSE(UnsafeRegistry::emit_audit_report(dir.str(), {}));
}

TEST(UnsafeRegistry, WritesBothReportsAndTheJsonCarriesEverySite) {
    const TempDir dir("basic_emit");
    const std::vector<UnsafeSite> sites{
        site("alpha.cpp", 10, "SIMD: aligned load", "SIMD"),
        site("beta.cpp", 20, "CUDA: device copy", "CUDA"),
    };
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(dir.str(), sites));

    const std::string json = read_file(dir.file("ms-unsafe-audit.json"));
    ASSERT_FALSE(json.empty());
    EXPECT_TRUE(contains(json, "\"total\": 2"));
    EXPECT_TRUE(contains(json, "alpha.cpp"));
    EXPECT_TRUE(contains(json, "beta.cpp"));
    EXPECT_TRUE(contains(json, "SIMD: aligned load"));
    EXPECT_FALSE(read_file(dir.file("ms-unsafe-audit.jsonl")).empty());
}

TEST(UnsafeRegistry, AppendingTheSameSiteTwiceStillReportsItOnce) {
    const TempDir dir("merge_dedupe");
    const std::vector<UnsafeSite> sites{site("same.cpp", 7, "SIMD: one", "SIMD")};

    // Two compiler invocations over the same header: the JSONL grows, the report
    // must not.
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(dir.str(), sites));
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(dir.str(), sites));

    const std::string json = read_file(dir.file("ms-unsafe-audit.json"));
    EXPECT_TRUE(contains(json, "\"total\": 1")) << json;
}

TEST(UnsafeRegistry, JustificationsSurviveTheEscapeAndUnescapeRoundTrip) {
    const TempDir dir("round_trip");
    // Every character the writer escapes, plus one it does not, in one string.
    const std::string awkward =
        "quote \" backslash \\ newline \n tab \t carriage \r end";
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(
        dir.str(), {site("awkward.cpp", 3, "Cat: " + awkward, "Cat")}));

    // The first pass wrote it; a second pass has to read its own output back, which
    // is where an asymmetric escape/unescape pair shows up.
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(
        dir.str(), {site("other.cpp", 4, "Other: plain", "Other")}));

    const std::string json = read_file(dir.file("ms-unsafe-audit.json"));
    EXPECT_TRUE(contains(json, "\"total\": 2"))
        << "a mangled re-parse would drop or duplicate the awkward site:\n" << json;
    // Written escaped, not raw: a literal newline would break the JSONL format the
    // next pass parses line by line.
    EXPECT_TRUE(contains(json, "\\n"));
    EXPECT_TRUE(contains(json, "\\t"));
    EXPECT_TRUE(contains(json, "\\\""));
    EXPECT_FALSE(contains(json, "quote \" backslash"))
        << "the quote reached the file unescaped";
}

TEST(UnsafeRegistry, AJustificationCannotImpersonateTheFieldsAroundIt) {
    const TempDir dir("injection");
    // The parser finds each field by searching for its key. A justification holding
    // that key verbatim sits earlier in the line than the real field, so nothing but
    // the escaping stops it being read instead. This is the test that says so.
    const std::string hostile =
        R"(Evil: ","category":"forged","file":"wrong.cpp","line":999,")";
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(
        dir.str(), {site("victim.cpp", 5, hostile, "Evil")}));
    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(
        dir.str(), {site("second.cpp", 6, "Plain: nothing odd", "Plain")}));

    const std::string json = read_file(dir.file("ms-unsafe-audit.json"));
    EXPECT_TRUE(contains(json, "victim.cpp"));
    EXPECT_FALSE(contains(json, "\"file\":\"wrong.cpp\""))
        << "the justification was parsed as a file field:\n" << json;
    EXPECT_FALSE(contains(json, "\"category\":\"forged\""))
        << "the justification was parsed as a category field:\n" << json;
}

TEST(UnsafeRegistry, AMalformedLineIsDroppedRatherThanEndingTheProcess) {
    const TempDir dir("malformed");
    {
        // Hand-written JSONL of the kind a truncated or concurrent write leaves
        // behind. std::stoi on a non-numeric line number used to abort here, because
        // this tree is built with -fno-exceptions.
        std::ofstream out(dir.file("ms-unsafe-audit.jsonl"));
        out << R"({"file":"good.cpp","line":11,"justification":"A: ok","category":"A"})" << '\n';
        out << R"({"file":"bad.cpp","line":not-a-number,"justification":"B: x","category":"B"})" << '\n';
        out << R"({"file":"huge.cpp","line":99999999999999999999,"justification":"C: x","category":"C"})" << '\n';
        out << R"({"file":"","line":13,"justification":"D: no file","category":"D"})" << '\n';
        out << "not json at all\n";
        out << '\n';
    }

    ASSERT_TRUE(UnsafeRegistry::emit_audit_report(
        dir.str(), {site("added.cpp", 14, "E: added", "E")}));

    const std::string json = read_file(dir.file("ms-unsafe-audit.json"));
    EXPECT_TRUE(contains(json, "good.cpp")) << "a well-formed line was lost";
    EXPECT_TRUE(contains(json, "added.cpp"));
    EXPECT_FALSE(contains(json, "bad.cpp")) << "a non-numeric line number was accepted";
    EXPECT_FALSE(contains(json, "huge.cpp")) << "an out-of-range line number was accepted";
    EXPECT_FALSE(contains(json, "\"file\":\"\"")) << "a site with no file was kept";
}

TEST(UnsafeRegistry, TheMacroEntryPointToleratesNullArguments) {
    auto& reg = UnsafeRegistry::instance();
    const int before = reg.count();
    ms::plugin::record_unsafe_annotation(nullptr, 1, "x: y");
    ms::plugin::record_unsafe_annotation("f.cpp", 1, nullptr);
    EXPECT_EQ(reg.count(), before);

    ms::plugin::record_unsafe_annotation("macro_path.cpp", 99, "Macro: recorded");
    EXPECT_EQ(reg.count(), before + 1);
}
