// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The reproducibility manifest exists so that a result can be re-derived years
// later, which means the manifest itself has to be trustworthy. A field that
// silently reports a stale or default value is worse than an absent one: it
// converts "we don't know what this run did" into a confident wrong answer.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "../scoped_env.hpp"
#include "ms/runtime/repro.hpp"
#include "ms/simd/isa.hpp"
#include "ms/simd/simd.hpp"

namespace {

// Restores the global seed so an expectation that fails mid-test cannot leave a
// seed set for every test that runs after it.
class ScopedSeed {
public:
    ScopedSeed() = default;
    ~ScopedSeed() { ms::runtime::clear_global_seed(); }
    ScopedSeed(const ScopedSeed&) = delete;
    ScopedSeed& operator=(const ScopedSeed&) = delete;
};

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

TEST(ReproManifest, ReportsTheBuildIdentity) {
    const auto m = ms::runtime::capture();
    EXPECT_FALSE(m.version.empty()) << "a manifest without a version identifies nothing";
    // commit and build_date come from the CMake configure step and can legitimately
    // be empty in a tree built outside git, so they are not required to be set --
    // but the version always is.
}

TEST(ReproManifest, IsaFieldMatchesWhatTheLibraryWillActuallyUse) {
    const auto m = ms::runtime::capture();
    const auto info = ms::simd::dispatch_info();
    // The manifest must not report a wider ISA than dispatch resolved to: that is
    // precisely the claim a reader would rely on when comparing two runs.
    if (info.isa.avx512f) {
        EXPECT_TRUE(contains(m.isa, "AVX-512"));
    } else {
        EXPECT_FALSE(contains(m.isa, "AVX-512"));
    }
    if (info.isa.avx2) {
        EXPECT_TRUE(contains(m.isa, "AVX2"));
    } else {
        EXPECT_FALSE(contains(m.isa, "AVX2"));
    }
    EXPECT_EQ(m.batch_width, info.batch_width);
}

TEST(ReproManifest, DefaultSeedIsMarkedAsNotChosen) {
    ms::runtime::clear_global_seed();
    const auto m = ms::runtime::capture();
    EXPECT_FALSE(m.seed_explicit)
        << "an unset seed reported as chosen would invite someone to quote it";
    EXPECT_TRUE(contains(ms::runtime::to_text(m), "(default)"));
}

TEST(ReproManifest, ExplicitSeedIsReportedAndMarked) {
    const ScopedSeed restore;
    ms::runtime::set_global_seed(1234567890123ULL);
    const auto m = ms::runtime::capture();
    EXPECT_EQ(m.seed, 1234567890123ULL);
    EXPECT_TRUE(m.seed_explicit);
    EXPECT_TRUE(contains(ms::runtime::to_text(m), "1234567890123"));
    EXPECT_TRUE(contains(ms::runtime::to_text(m), "(set)"));
}

TEST(ReproManifest, ClearingTheSeedReturnsToTheDefault) {
    ms::runtime::set_global_seed(42);
    ASSERT_TRUE(ms::runtime::global_seed_is_explicit());
    ms::runtime::clear_global_seed();
    EXPECT_FALSE(ms::runtime::global_seed_is_explicit());
    EXPECT_NE(ms::runtime::global_seed(), 42U)
        << "clear must restore the default, not merely drop the explicit flag";
}

TEST(ReproManifest, RecordsAForcedIsaCeiling) {
    // A run under MS_FORCE_ISA is not comparable with one that was not, so the
    // ceiling has to appear in the manifest rather than being invisible.
    ms::runtime::ReproManifest forced;
    {
        const ms::testing::ScopedEnv env("MS_FORCE_ISA", "sse2");
        forced = ms::runtime::capture();
    }

    EXPECT_EQ(forced.forced_isa, "sse2");
    EXPECT_TRUE(contains(ms::runtime::to_text(forced), "sse2"));

    const auto unforced = ms::runtime::capture();
    EXPECT_TRUE(unforced.forced_isa.empty());
    EXPECT_TRUE(contains(ms::runtime::to_text(unforced), "(unset)"));
}

TEST(ReproManifest, JsonIsWellFormedAndCarriesEveryField) {
    const ScopedSeed restore;
    ms::runtime::set_global_seed(7);
    const std::string json = ms::runtime::to_json(ms::runtime::capture());

    ASSERT_FALSE(json.empty());
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(), '}');
    for (const char* key : {"version", "commit", "built", "isa", "forced_isa", "kernel",
                            "batch_width", "hardware_threads", "max_workers", "seed",
                            "seed_explicit"}) {
        EXPECT_TRUE(contains(json, std::string("\"") + key + "\":"))
            << "missing field: " << key;
    }
    EXPECT_TRUE(contains(json, "\"seed\":7"));
    EXPECT_TRUE(contains(json, "\"seed_explicit\":true"));

    // Quotes must balance, or a consumer parsing the manifest gets a broken record
    // rather than a missing one.
    int quotes = 0;
    for (std::size_t i = 0; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) {
            ++quotes;
        }
    }
    EXPECT_EQ(quotes % 2, 0) << "unbalanced quotes in: " << json;
}

TEST(ReproManifest, TextFormIsStableAcrossCallsWithNothingChanged) {
    const ScopedSeed restore;
    ms::runtime::set_global_seed(99);
    EXPECT_EQ(ms::runtime::to_text(ms::runtime::capture()),
              ms::runtime::to_text(ms::runtime::capture()))
        << "a manifest that varies between two adjacent calls cannot certify a run";
}
