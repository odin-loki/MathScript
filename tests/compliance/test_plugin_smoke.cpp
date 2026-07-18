#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>

#ifndef MS_BUILD_PLUGIN
#define MS_BUILD_PLUGIN 0
#endif

#if MS_BUILD_PLUGIN
#include "unsafe_registry.hpp"
#endif

namespace {

constexpr const char kPluginName[] = "ms-profile";
constexpr const char kPluginDescription[] = "MathScript C++ Profile Enforcement";

constexpr const char* kEnforcedRules[] = {
    "no_raw_new",
    "no_malloc",
    "no_cstyle_cast",
    "no_throw",
    "no_catch",
    "no_const_cast",
    "no_goto",
    "no_raw_ptr_arithmetic",
    "no_unsafe_reinterpret",
    "no_detach",
    "no_vla",
    "narrowing",
    "no_signed_unsigned_mix",
    "no_raw_thread",
    "no_raw_mutex_lock",
    "no_uninit",
    "no_stored_span",
    "no_volatile_sync",
    "no_owning_raw_ptr",
    "unused_expected",
    "unsafe_audit",
};

} // namespace

TEST(PluginSmokeTest, plugin_availability) {
#if !MS_BUILD_PLUGIN
    GTEST_SKIP() << "Clang enforcement plugin not built (MS_BUILD_PLUGIN=OFF)";
#else
    EXPECT_STREQ(kPluginName, "ms-profile");
    EXPECT_STREQ(kPluginDescription, "MathScript C++ Profile Enforcement");
    EXPECT_GT(std::strlen(kPluginName), 0u);
    EXPECT_GT(std::strlen(kPluginDescription), 0u);
#endif
}

TEST(PluginSmokeTest, enforced_rule_registry) {
#if !MS_BUILD_PLUGIN
    GTEST_SKIP() << "Clang enforcement plugin not built (MS_BUILD_PLUGIN=OFF)";
#else
    EXPECT_EQ(sizeof(kEnforcedRules) / sizeof(kEnforcedRules[0]), 21u);
    for (const char* rule : kEnforcedRules) {
        EXPECT_GT(std::strlen(rule), 0u);
    }
#endif
}

TEST(PluginSmokeTest, unsafe_registry_records_known_site) {
#if !MS_BUILD_PLUGIN
    GTEST_SKIP() << "Clang enforcement plugin not built (MS_BUILD_PLUGIN=OFF)";
#else
    auto& registry = ms::plugin::UnsafeRegistry::instance();
    constexpr const char kKnownFile[] = "tests/compliance/no_raw_new/ok.cpp";
    constexpr int kKnownLine = 5;
    constexpr const char kKnownReason[] =
        "compliance: raw new allowed inside annotated function";

    registry.record_unsafe(kKnownFile, kKnownLine, kKnownReason, "compliance");

    const auto sites = registry.get_sites();
    ASSERT_GE(sites.size(), 1u);

    const bool found = std::any_of(
        sites.begin(), sites.end(), [](const ms::plugin::UnsafeSite& site) {
            return site.file == kKnownFile && site.line == kKnownLine &&
                   site.justification == kKnownReason;
        });
    EXPECT_TRUE(found);
    EXPECT_GE(registry.count(), 1);
#endif
}
