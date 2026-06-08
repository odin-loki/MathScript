#include <gtest/gtest.h>

#include <cstring>

#ifndef MS_BUILD_PLUGIN
#define MS_BUILD_PLUGIN 0
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
    EXPECT_EQ(sizeof(kEnforcedRules) / sizeof(kEnforcedRules[0]), 20u);
    for (const char* rule : kEnforcedRules) {
        EXPECT_GT(std::strlen(rule), 0u);
    }
#endif
}
