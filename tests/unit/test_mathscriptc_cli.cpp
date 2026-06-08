#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

#ifndef MATHSCRIPTC_PATH
#error "MATHSCRIPTC_PATH must be defined by CMake"
#endif

TEST(MathscriptcCliTest, version_exits_zero) {
#ifdef _WIN32
    const std::string cmd = std::string("\"") + MATHSCRIPTC_PATH + "\" --version >nul 2>&1";
#else
    const std::string cmd = std::string("\"") + MATHSCRIPTC_PATH + "\" --version >/dev/null 2>&1";
#endif
    const int rc = std::system(cmd.c_str());
    EXPECT_EQ(rc, 0);
}
