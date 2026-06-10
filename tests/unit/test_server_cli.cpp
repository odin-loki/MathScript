#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

#ifndef MATHSCRIPT_SERVER_PATH
#error "MATHSCRIPT_SERVER_PATH must be defined by CMake"
#endif

namespace {

int run_server(const std::string& args) {
#ifdef _WIN32
    const std::string cmd = std::string("\"") + MATHSCRIPT_SERVER_PATH + "\" " + args + " >nul 2>&1";
#else
    const std::string cmd = std::string("\"") + MATHSCRIPT_SERVER_PATH + "\" " + args + " >/dev/null 2>&1";
#endif
    return std::system(cmd.c_str());
}

} // namespace

TEST(ServerCliTest, version_exits_zero) {
    EXPECT_EQ(run_server("--version"), 0);
}

TEST(ServerCliTest, help_exits_zero) {
    EXPECT_EQ(run_server("--help"), 0);
}

TEST(ServerCliTest, no_args_completes) {
    const int rc = run_server("");
    EXPECT_EQ(rc, 0);
}
