#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

#ifndef MATHSCRIPT_REPL_PATH
#error "MATHSCRIPT_REPL_PATH must be defined by CMake"
#endif

TEST(ReplCliTest, version_exits_zero) {
#ifdef _WIN32
    const std::string cmd = std::string("\"") + MATHSCRIPT_REPL_PATH + "\" --version >nul 2>&1";
#else
    const std::string cmd = std::string("\"") + MATHSCRIPT_REPL_PATH + "\" --version >/dev/null 2>&1";
#endif
    EXPECT_EQ(std::system(cmd.c_str()), 0);
}

TEST(ReplCliTest, eval_command_exits_zero) {
#ifdef _WIN32
    const std::string cmd =
        std::string("\"") + MATHSCRIPT_REPL_PATH + "\" -e x=42 >nul 2>&1";
#else
    const std::string cmd =
        std::string("\"") + MATHSCRIPT_REPL_PATH + "\" -e \"surf([1, 2; 3, 4])\" >/dev/null 2>&1";
#endif
    EXPECT_EQ(std::system(cmd.c_str()), 0);
}

TEST(ReplCliTest, eval_error_exits_nonzero) {
#ifdef _WIN32
    const std::string cmd =
        std::string("\"") + MATHSCRIPT_REPL_PATH + "\" -e \"not_a_command()\" >nul 2>&1";
#else
    const std::string cmd =
        std::string("\"") + MATHSCRIPT_REPL_PATH + "\" -e \"not_a_command()\" >/dev/null 2>&1";
#endif
    EXPECT_NE(std::system(cmd.c_str()), 0);
}
