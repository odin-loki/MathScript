#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#ifndef MATHSCRIPT_REPL_PATH
#error "MATHSCRIPT_REPL_PATH must be defined by CMake"
#endif

namespace {

int run_repl(const std::string& args) {
#ifdef _WIN32
    const std::string cmd = std::string("\"") + MATHSCRIPT_REPL_PATH + "\" " + args + " >nul 2>&1";
#else
    const std::string cmd = std::string("\"") + MATHSCRIPT_REPL_PATH + "\" " + args + " >/dev/null 2>&1";
#endif
    return std::system(cmd.c_str());
}

} // namespace

TEST(ReplCliTest, version_exits_zero) {
    EXPECT_EQ(run_repl("--version"), 0);
}

TEST(ReplCliTest, eval_command_exits_zero) {
#ifdef _WIN32
    EXPECT_EQ(run_repl("-e x=42"), 0);
#else
    EXPECT_EQ(run_repl("-e \"surf([1, 2; 3, 4])\""), 0);
#endif
}

TEST(ReplCliTest, eval_error_exits_nonzero) {
    // Unknown scalar function with a numeric argument (repl returns 1 on eval failure).
    EXPECT_NE(run_repl("-e \"nosuch(1.0)\""), 0);
}

TEST(ReplCliTest, eval_load_flag_exits_zero) {
    const std::string filename = "repl_load_cli_test.ms";
    {
        std::ofstream out(filename);
        out << "x = 1\n";
    }

    // -e exits after load; --load alone would enter interactive mode.
    const int status = run_repl("--load " + filename + " -e version");
    std::filesystem::remove(filename);
    EXPECT_EQ(status, 0);
}

TEST(ReplCliTest, debug_flag_exits_zero) {
    EXPECT_EQ(run_repl("--debug -e x=1"), 0);
}
