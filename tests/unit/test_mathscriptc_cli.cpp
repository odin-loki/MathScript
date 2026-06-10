#include <cstdlib>
#include <filesystem>
#include <fstream>
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

TEST(MathscriptcCliTest, script_file_exits_zero) {
    const std::filesystem::path exe_path = MATHSCRIPTC_PATH;
    const auto path = exe_path.parent_path() / "mathscriptc_test_script.ms";
    {
        std::ofstream out(path.string(), std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        out << "x = 42\n";
        out << "y = x + 1\n";
        out.close();
    }

    const std::string script_path = path.generic_string();
#ifdef _WIN32
    const std::string cmd = "cd /d \"" + exe_path.parent_path().generic_string()
        + "\" && mathscriptc.exe mathscriptc_test_script.ms >nul 2>&1";
#else
    const std::string cmd =
        std::string("\"") + MATHSCRIPTC_PATH + "\" \"" + script_path + "\" >/dev/null 2>&1";
#endif
    const int rc = std::system(cmd.c_str());
    EXPECT_EQ(rc, 0);

    std::filesystem::remove(path);
}
