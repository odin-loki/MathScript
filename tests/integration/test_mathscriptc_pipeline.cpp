#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#ifdef _WIN32
#include <process.h>
#endif

#ifndef MATHSCRIPTC_PATH
#error "MATHSCRIPTC_PATH must be defined by CMake"
#endif

namespace {

std::filesystem::path make_script_path(const char* name) {
    return std::filesystem::current_path() / name;
}

bool write_script(const std::filesystem::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out << body;
    out.flush();
    return out.good();
}

int run_mathscriptc(const std::filesystem::path& script_path) {
    const std::string exe = MATHSCRIPTC_PATH;
    const std::string script = script_path.generic_string();
#ifdef _WIN32
    return static_cast<int>(_spawnl(_P_WAIT, exe.c_str(), exe.c_str(), script.c_str(), nullptr));
#else
    const std::string cmd = std::string("\"") + exe + "\" \"" + script + "\" >/dev/null 2>&1";
    return std::system(cmd.c_str());
#endif
}

} // namespace

TEST(MathscriptcPipeline, multiline_script_with_matrix_and_scalar) {
    const auto script_path = make_script_path("mathscript_integration_script.ms");

    ASSERT_TRUE(write_script(script_path,
                             "# test script\n"
                             "A = [4, 2; 2, 3]\n"
                             "d = det(A)\n"
                             "x = sqrt(d)\n"));
    ASSERT_TRUE(std::filesystem::exists(script_path));

    EXPECT_EQ(run_mathscriptc(script_path), 0);

    std::filesystem::remove(script_path);
}

TEST(MathscriptcPipeline, script_error_continues_processing) {
    const auto script_path = make_script_path("mathscript_integration_error_script.ms");

    ASSERT_TRUE(write_script(script_path,
                             "a = 1\n"
                             "b = nosuch(1)\n"
                             "c = sqrt(4)\n"));
    ASSERT_TRUE(std::filesystem::exists(script_path));

    // mathscriptc continues all lines but exits non-zero because b= failed
    EXPECT_NE(run_mathscriptc(script_path), 0);

    std::filesystem::remove(script_path);
}
