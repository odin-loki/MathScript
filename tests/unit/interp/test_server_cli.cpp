#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#ifndef MATHSCRIPT_SERVER_PATH
#error "MATHSCRIPT_SERVER_PATH must be defined by CMake"
#endif

namespace {

// POSIX std::system returns a wait status; decode it so exit 1 is not compared as 256.
int exit_code(int status) {
#ifdef _WIN32
    return status;
#else
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return status;
#endif
}

int run_server(const std::string& args) {
#ifdef _WIN32
    // cmd.exe strips the outermost pair of quotes when the line both begins with
    // a quote and contains more of them, which mangles an argument like
    // -e "x = 6" -- the two cases here that carry quotes were the only two of
    // seventeen that failed on Windows. Wrapping the whole line in one further
    // pair is the documented way to keep the inner quoting intact.
    const std::string cmd = std::string("\"\"") + MATHSCRIPT_SERVER_PATH + "\" " + args + " <nul >nul 2>&1\"";
#else
    const std::string cmd = std::string("\"") + MATHSCRIPT_SERVER_PATH + "\" " + args + " </dev/null >/dev/null 2>&1";
#endif
    return exit_code(std::system(cmd.c_str()));
}

// Runs the server and returns its combined stdout/stderr.
std::string capture_server(const std::string& args, const std::string& stdin_path = "") {
    const std::string out_path = "server_cli_capture.txt";
#ifdef _WIN32
    const std::string in_redirect = stdin_path.empty() ? " <nul" : (" <" + stdin_path);
#else
    const std::string in_redirect = stdin_path.empty() ? " </dev/null" : (" <" + stdin_path);
#endif
#ifdef _WIN32
    // Same cmd.exe quote stripping as in run_server.
    const std::string cmd = std::string("\"\"") + MATHSCRIPT_SERVER_PATH + "\" " + args
        + in_redirect + " >" + out_path + " 2>&1\"";
#else
    const std::string cmd = std::string("\"") + MATHSCRIPT_SERVER_PATH + "\" " + args
        + in_redirect + " >" + out_path + " 2>&1";
#endif
    (void)std::system(cmd.c_str());

    std::ifstream in(out_path);
    std::ostringstream buf;
    buf << in.rdbuf();
    in.close();
    std::filesystem::remove(out_path);
    return buf.str();
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

TEST(ServerCliTest, version_reports_binary_name) {
    const std::string out = capture_server("--version");
    EXPECT_NE(out.find("mathscript-server"), std::string::npos);
}

TEST(ServerCliTest, help_lists_command_sources) {
    const std::string out = capture_server("--help");
    EXPECT_NE(out.find("--script"), std::string::npos);
    EXPECT_NE(out.find("--serve"), std::string::npos);
    EXPECT_NE(out.find("--eval"), std::string::npos);
}

TEST(ServerCliTest, unknown_option_exits_nonzero) {
    EXPECT_NE(run_server("--definitely-not-an-option"), 0);
}

TEST(ServerCliTest, eval_requires_an_argument) {
    EXPECT_NE(run_server("--eval"), 0);
}

TEST(ServerCliTest, script_requires_a_path) {
    EXPECT_NE(run_server("--script"), 0);
}

TEST(ServerCliTest, banner_reports_rank_and_backend) {
    const std::string out = capture_server("");
    EXPECT_NE(out.find("rank 0/1"), std::string::npos);
    EXPECT_NE(out.find("backend="), std::string::npos);
    EXPECT_NE(out.find("cluster ready (1 ranks)"), std::string::npos);
}

TEST(ServerCliTest, quiet_suppresses_the_banner) {
    const std::string out = capture_server("--quiet");
    EXPECT_EQ(out.find("cluster ready"), std::string::npos);
}

TEST(ServerCliTest, eval_executes_on_the_node) {
    const std::string out = capture_server("--quiet -e \"x = 6\" -e \"y = x * 7\"");
    EXPECT_NE(out.find("42"), std::string::npos);
}

TEST(ServerCliTest, eval_reports_errors_without_failing_the_node) {
    const std::string out = capture_server("--quiet -e \"nosuch(1.0)\"");
    EXPECT_NE(out.find("error"), std::string::npos);
    EXPECT_EQ(run_server("--quiet -e \"nosuch(1.0)\""), 0);
}

TEST(ServerCliTest, script_runs_every_line) {
    const std::string filename = "server_cli_script.ms";
    {
        std::ofstream out(filename);
        out << "# a comment line\n";
        out << "a = 3\n";
        out << "b = a + 39\n";
    }
    const std::string out = capture_server("--quiet --script " + filename);
    std::filesystem::remove(filename);
    EXPECT_NE(out.find("42"), std::string::npos);
}

TEST(ServerCliTest, missing_script_is_reported_not_fatal) {
    const std::string out = capture_server("--quiet --script no_such_script_file.ms");
    EXPECT_NE(out.find("cannot open script"), std::string::npos);
    EXPECT_EQ(run_server("--quiet --script no_such_script_file.ms"), 0);
}

TEST(ServerCliTest, serve_reads_commands_from_stdin) {
    const std::string filename = "server_cli_stdin.txt";
    {
        std::ofstream out(filename);
        out << "p = 40\n";
        out << "q = p + 2\n";
    }
    const std::string out = capture_server("--quiet --serve", filename);
    std::filesystem::remove(filename);
    EXPECT_NE(out.find("42"), std::string::npos);
}

TEST(ServerCliTest, serve_stops_on_exit_command) {
    const std::string filename = "server_cli_exit.txt";
    {
        std::ofstream out(filename);
        out << "m = 1\n";
        out << "exit\n";
        out << "m = 999\n";
    }
    const std::string out = capture_server("--quiet --serve", filename);
    std::filesystem::remove(filename);
    EXPECT_EQ(out.find("999"), std::string::npos);
}

TEST(ServerCliTest, piped_stdin_serves_without_an_explicit_flag) {
    const std::string filename = "server_cli_piped.txt";
    {
        std::ofstream out(filename);
        out << "z = 21\n";
        out << "w = z + 21\n";
    }
    const std::string out = capture_server("--quiet", filename);
    std::filesystem::remove(filename);
    EXPECT_NE(out.find("42"), std::string::npos);
}
