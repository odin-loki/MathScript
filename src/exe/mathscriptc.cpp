// MathScript script runner
// Executes .ms files as REPL command sequences

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/version.hpp"

namespace {

void print_usage(std::ostream& out) {
    out << "MathScript script runner (mathscriptc)\n\n"
        << "Usage:\n"
        << "  mathscriptc [options]\n"
        << "  mathscriptc <file.ms> [options]\n\n"
        << "Options:\n"
        << "  -h, --help     Show this help message\n"
        << "      --version  Show version information\n\n"
        << "Executes a .ms file as a sequence of REPL commands (one per line).\n"
        << "Empty lines and lines starting with '#' are skipped.\n";
}

void print_version(std::ostream& out) {
    out << "mathscriptc " << ms::VERSION_STRING
        << " (commit " << ms::BUILD_COMMIT
        << ", built " << ms::BUILD_DATE << ")\n";
}

bool is_option(const char* arg, const char* long_opt, const char* short_opt = nullptr) {
    return std::strcmp(arg, long_opt) == 0
        || (short_opt != nullptr && std::strcmp(arg, short_opt) == 0);
}

bool run_script_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "mathscriptc: cannot open file: " << path << '\n';
        return false;
    }

    ms::interp::Interpreter interp;
    bool all_ok = true;
    std::string line;
    while (std::getline(in, line)) {
        if (ms::interp::Interpreter::is_script_skip_line(line)) {
            continue;
        }

        const std::string trimmed = ms::interp::Interpreter::trim(line);
        auto result = interp.execute(trimmed);
        if (result) {
            if (!result->empty()) {
                std::cout << *result;
            }
        } else {
            std::cerr << "error: " << ms::format_error(result.error()) << '\n';
            all_ok = false;
        }
    }
    return all_ok;
}

} // namespace

int main(int argc, char** argv) {
    bool have_source = false;
    std::string source;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (is_option(arg, "--help", "-h")) {
            print_usage(std::cout);
            return 0;
        }
        if (is_option(arg, "--version")) {
            print_version(std::cout);
            return 0;
        }
        if (arg[0] == '-') {
            std::cerr << "mathscriptc: unknown option: " << arg << '\n'
                      << "Try 'mathscriptc --help' for more information.\n";
            return 1;
        }
        if (!have_source) {
            source = arg;
            have_source = true;
        } else {
            std::cerr << "mathscriptc: unexpected argument: " << arg << '\n'
                      << "Try 'mathscriptc --help' for more information.\n";
            return 1;
        }
    }

    if (!have_source) {
        print_usage(std::cerr);
        return 1;
    }

    return run_script_file(source) ? 0 : 1;
}
