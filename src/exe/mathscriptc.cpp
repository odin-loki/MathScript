// MathScript Headless Compiler
// Compiles .cpp files to executable

#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#include "ms/version.hpp"

namespace {

void print_usage(std::ostream& out) {
    out << "MathScript batch compiler (mathscriptc)\n\n"
        << "Usage:\n"
        << "  mathscriptc [options]\n"
        << "  mathscriptc <source.cpp> [options]\n\n"
        << "Options:\n"
        << "  -h, --help     Show this help message\n"
        << "      --version  Show version information\n\n"
        << "Compiles MathScript C++ profile sources into standalone executables.\n"
        << "The batch compiler is not yet implemented.\n";
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

    std::cerr << "mathscriptc: batch compiler is not yet implemented.\n"
              << "  Source: " << source << '\n'
              << "  Planned output: "
              << source.substr(0, source.length() > 4 ? source.length() - 4 : 0) << ".exe\n";
    return 2;
}
