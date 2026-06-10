#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ms/interp/repl_engine.hpp"
#include "ms/ms.hpp"
#include "ms/version.hpp"

namespace {

bool is_option(const char* arg, const char* long_opt, const char* short_opt = nullptr) {
    return std::strcmp(arg, long_opt) == 0
        || (short_opt != nullptr && std::strcmp(arg, short_opt) == 0);
}

void print_usage(std::ostream& out) {
    out << "MathScript REPL (console mode)\n\n"
        << "Usage:\n"
        << "  mathscript-repl [options]\n\n"
        << "Options:\n"
        << "  -e, --eval <command>  Execute a REPL command and exit (repeatable)\n"
        << "  -h, --help            Show this help message\n"
        << "      --version         Show version information\n\n"
        << "Without -e, runs an interactive session (type 'help' for commands).\n";
}

void print_version(std::ostream& out) {
    out << "mathscript-repl " << ms::VERSION_STRING
        << " (commit " << ms::BUILD_COMMIT
        << ", built " << ms::BUILD_DATE << ")\n";
}

bool run_command(ms::interp::Interpreter& interp, const std::string& line) {
    const std::string trimmed = ms::interp::Interpreter::trim(line);
    if (trimmed.empty()) {
        return true;
    }

    auto result = interp.execute(trimmed);
    if (result) {
        if (!result->empty()) {
            std::cout << *result;
        }
        return true;
    }

    std::cout << "error: " << ms::format_error(result.error()) << "\n";
    return false;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> eval_lines;
    bool show_help = false;
    bool show_version = false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (is_option(arg, "--help", "-h")) {
            show_help = true;
            continue;
        }
        if (is_option(arg, "--version")) {
            show_version = true;
            continue;
        }
        if (is_option(arg, "--eval", "-e")) {
            if (i + 1 >= argc) {
                std::cerr << "mathscript-repl: option requires an argument: " << arg << '\n';
                return 1;
            }
            eval_lines.push_back(argv[++i]);
            continue;
        }
        std::cerr << "mathscript-repl: unknown option: " << arg << '\n'
                  << "Try 'mathscript-repl --help' for more information.\n";
        return 1;
    }

    if (show_version) {
        print_version(std::cout);
        return 0;
    }
    if (show_help) {
        print_usage(std::cout);
        return 0;
    }

    ms::interp::Interpreter interp;

    if (!eval_lines.empty()) {
        bool ok = true;
        for (const auto& line : eval_lines) {
            ok = run_command(interp, line) && ok;
        }
        return ok ? 0 : 1;
    }

    std::cout << "MathScript REPL " << ms::VERSION_STRING << " (console mode)\n";
    std::cout << "Type 'help' for commands, 'exit' to quit.\n";

    std::string line;
    while (true) {
        std::cout << "ms> ";
        if (!std::getline(std::cin, line)) {
            break;
        }
        const std::string trimmed = ms::interp::Interpreter::trim(line);
        if (trimmed == "exit" || trimmed == "quit") {
            break;
        }
        if (trimmed.empty()) {
            continue;
        }
        (void)run_command(interp, line);
    }

    return 0;
}
