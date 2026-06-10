#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ms/interp/jit_backend.hpp"
#include "ms/interp/jit_backend_impl.hpp"
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
        << "      --load <file.ms>  Load session before eval or interactive mode\n"
        << "      --jit             Use LLVM ORC backend when linked (falls back to REPL)\n"
        << "      --jit-stats       Log JIT dispatch path for each command (requires --jit)\n"
        << "  -h, --help            Show this help message\n"
        << "      --version         Show version information\n\n"
        << "Without -e, runs an interactive session (type 'help' for commands).\n";
}

void print_version(std::ostream& out) {
    out << "mathscript-repl " << ms::VERSION_STRING
        << " (commit " << ms::BUILD_COMMIT
        << ", built " << ms::BUILD_DATE << ")\n";
}

bool run_command(ms::interp::JitBackend& backend, const std::string& line) {
    const std::string trimmed = ms::interp::Interpreter::trim(line);
    if (trimmed.empty()) {
        return true;
    }

    auto result = backend.execute(trimmed);
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
    std::string load_path;
    bool show_help = false;
    bool show_version = false;
    bool use_jit = false;
    bool jit_stats = false;

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
        if (is_option(arg, "--jit")) {
            use_jit = true;
            continue;
        }
        if (is_option(arg, "--jit-stats")) {
            jit_stats = true;
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
        if (is_option(arg, "--load")) {
            if (i + 1 >= argc) {
                std::cerr << "mathscript-repl: option requires an argument: " << arg << '\n';
                return 1;
            }
            load_path = argv[++i];
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

    auto backend = ms::interp::create_backend(use_jit ? ms::interp::Backend::OrcJit
                                                      : ms::interp::Backend::Repl);
#if defined(MS_JIT_LLVM)
    if (jit_stats && use_jit) {
        ms::interp::set_orc_jit_stats_enabled(true);
    }
#endif

    bool ok = true;
    if (!load_path.empty()) {
        ok = run_command(*backend, "load " + load_path) && ok;
    }

    if (!eval_lines.empty()) {
        for (const auto& line : eval_lines) {
            ok = run_command(*backend, line) && ok;
        }
        return ok ? 0 : 1;
    }

    std::cout << "MathScript REPL " << ms::VERSION_STRING << " (console mode)";
    if (use_jit) {
        std::cout << " [" << backend->backend_name() << "]";
    }
    std::cout << "\nType 'help' for commands, 'exit' to quit.\n";

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
        (void)run_command(*backend, line);
    }

    return 0;
}
