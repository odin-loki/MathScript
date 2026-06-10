#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ms/error/error_types.hpp"
#include "ms/interp/jit_backend.hpp"
#include "ms/interp/jit_backend_impl.hpp"
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
        << "      --load <file.ms>  Load session before eval or interactive mode\n"
        << "      --eval-file <path>  Execute script lines from file before interactive mode\n"
        << "      --debug           Trace each command (timing, state diff, parse kind)\n"
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

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool is_plot_command(const std::string& lcmd) {
    if (lcmd == "show" || lcmd.rfind("saveplot ", 0) == 0) {
        return true;
    }
    static const char* plot_prefixes[] = {"plot(", "scatter(", "hist(", "imshow(", "spy(", "surf("};
    for (const char* prefix : plot_prefixes) {
        if (lcmd.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

std::string classify_repl_line(const std::string& line) {
    const std::string cmd = ms::interp::Interpreter::trim(line);
    if (cmd.empty()) {
        return "empty";
    }

    const std::string lcmd = lower_copy(cmd);
    if (lcmd == "help" || lcmd == "vars" || lcmd == "version" || lcmd == "clear" ||
        lcmd == "topology" || lcmd == "simd" || lcmd == "dispatch" || lcmd == "balance" ||
        lcmd == "gpu" || lcmd == "mpi" || lcmd == "frameworks" || lcmd.rfind("izaac seed ", 0) == 0 ||
        lcmd == "axiom evolve") {
        return "builtin";
    }
    if (lcmd.rfind("save ", 0) == 0) {
        return "save_session";
    }
    if (lcmd.rfind("load ", 0) == 0) {
        return "load_session";
    }
    if (is_plot_command(lcmd)) {
        return "plot_command";
    }

    const auto eq = cmd.find('=');
    if (eq != std::string::npos) {
        std::string name;
        double value = 0.0;
        if (ms::interp::Interpreter::try_parse_scalar_assignment(cmd, name, value)) {
            return "scalar_literal";
        }

        ms::interp::MultiMatrixCallAssign multi_call{};
        if (ms::interp::Interpreter::try_parse_multi_matrix_call_assignment(cmd, multi_call)) {
            return "multi_assign";
        }

        ms::interp::MatrixCallAssign matrix_call{};
        if (ms::interp::Interpreter::try_parse_matrix_call_assignment(cmd, matrix_call)) {
            return "matrix_call";
        }

        ms::interp::ScalarMatrixCallAssign scalar_matrix_call{};
        if (ms::interp::Interpreter::try_parse_scalar_matrix_call_assignment(cmd, scalar_matrix_call)) {
            return "scalar_matrix_call";
        }

        std::string expr;
        if (ms::interp::Interpreter::try_parse_scalar_expr_assignment(cmd, name, expr)) {
            return "scalar_expr";
        }

        const std::string rhs = ms::interp::Interpreter::trim(cmd.substr(eq + 1));
        if (!rhs.empty() && rhs.front() == '[') {
            return "matrix_assign";
        }

        return "assign";
    }

    if (cmd.find('(') != std::string::npos) {
        return "function_call";
    }

    return "unknown";
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

class DebugTracer {
public:
    explicit DebugTracer(ms::interp::JitBackend& backend) : backend_(backend) {}

    bool trace(const std::string& line) {
        const std::string trimmed = ms::interp::Interpreter::trim(line);
        if (trimmed.empty()) {
            return true;
        }

        const auto before_scalars = backend_.state().scalars;
        const auto before_matrices = backend_.state().matrices;

        std::cerr << "[debug] >> " << trimmed << "\n";

        const auto t0 = std::chrono::high_resolution_clock::now();
        auto result = backend_.execute(trimmed);
        const auto t1 = std::chrono::high_resolution_clock::now();

        for (const auto& [k, v] : backend_.state().scalars) {
            const auto it = before_scalars.find(k);
            if (it == before_scalars.end()) {
                std::cerr << "[debug] " << k << " = " << v << " (was: <unset>)\n";
            } else if (it->second != v) {
                std::cerr << "[debug] " << k << " = " << v << " (was: " << it->second << ")\n";
            }
        }

        for (const auto& [k, v] : backend_.state().matrices) {
            if (!before_matrices.count(k)) {
                std::cerr << "[debug] matrix " << k << " (" << v.rows() << "x" << v.cols() << ") (new)\n";
            }
        }

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[debug] elapsed: " << std::fixed << std::setprecision(3) << ms << "ms\n";
        std::cerr << "[debug] kind: " << classify_repl_line(trimmed) << "\n";

        if (result) {
            if (!result->empty()) {
                std::cerr << "[debug] output: " << *result;
            }
            return true;
        }

        std::cerr << "[debug] error: " << ms::format_error(result.error()) << "\n";
        return false;
    }

private:
    ms::interp::JitBackend& backend_;
};

bool run_eval_file(ms::interp::JitBackend& backend, const std::string& path, bool use_debug,
                   DebugTracer* tracer) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "mathscript-repl: cannot open file: " << path << '\n';
        return false;
    }

    bool ok = true;
    std::string line;
    while (std::getline(in, line)) {
        const std::string trimmed = ms::interp::Interpreter::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        const bool line_ok = use_debug ? tracer->trace(trimmed) : run_command(backend, trimmed);
        ok = line_ok && ok;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> eval_lines;
    std::string load_path;
    std::string eval_file_path;
    bool show_help = false;
    bool show_version = false;
    bool use_jit = false;
    bool jit_stats = false;
    bool use_debug = false;

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
        if (is_option(arg, "--debug")) {
            use_debug = true;
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
        if (is_option(arg, "--eval-file")) {
            if (i + 1 >= argc) {
                std::cerr << "mathscript-repl: option requires an argument: " << arg << '\n';
                return 1;
            }
            eval_file_path = argv[++i];
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

    DebugTracer tracer(*backend);
    const auto run_line = [&](const std::string& line) -> bool {
        return use_debug ? tracer.trace(line) : run_command(*backend, line);
    };

    bool ok = true;
    if (!load_path.empty()) {
        ok = run_line("load " + load_path) && ok;
    }

    if (!eval_file_path.empty()) {
        ok = run_eval_file(*backend, eval_file_path, use_debug, &tracer) && ok;
    }

    if (!eval_lines.empty()) {
        for (const auto& line : eval_lines) {
            ok = run_line(line) && ok;
        }
        return ok ? 0 : 1;
    }

    std::cout << "MathScript REPL " << ms::VERSION_STRING << " (console mode)";
    if (use_jit) {
        std::cout << " [" << backend->backend_name() << "]";
    }
    if (use_debug) {
        std::cout << " [debug mode]";
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
        (void)run_line(line);
    }

    return 0;
}
