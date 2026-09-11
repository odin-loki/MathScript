// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MathScript headless compute node.
//
// `mathscript-server` is the SPMD entry point described in docs/ARCHITECTURE.md:
// every rank runs the same MathScript command stream against its own
// `ms::interp::Interpreter`, so the distributed kernels (`dist_matmul`,
// `dist_cg`, `mpi_allreduce_*`, …) see every rank arrive at the same collective
// in the same order. Rank 0 owns stdout; worker ranks stay silent unless
// `--verbose-ranks` is given.
//
// Command sources, in the order they are consumed:
//   --script <file>   every rank opens and executes the file itself (the usual
//                     HPC shared-filesystem pattern -- no broadcast needed)
//   -e <command>      executed on every rank, repeatable
//   --serve           rank 0 reads lines from stdin and broadcasts each one
//
// With no source at all this is a readiness probe: print the banner, join one
// barrier and exit 0, so `mathscript-server` in a launcher script tells you the
// ranks came up. When stdin is not a terminal (a pipe or /dev/null) the serve
// loop runs instead, which makes `echo 'x = 1' | mathscript-server` work.

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ms/distributed/mpi_context.hpp"
#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/version.hpp"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

bool is_option(const char* arg, const char* long_opt, const char* short_opt = nullptr) {
    return std::strcmp(arg, long_opt) == 0
        || (short_opt != nullptr && std::strcmp(arg, short_opt) == 0);
}

bool stdin_is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

void print_usage(std::ostream& out) {
    out << "MathScript headless compute node\n\n"
        << "Usage:\n"
        << "  mathscript-server [options]\n\n"
        << "Options:\n"
        << "  -e, --eval <command>   Execute a command on every rank (repeatable)\n"
        << "      --script <file>    Execute script lines from <file> on every rank\n"
        << "      --serve            Read commands from stdin until EOF and broadcast them\n"
        << "      --verbose-ranks    Let every rank write to stdout, not just rank 0\n"
        << "  -q, --quiet            Suppress the startup banner\n"
        << "  -h, --help             Show this help message\n"
        << "      --version          Show version information\n\n"
        << "With no command source, prints the banner, joins one barrier and exits\n"
        << "(a launcher readiness probe). When stdin is a pipe rather than a terminal,\n"
        << "the serve loop runs instead.\n";
}

void print_version(std::ostream& out) {
    out << "mathscript-server " << ms::VERSION_STRING
        << " (commit " << ms::BUILD_COMMIT
        << ", built " << ms::BUILD_DATE << ")\n";
}

// Broadcast one command line from rank 0 to every rank.
//
// ms::distributed exposes a `double` broadcast as its only payload primitive
// (see mpi_context.hpp), so a line travels as its length followed by one
// character per broadcast. Command lines are short and this runs once per
// interactive line, never inside a numeric kernel, so the extra collectives do
// not matter; the alternative -- widening the distributed ABI -- would be a
// larger change than a serve loop warrants. On the single-rank / no-MPI build
// `bcast` is the identity and this is a plain copy.
std::string broadcast_line(const ms::distributed::MPIContext& ctx, const std::string& line) {
    if (ms::distributed::size(ctx) <= 1) {
        return line;
    }

    const double len_d = ms::distributed::bcast(ctx, static_cast<double>(line.size()));
    if (!(len_d >= 0.0)) {
        return std::string{};
    }
    const std::size_t len = static_cast<std::size_t>(len_d);

    std::string out;
    out.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        const double ch = ms::distributed::bcast(
            ctx, i < line.size() ? static_cast<double>(static_cast<unsigned char>(line[i])) : 0.0);
        out.push_back(static_cast<char>(static_cast<unsigned char>(ch)));
    }
    return out;
}

// Returns false when the command asked the node to shut down.
bool run_command(ms::interp::Interpreter& interp,
                 const std::string& line,
                 bool writes_stdout,
                 int rank_id) {
    const std::string cmd = ms::interp::Interpreter::trim(line);
    if (cmd.empty() || ms::interp::Interpreter::is_script_skip_line(cmd)) {
        return true;
    }
    if (cmd == "exit" || cmd == "quit") {
        return false;
    }

    auto result = interp.execute(cmd);
    if (!writes_stdout) {
        return true;
    }
    if (result) {
        if (!result->empty()) {
            std::cout << *result;
        }
    } else {
        std::cout << "rank " << rank_id << " error: " << ms::format_error(result.error()) << "\n";
    }
    return true;
}

bool run_script(ms::interp::Interpreter& interp,
                const std::string& path,
                bool writes_stdout,
                int rank_id) {
    std::ifstream in(path);
    if (!in) {
        if (writes_stdout) {
            std::cout << "error: cannot open script: " << path << "\n";
        }
        return true;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!run_command(interp, line, writes_stdout, rank_id)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> eval_commands;
    std::string script_path;
    bool serve = false;
    bool quiet = false;
    bool verbose_ranks = false;

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
        if (is_option(arg, "--eval", "-e")) {
            if (i + 1 >= argc) {
                std::cerr << "error: --eval requires a command\n";
                return 2;
            }
            ++i;
            eval_commands.emplace_back(argv[i]);
            continue;
        }
        if (is_option(arg, "--script")) {
            if (i + 1 >= argc) {
                std::cerr << "error: --script requires a path\n";
                return 2;
            }
            ++i;
            script_path = argv[i];
            continue;
        }
        if (is_option(arg, "--serve")) {
            serve = true;
            continue;
        }
        if (is_option(arg, "--quiet", "-q")) {
            quiet = true;
            continue;
        }
        if (is_option(arg, "--verbose-ranks")) {
            verbose_ranks = true;
            continue;
        }
        std::cerr << "error: unknown option: " << arg << "\n";
        print_usage(std::cerr);
        return 2;
    }

    auto ctx = ms::distributed::init(argc, argv);
    const int r = ms::distributed::rank(ctx);
    const int n = ms::distributed::size(ctx);
    const bool writes_stdout = verbose_ranks || r == 0;

    if (!quiet && writes_stdout) {
        std::cout << "mathscript-server " << ms::VERSION_STRING
                  << " rank " << r << "/" << n
                  << " backend=" << ms::distributed::backend_name(ctx) << "\n";
    }

    ms::distributed::barrier(ctx);
    if (!quiet && r == 0) {
        std::cout << "cluster ready (" << n << " ranks)\n";
    }

    // No explicit source: serve a piped stdin, otherwise act as a readiness probe.
    const bool have_source = !script_path.empty() || !eval_commands.empty() || serve;
    if (!have_source && !stdin_is_terminal()) {
        serve = true;
    }

    ms::interp::Interpreter interp;
    bool running = true;

    if (running && !script_path.empty()) {
        running = run_script(interp, script_path, writes_stdout, r);
    }

    for (const auto& cmd : eval_commands) {
        if (!running) {
            break;
        }
        running = run_command(interp, cmd, writes_stdout, r);
    }

    if (running && serve) {
        while (running) {
            std::string line;
            bool have_line = false;
            if (r == 0) {
                have_line = static_cast<bool>(std::getline(std::cin, line));
                if (!have_line) {
                    line = "exit";
                }
            }
            line = broadcast_line(ctx, line);
            running = run_command(interp, line, writes_stdout, r);
            if (r == 0 && !have_line) {
                break;
            }
        }
    }

    ms::distributed::barrier(ctx);
    if (!quiet && r == 0) {
        std::cout << "shutdown\n";
    }
    ms::distributed::finalize(ctx);
    return 0;
}
