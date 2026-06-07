#include "ms/distributed/mpi_context.hpp"
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/simd/simd.hpp"
#include "ms/special/special.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace ms::interp {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool parse_number(const std::string& text, double& value) {
    try {
        size_t pos = 0;
        value = std::stod(text, &pos);
        return pos == text.size();
    } catch (...) {
        return false;
    }
}

} // namespace

std::string Interpreter::trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

Result<Matrix<double>> Interpreter::parse_matrix(const std::string& text) const {
    std::string s = trim(text);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
        return std::unexpected(DomainError{"parse_matrix", "expected [ ... ]"});
    }
    s = s.substr(1, s.size() - 2);
    if (s.empty()) {
        return Matrix<double>(0, 0);
    }

    std::vector<std::vector<double>> rows;
    std::stringstream row_stream(s);
    std::string row_text;
    while (std::getline(row_stream, row_text, ';')) {
        row_text = trim(row_text);
        if (row_text.empty()) {
            continue;
        }
        std::vector<double> row;
        std::stringstream col_stream(row_text);
        std::string cell;
        while (std::getline(col_stream, cell, ',')) {
            cell = trim(cell);
            double value = 0.0;
            if (!parse_number(cell, value)) {
                return std::unexpected(DomainError{"parse_matrix", "invalid number: " + cell});
            }
            row.push_back(value);
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        return std::unexpected(DomainError{"parse_matrix", "no rows found"});
    }
    const size_t cols = rows[0].size();
    for (const auto& row : rows) {
        if (row.size() != cols) {
            return std::unexpected(DimensionMismatch{rows.size(), row.size()});
        }
    }

    Matrix<double> m(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = rows[i][j];
        }
    }
    return m;
}

Result<Matrix<double>> Interpreter::resolve_matrix(const std::string& name) const {
    const auto it = state_.matrices.find(name);
    if (it == state_.matrices.end()) {
        return std::unexpected(DomainError{"resolve", "unknown matrix: " + name});
    }
    return it->second;
}

void print_matrix(std::ostream& out, const Matrix<double>& m) {
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < m.rows(); ++i) {
        out << "  [";
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << m(i, j);
        }
        out << "]\n";
    }
}

std::string matrix_to_line(const Matrix<double>& m) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < m.rows(); ++i) {
        if (i > 0) {
            out << "; ";
        }
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << m(i, j);
        }
    }
    out << "]";
    return out.str();
}

Result<std::string> Interpreter::set_plot(const Matrix<double>& xs, const Matrix<double>& ys) {
    const size_t n = ys.rows() * ys.cols();
    if (n == 0) {
        return std::unexpected(DomainError{"plot", "empty data"});
    }
    std::vector<double> x;
    std::vector<double> y;
    y.reserve(n);
    if (xs.rows() == 0 && xs.cols() == 0) {
        x.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            x.push_back(static_cast<double>(i));
        }
    } else {
        const size_t xn = xs.rows() * xs.cols();
        if (xn != n) {
            return std::unexpected(DimensionMismatch{xn, n});
        }
        x.reserve(n);
        if (xs.cols() == 1) {
            for (size_t i = 0; i < xs.rows(); ++i) {
                x.push_back(xs(i, 0));
            }
        } else {
            for (size_t i = 0; i < xs.cols(); ++i) {
                x.push_back(xs(0, i));
            }
        }
    }
    if (ys.cols() == 1) {
        for (size_t i = 0; i < ys.rows(); ++i) {
            y.push_back(ys(i, 0));
        }
    } else {
        for (size_t i = 0; i < ys.cols(); ++i) {
            y.push_back(ys(0, i));
        }
    }
    state_.plot.x = std::move(x);
    state_.plot.y = std::move(y);
    state_.plot.kind = PlotSeries::Kind::Line;
    state_.plot.valid = true;
    return "plot updated (" + std::to_string(state_.plot.y.size()) + " points)\n";
}

Result<std::string> Interpreter::set_plot_bars(std::vector<double> x, std::vector<double> y) {
    if (x.size() != y.size() || x.empty()) {
        return std::unexpected(DimensionMismatch{x.size(), y.size()});
    }
    state_.plot.x = std::move(x);
    state_.plot.y = std::move(y);
    state_.plot.kind = PlotSeries::Kind::Bar;
    state_.plot.valid = true;
    return "histogram updated (" + std::to_string(state_.plot.y.size()) + " bins)\n";
}

Result<void> Interpreter::save_session(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        return std::unexpected(DomainError{"save", "cannot open: " + path});
    }
    out << "# MathScript session\n";
    for (const auto& [name, value] : state_.scalars) {
        out << "scalar " << name << " = " << value << "\n";
    }
    for (const auto& [name, matrix] : state_.matrices) {
        out << "matrix " << name << " = " << matrix_to_line(matrix) << "\n";
    }
    return {};
}

Result<void> Interpreter::load_session(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected(DomainError{"load", "cannot open: " + path});
    }
    reset();
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string lhs = trim(line.substr(0, eq));
        const std::string rhs = trim(line.substr(eq + 1));
        const auto space = lhs.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        const std::string kind = lower(lhs.substr(0, space));
        const std::string name = trim(lhs.substr(space + 1));
        if (kind == "scalar") {
            double value = 0.0;
            if (!parse_number(rhs, value)) {
                return std::unexpected(DomainError{"load", "invalid scalar: " + name});
            }
            state_.scalars[name] = value;
        } else if (kind == "matrix") {
            auto matrix = parse_matrix(rhs);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            state_.matrices[name] = *matrix;
        }
    }
    return {};
}

Result<std::string> Interpreter::execute(const std::string& line) {
    const std::string cmd = trim(line);
    if (cmd.empty()) {
        return std::string{};
    }
    state_.history.push_back(cmd);

    const std::string lcmd = lower(cmd);
    if (lcmd == "help") {
        return std::string{
            "Commands:\n"
            "  help, vars, clear, topology, simd, dispatch, balance, gpu, mpi, frameworks, exit\n"
            "  izaac seed <n>   gria alpha [data]   axiom evolve\n"
            "  save <file.ms>  load <file.ms>\n"
            "  name = [1, 2; 3, 4]     matrix assignment\n"
            "  name = 3.14              scalar assignment\n"
            "  plot([y...])  plot([x...], [y...])  hist([...])\n"
            "  det(A), trace(A), norm(A), rank(A), cond(A)\n"
            "  lu(A), qr(A), chol(A), solve(A,B), matmul(A,B), eig_sym(A), svd(A)\n"
            "  fft([1,2,3,4])           vector FFT magnitude\n"
            "  erf(x), gamma(x), bessel_j0(x), spherical_jn(n,x)\n"
            "  kelvin_ber(0,x), struve_h(n,x), bessel_zero_jnu(nu,n)\n"
            "  kummer_m(a,b,x), hypergeo_2f1(a,b,c,x), whittaker_m(k,mu,x)\n"
            "  jacobi_p(n,a,b,x), ellip_k(k), jacobi_sn(u,k)\n"
            "  theta3(z,q), zeta(s), polylog(n,z), mathieu_ce(n,q,x)\n"
            "  heun_g(a,q,alpha,beta,gamma,delta,z), painleve1(x,y0,yp0)\n"
            "  legendre_p(n,x), beta(a,b)\n"};
    }
    if (lcmd == "vars") {
        std::ostringstream out;
        for (const auto& [name, value] : state_.scalars) {
            out << name << " = " << value << "\n";
        }
        for (const auto& [name, matrix] : state_.matrices) {
            out << name << " (" << matrix.rows() << "x" << matrix.cols() << ")\n";
        }
        if (state_.scalars.empty() && state_.matrices.empty()) {
            out << "(empty session)\n";
        }
        return out.str();
    }
    if (lcmd == "clear") {
        reset();
        return std::string{"session cleared\n"};
    }
    if (lcmd == "topology") {
        const auto topo = detect_topology();
        std::ostringstream out;
        out << "cores=" << topo.total_cores()
            << " threads=" << topo.total_threads()
            << " gpus=" << topo.total_gpus << "\n";
        return out.str();
    }
    if (lcmd == "simd") {
        const auto info = simd::dispatch_info();
        return "ISA: " + simd::isa_summary(info.isa) + "\n";
    }
    if (lcmd == "gpu") {
        std::ostringstream out;
        out << "cuda=" << (has_cuda() ? "yes" : "no")
            << " devices=" << get_gpu_count() << "\n";
        if (get_gpu_count() > 0) {
            out << "device0=" << get_gpu_model(0) << "\n";
        }
        return out.str();
    }
    if (lcmd == "dispatch") {
        const auto d = decide(512, ExecPolicy::AUTO);
        std::ostringstream out;
        out << "auto@512 backend="
            << (d.backend == Backend::CUDA ? "cuda" : "cpu")
            << " threads=" << d.n_threads << "\n";
        return out.str();
    }
    if (lcmd == "balance") {
        const auto lb = balance(1024, ExecPolicy::AUTO);
        std::ostringstream out;
        out << "backend=" << (lb.backend == Backend::CUDA ? "cuda" : "cpu")
            << " device=" << lb.cuda_device
            << " threads=" << lb.cpu_threads << "\n";
        return out.str();
    }
    if (lcmd.rfind("save ", 0) == 0) {
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) {
            return std::unexpected(DomainError{"save", "missing path"});
        }
        auto saved = save_session(path);
        if (!saved) {
            return std::unexpected(saved.error());
        }
        return "saved session to " + path + "\n";
    }
    if (lcmd.rfind("load ", 0) == 0) {
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) {
            return std::unexpected(DomainError{"load", "missing path"});
        }
        auto loaded = load_session(path);
        if (!loaded) {
            return std::unexpected(loaded.error());
        }
        return "loaded session from " + path + "\n";
    }

    if (lcmd == "frameworks") {
        return std::string{
            "loaded: GRIA, Izaac, CyphaDIF, CellAI, AXIOM\n"
            "  gria alpha [..]  izaac seed N  axiom evolve\n"};
    }
    if (lcmd.rfind("izaac seed ", 0) == 0) {
        const std::string arg = trim(cmd.substr(11));
        double seed_value = 0.0;
        if (!parse_number(arg, seed_value)) {
            return std::unexpected(DomainError{"izaac", "expected numeric seed"});
        }
        izaac::seed_session(static_cast<uint64_t>(seed_value));
        return "izaac session seeded\n";
    }
    if (lcmd == "axiom evolve") {
        auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
        axiom::Axiom engine(axiom::EvolutionConfig{.population_size = 20, .max_generations = 25}, registry);
        ColMatrix<double> data{{1, 2}, {3, 4}, {5, 6}};
        const auto best = engine
                                .evolve(
                                    [&](const axiom::Algorithm& a) {
                                        return engine.gria_fitness(a, data);
                                    },
                                    [](const axiom::Algorithm& a) { return a.fitness > 0.45; })
                                .value();
        std::ostringstream out;
        out << "axiom best fitness=" << best.fitness << " alpha-target fit\n";
        return out.str();
    }

    if (lcmd == "mpi") {
        static ms::distributed::MPIContext mpi = ms::distributed::init(0, nullptr);
        std::ostringstream out;
        out << "backend=" << ms::distributed::backend_name(mpi)
            << " rank=" << ms::distributed::rank(mpi)
            << " size=" << ms::distributed::size(mpi) << "\n";
        return out.str();
    }

    const auto eq = cmd.find('=');
    if (eq != std::string::npos) {
        const std::string lhs = trim(cmd.substr(0, eq));
        const std::string rhs = trim(cmd.substr(eq + 1));
        if (lhs.empty()) {
            return std::unexpected(DomainError{"assign", "missing variable name"});
        }

        double scalar = 0.0;
        if (parse_number(rhs, scalar)) {
            state_.scalars[lhs] = scalar;
            return lhs + " = " + std::to_string(scalar) + "\n";
        }

        auto matrix = parse_matrix(rhs);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        state_.matrices[lhs] = *matrix;
        std::ostringstream out;
        out << lhs << " =\n";
        print_matrix(out, *matrix);
        return out.str();
    }

    static const std::regex plot_binary(
        R"(plot\s*\(\s*(\[[^\]]+\])\s*,\s*(\[[^\]]+\])\s*\))",
        std::regex::icase);
    std::smatch match;
    if (std::regex_match(cmd, match, plot_binary)) {
        auto xs = parse_matrix(match[1].str());
        auto ys = parse_matrix(match[2].str());
        if (!xs || !ys) {
            return std::unexpected(DomainError{"plot", "invalid x/y vectors"});
        }
        return set_plot(*xs, *ys);
    }

    static const std::regex unary(R"((\w+)\(([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, unary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg = trim(match[2].str());

        if (fn == "plot") {
            auto ys = parse_matrix(arg);
            if (!ys) {
                ys = resolve_matrix(arg);
            }
            if (!ys) {
                return std::unexpected(ys.error());
            }
            Matrix<double> empty(0, 0);
            return set_plot(empty, *ys);
        }

        if (fn == "erf" || fn == "erfc" || fn == "gamma" || fn == "bessel_j0" || fn == "fresnel_c" ||
            fn == "fresnel_s") {
            double value = 0.0;
            if (!parse_number(arg, value)) {
                return std::unexpected(DomainError{"special", "expected numeric argument"});
            }
            std::ostringstream out;
            if (fn == "erf") {
                out << erf(value);
            } else if (fn == "erfc") {
                out << erfc(value);
            } else if (fn == "gamma") {
                out << gamma_func(value);
            } else if (fn == "bessel_j0") {
                out << bessel_j0(value);
            } else if (fn == "fresnel_c") {
                out << fresnel_c(value);
            } else {
                out << fresnel_s(value);
            }
            out << "\n";
            return out.str();
        }

        if (fn == "gria") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            std::vector<double> flat;
            for (size_t i = 0; i < data->rows(); ++i) {
                for (size_t j = 0; j < data->cols(); ++j) {
                    flat.push_back((*data)(i, j));
                }
            }
            const double alpha = gria::compute_alpha<double>(flat, [](std::span<const double> input) {
                std::vector<double> out;
                out.reserve(input.size());
                for (double v : input) {
                    out.push_back(v * v);
                }
                return out;
            });
            std::ostringstream out;
            out << "alpha=" << alpha << " class="
                << (gria::classify(alpha) == gria::ComputeClass::Critical ? "critical"
                    : gria::classify(alpha) == gria::ComputeClass::Irreversible ? "irreversible"
                                                                                  : "reversible")
                << "\n";
            return out.str();
        }

        if (fn == "hist") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            std::vector<double> values;
            values.reserve(data->rows() * data->cols());
            for (size_t i = 0; i < data->rows(); ++i) {
                for (size_t j = 0; j < data->cols(); ++j) {
                    values.push_back((*data)(i, j));
                }
            }
            if (values.empty()) {
                return std::unexpected(DomainError{"hist", "empty data"});
            }
            const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
            const double vmin = *min_it;
            const double vmax = *max_it;
            constexpr size_t bins = 10;
            std::vector<double> counts(bins, 0.0);
            const double width = (vmax - vmin) > 0.0 ? (vmax - vmin) / static_cast<double>(bins) : 1.0;
            for (double value : values) {
                size_t idx = static_cast<size_t>((value - vmin) / width);
                if (idx >= bins) {
                    idx = bins - 1;
                }
                counts[idx] += 1.0;
            }
            std::vector<double> centers(bins);
            for (size_t i = 0; i < bins; ++i) {
                centers[i] = vmin + (static_cast<double>(i) + 0.5) * width;
            }
            return set_plot_bars(std::move(centers), std::move(counts));
        }

        if (fn == "fft") {
            auto vec = parse_matrix(arg);
            if (!vec || vec->cols() != 1) {
                if (vec && vec->rows() == 1) {
                    Matrix<double> col(vec->cols(), 1);
                    for (size_t i = 0; i < vec->cols(); ++i) {
                        col(i, 0) = (*vec)(0, i);
                    }
                    vec = col;
                } else if (!vec) {
                    return std::unexpected(vec.error());
                } else {
                    return std::unexpected(DomainError{"fft", "expected vector [a,b,c,...]"});
                }
            }
            std::vector<double> data(vec->rows());
            for (size_t i = 0; i < vec->rows(); ++i) {
                data[i] = (*vec)(i, 0);
            }
            auto spectrum = fft(data);
            if (!spectrum) {
                return std::unexpected(spectrum.error());
            }
            std::ostringstream out;
            out << "fft magnitudes:\n";
            for (size_t i = 0; i < spectrum->size(); ++i) {
                out << "  [" << i << "] " << std::abs((*spectrum)[i]) << "\n";
            }
            return out.str();
        }

        auto matrix = parse_matrix(arg);
        if (matrix) {
            state_.matrices["_"] = *matrix;
        } else {
            matrix = resolve_matrix(arg);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
        }

        std::ostringstream out;
        if (fn == "det") {
            out << det(*matrix).value_or(0.0) << "\n";
        } else if (fn == "trace") {
            out << trace(*matrix).value_or(0.0) << "\n";
        } else if (fn == "norm") {
            out << norm(*matrix).value_or(0.0) << "\n";
        } else if (fn == "rank") {
            out << rank(*matrix).value_or(0.0) << "\n";
        } else if (fn == "cond") {
            out << cond(*matrix).value_or(0.0) << "\n";
        } else if (fn == "eig_sym") {
            auto eig_result = eig_sym(*matrix);
            if (!eig_result) {
                return std::unexpected(eig_result.error());
            }
            out << "eigenvalues:\n";
            for (size_t i = 0; i < eig_result->values.rows(); ++i) {
                out << "  " << eig_result->values(i, 0) << "\n";
            }
        } else if (fn == "svd") {
            auto s = svd(*matrix);
            if (!s) {
                return std::unexpected(s.error());
            }
            out << "singular values:\n";
            for (size_t i = 0; i < s->S.rows(); ++i) {
                out << "  " << s->S(i, 0) << "\n";
            }
        } else if (fn == "lu" || fn == "qr" || fn == "chol") {
            out << fn << " computed for " << matrix->rows() << "x" << matrix->cols() << " matrix\n";
            if (fn == "lu") {
                (void)lu(*matrix);
            } else if (fn == "qr") {
                (void)qr(*matrix);
            } else {
                (void)chol(*matrix);
            }
        } else {
            return std::unexpected(DomainError{"repl", "unknown function: " + fn});
        }
        return out.str();
    }

    static const std::regex binary(R"((\w+)\(([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, binary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg_a = trim(match[2].str());
        const std::string arg_b = trim(match[3].str());

        if (fn == "beta") {
            double a = 0.0;
            double b = 0.0;
            if (!parse_number(arg_a, a) || !parse_number(arg_b, b)) {
                return std::unexpected(DomainError{"beta", "expected numeric arguments beta(a,b)"});
            }
            return std::to_string(beta_func(a, b)) + "\n";
        }

        if (fn == "legendre_p") {
            double n = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"legendre_p", "expected legendre_p(n,x)"});
            }
            return std::to_string(legendre_p(static_cast<int>(n), x)) + "\n";
        }

        auto A = resolve_matrix(arg_a);
        if (!A) {
            A = parse_matrix(arg_a);
        }
        auto B = resolve_matrix(arg_b);
        if (!B) {
            B = parse_matrix(arg_b);
        }
        if (!A || !B) {
            return std::unexpected(DomainError{"repl", "invalid matrix arguments"});
        }
        if (fn == "plot") {
            return set_plot(*A, *B);
        }
        if (fn == "solve") {
            auto x = solve(*A, *B);
            if (!x) {
                return std::unexpected(x.error());
            }
            std::ostringstream out;
            out << "x =\n";
            print_matrix(out, *x);
            return out.str();
        }
        if (fn == "matmul") {
            auto C = matmul(*A, *B);
            if (!C) {
                return std::unexpected(C.error());
            }
            std::ostringstream out;
            out << "C =\n";
            print_matrix(out, *C);
            return out.str();
        }
    }

    return std::unexpected(DomainError{"repl", "could not parse: " + cmd});
}

} // namespace ms::interp
