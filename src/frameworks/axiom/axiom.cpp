#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>

namespace ms::axiom {

PrimitiveRegistry PrimitiveRegistry::build_from_ms_namespace() {
    PrimitiveRegistry registry;
    registry.function_names = {
        "matmul", "solve", "lu", "qr", "svd", "eig_sym", "fft", "det", "trace", "norm"};
    for (const auto& name : registry.function_names) {
        registry.function_symbols.emplace_back(name.c_str());
    }
    return registry;
}

Axiom::Axiom(EvolutionConfig cfg, PrimitiveRegistry primitives)
    : cfg_(std::move(cfg)), primitives_(std::move(primitives)) {
    population_.resize(cfg_.population_size);
    std::mt19937 gen{42};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& algo : population_) {
        algo.representation = Sym("x");
        algo.evaluation = Sym("f(x)");
        algo.selection = Sym("tournament");
        algo.mutation = Sym("mutate");
        algo.fitness = dist(gen);
    }
}

Result<Algorithm> Axiom::evolve(
    const std::function<double(const Algorithm&)>& objective,
    const std::function<bool(const Algorithm&)>& termination) {
    if (population_.empty()) {
        return std::unexpected(DomainError{"axiom", "empty population"});
    }

    izaac::CSPRNG fallback(std::array<uint8_t, 32>{});
    izaac::CSPRNG& rng = izaac::g_session_rng ? *izaac::g_session_rng : fallback;

    Algorithm best = population_.front();
    best.fitness = objective(best);

    for (size_t generation = 0; generation < cfg_.max_generations; ++generation) {
        for (auto& algo : population_) {
            algo.fitness = objective(algo);
            if (algo.fitness > best.fitness) {
                best = algo;
            }
        }
        if (termination(best)) {
            break;
        }
        for (auto& algo : population_) {
            algo.fitness += (rng.next_f64() - 0.5) * cfg_.mutation_rate;
        }
    }

    return best;
}

Result<Matrix<double>> Axiom::evaluate(const Algorithm& algo, const Matrix<double>& data) const {
    Matrix<double> output(data.rows(), 1);
    for (size_t i = 0; i < data.rows(); ++i) {
        std::map<std::string, double> env;
        for (size_t j = 0; j < data.cols(); ++j) {
            env["x" + std::to_string(j)] = data(i, j);
        }
        output(i, 0) = algo.representation.eval(env);
    }
    return output;
}

double Axiom::gria_fitness(const Algorithm& algo, const Matrix<double>& data) const {
    (void)algo;
    Matrix<double> transformed(data.rows(), data.cols());
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            transformed(i, j) = std::tanh(data(i, j));
        }
    }
    const double alpha = gria::matrix_alpha(data, transformed);
    return 1.0 - std::abs(alpha - cfg_.target_alpha);
}

} // namespace ms::axiom
