#pragma once

#include "ms/core/sym.hpp"
#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ms::axiom {

struct Algorithm {
    ms::Sym representation;
    ms::Sym evaluation;
    ms::Sym selection;
    ms::Sym mutation;
    double fitness = 0.0;
};

struct PrimitiveRegistry {
    std::vector<std::string> function_names;
    std::vector<ms::Sym> function_symbols;

    static PrimitiveRegistry build_from_ms_namespace();
};

struct EvolutionConfig {
    size_t population_size = 100;
    size_t max_generations = 1000;
    double mutation_rate = 0.1;
    double crossover_rate = 0.7;
    size_t tournament_size = 5;
    double target_alpha = 0.5;
    size_t max_depth = 10;
    bool use_cuda = true;
};

struct GPNode;

class Axiom {
public:
    Axiom(EvolutionConfig cfg, PrimitiveRegistry primitives);
    ~Axiom();

    Result<Algorithm> evolve(
        const std::function<double(const Algorithm&)>& objective,
        const std::function<bool(const Algorithm&)>& termination);

    Result<Matrix<double>> evaluate(const Algorithm& algo, const Matrix<double>& data) const;
    double gria_fitness(const Algorithm& algo, const Matrix<double>& data) const;
    const std::vector<Algorithm>& population() const { return population_; }

private:
    EvolutionConfig cfg_;
    PrimitiveRegistry primitives_;
    std::vector<Algorithm> population_;
    std::vector<std::unique_ptr<GPNode>> gp_trees_;
    size_t num_vars_ = 2;

    void sync_representation(size_t index);
};

} // namespace ms::axiom
