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

/// Pool of unary scalar functions available to GP tree generation.
/// Names must match ms::Sym's supported call grammar (sin/cos/exp/log/sqrt/tanh).
struct PrimitiveRegistry {
    std::vector<std::string> function_names;
    std::vector<ms::Sym> function_symbols;

    /// Builds a registry from ms::Sym's supported scalar unary-function grammar.
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

    /// Supervised regression fitness: mean squared error of `algo` evaluated row-by-row
    /// over `inputs` (column j bound to variable "xj", matching evaluate()'s convention)
    /// against `targets[i]`.
    ///
    /// Conventions (no exceptions are thrown; malformed inputs degrade gracefully):
    ///  - Row count is `min(inputs.rows(), targets.size())`; a mismatch is not an error,
    ///    the extra rows/targets are simply ignored. If either is empty, returns 0.0.
    ///  - `ms::Sym::eval` already guards div-by-zero, log(<=0), and sqrt(<0) by returning
    ///    0.0 (see sym.cpp), so those cannot produce NaN/Inf. The remaining risk is
    ///    numeric overflow (e.g. exp() of a large argument, or repeated multiplication
    ///    in a deep tree) producing +-Inf, or Inf-Inf producing NaN. Any row whose
    ///    prediction or squared error is non-finite is penalized with a large-but-finite
    ///    sentinel (1e12) instead of letting NaN/Inf propagate into the aggregate score,
    ///    which would otherwise silently destroy selection pressure across the whole
    ///    population in an evolutionary loop.
    double mse_fitness(const Algorithm& algo, const Matrix<double>& inputs, const std::vector<double>& targets) const;

    /// Root of mse_fitness(); same conventions apply (see mse_fitness doc comment).
    double rmse_fitness(const Algorithm& algo, const Matrix<double>& inputs, const std::vector<double>& targets) const;

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
