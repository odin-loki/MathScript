#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ms::axiom {

struct GPNode {
    enum class Kind { Constant, Variable, BinaryOp, UnaryFunc } kind = Kind::Constant;
    double const_value = 0.0;
    int var_index = 0;
    char binary_op = '+';
    std::string func_name;
    std::unique_ptr<GPNode> left;
    std::unique_ptr<GPNode> right;
};

namespace {

constexpr char kBinaryOps[] = {'+', '-', '*', '/'};

izaac::CSPRNG& session_rng() {
    static izaac::CSPRNG fallback(std::array<uint8_t, 32>{});
    return izaac::g_session_rng ? *izaac::g_session_rng : fallback;
}

int random_int(izaac::CSPRNG& rng, int lo, int hi_exclusive) {
    if (hi_exclusive <= lo) {
        return lo;
    }
    const double u = rng.next_f64();
    return lo + static_cast<int>(u * static_cast<double>(hi_exclusive - lo));
}

double random_double(izaac::CSPRNG& rng, double lo, double hi) {
    return lo + rng.next_f64() * (hi - lo);
}

std::unique_ptr<GPNode> make_constant(double value) {
    auto node = std::make_unique<GPNode>();
    node->kind = GPNode::Kind::Constant;
    node->const_value = value;
    return node;
}

std::unique_ptr<GPNode> make_variable(int index) {
    auto node = std::make_unique<GPNode>();
    node->kind = GPNode::Kind::Variable;
    node->var_index = index;
    return node;
}

std::unique_ptr<GPNode> make_binary(char op, std::unique_ptr<GPNode> left, std::unique_ptr<GPNode> right) {
    auto node = std::make_unique<GPNode>();
    node->kind = GPNode::Kind::BinaryOp;
    node->binary_op = op;
    node->left = std::move(left);
    node->right = std::move(right);
    return node;
}

std::unique_ptr<GPNode> make_unary(const std::string& func, std::unique_ptr<GPNode> arg) {
    auto node = std::make_unique<GPNode>();
    node->kind = GPNode::Kind::UnaryFunc;
    node->func_name = func;
    node->left = std::move(arg);
    return node;
}

std::unique_ptr<GPNode> random_terminal(size_t n_vars, izaac::CSPRNG& rng) {
    if (n_vars == 0 || rng.next_f64() < 0.5) {
        return make_constant(random_double(rng, -5.0, 5.0));
    }
    return make_variable(random_int(rng, 0, static_cast<int>(n_vars)));
}

std::unique_ptr<GPNode> random_tree(
    size_t depth,
    size_t max_depth,
    size_t n_vars,
    const std::vector<std::string>& available_funcs,
    izaac::CSPRNG& rng) {
    if (depth >= max_depth || (depth > 0 && rng.next_f64() < 0.35)) {
        return random_terminal(n_vars, rng);
    }

    if (rng.next_f64() < 0.65 || available_funcs.empty()) {
        const char op = kBinaryOps[random_int(rng, 0, static_cast<int>(std::size(kBinaryOps)))];
        auto left = random_tree(depth + 1, max_depth, n_vars, available_funcs, rng);
        auto right = random_tree(depth + 1, max_depth, n_vars, available_funcs, rng);
        return make_binary(op, std::move(left), std::move(right));
    }

    const std::string& func =
        available_funcs[static_cast<size_t>(random_int(rng, 0, static_cast<int>(available_funcs.size())))];
    auto arg = random_tree(depth + 1, max_depth, n_vars, available_funcs, rng);
    return make_unary(func, std::move(arg));
}

std::string format_constant(double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    return buf;
}

std::string to_sym_string(const GPNode& node) {
    switch (node.kind) {
    case GPNode::Kind::Constant:
        return format_constant(node.const_value);
    case GPNode::Kind::Variable:
        return "x" + std::to_string(node.var_index);
    case GPNode::Kind::BinaryOp: {
        const std::string left = node.left ? to_sym_string(*node.left) : "0";
        const std::string right = node.right ? to_sym_string(*node.right) : "0";
        return "(" + left + node.binary_op + right + ")";
    }
    case GPNode::Kind::UnaryFunc: {
        const std::string arg = node.left ? to_sym_string(*node.left) : "0";
        return node.func_name + "(" + arg + ")";
    }
    }
    return "0";
}

std::unique_ptr<GPNode> clone_tree(const GPNode& node) {
    auto copy = std::make_unique<GPNode>();
    copy->kind = node.kind;
    copy->const_value = node.const_value;
    copy->var_index = node.var_index;
    copy->binary_op = node.binary_op;
    copy->func_name = node.func_name;
    if (node.left) {
        copy->left = clone_tree(*node.left);
    }
    if (node.right) {
        copy->right = clone_tree(*node.right);
    }
    return copy;
}

void collect_nodes(const GPNode& node, std::vector<const GPNode*>& out) {
    out.push_back(&node);
    if (node.left) {
        collect_nodes(*node.left, out);
    }
    if (node.right) {
        collect_nodes(*node.right, out);
    }
}

std::vector<const GPNode*> collect_nodes(const GPNode& node) {
    std::vector<const GPNode*> out;
    collect_nodes(node, out);
    return out;
}

size_t tree_depth(const GPNode& node) {
    const size_t left_depth = node.left ? tree_depth(*node.left) : 0;
    const size_t right_depth = node.right ? tree_depth(*node.right) : 0;
    return 1 + std::max(left_depth, right_depth);
}

std::optional<size_t> find_node_depth(const GPNode& node, const GPNode* target, size_t depth) {
    if (&node == target) {
        return depth;
    }
    if (node.left) {
        if (auto found = find_node_depth(*node.left, target, depth + 1)) {
            return found;
        }
    }
    if (node.right) {
        if (auto found = find_node_depth(*node.right, target, depth + 1)) {
            return found;
        }
    }
    return std::nullopt;
}

bool replace_subtree(GPNode& root, GPNode* target, std::unique_ptr<GPNode> replacement) {
    if (root.left.get() == target) {
        root.left = std::move(replacement);
        return true;
    }
    if (root.right.get() == target) {
        root.right = std::move(replacement);
        return true;
    }
    if (root.left && replace_subtree(*root.left, target, std::move(replacement))) {
        return true;
    }
    if (root.right && replace_subtree(*root.right, target, std::move(replacement))) {
        return true;
    }
    return false;
}

std::unique_ptr<GPNode> crossover_offspring(
    const GPNode& parent_a, const GPNode& parent_b, izaac::CSPRNG& rng) {
    auto offspring = clone_tree(parent_a);
    const auto nodes_a = collect_nodes(*offspring);
    const auto nodes_b = collect_nodes(parent_b);
    if (nodes_a.empty() || nodes_b.empty()) {
        return offspring;
    }

    GPNode* point_a = const_cast<GPNode*>(nodes_a[random_int(rng, 0, static_cast<int>(nodes_a.size()))]);
    const GPNode* point_b = nodes_b[random_int(rng, 0, static_cast<int>(nodes_b.size()))];
    auto graft = clone_tree(*point_b);

    if (point_a == offspring.get()) {
        return graft;
    }
    replace_subtree(*offspring, point_a, std::move(graft));
    return offspring;
}

std::unique_ptr<GPNode> subtree_mutation(
    const GPNode& tree,
    size_t max_depth,
    size_t n_vars,
    const std::vector<std::string>& available_funcs,
    izaac::CSPRNG& rng,
    double mutation_rate) {
    auto result = clone_tree(tree);
    if (rng.next_f64() >= mutation_rate) {
        return result;
    }

    const auto nodes = collect_nodes(*result);
    if (nodes.empty()) {
        return result;
    }

    GPNode* target = const_cast<GPNode*>(nodes[random_int(rng, 0, static_cast<int>(nodes.size()))]);
    const size_t target_depth = find_node_depth(*result, target, 1).value_or(1);
    const size_t remaining = max_depth >= target_depth ? max_depth - target_depth + 1 : 1;
    const size_t subtree_max = std::max<size_t>(1, std::min(remaining, size_t{3}));
    auto replacement = random_tree(0, subtree_max, n_vars, available_funcs, rng);

    if (target == result.get()) {
        return replacement;
    }
    replace_subtree(*result, target, std::move(replacement));
    return result;
}

std::unique_ptr<GPNode> enforce_max_depth(
    std::unique_ptr<GPNode> tree,
    size_t max_depth,
    size_t n_vars,
    const std::vector<std::string>& available_funcs,
    izaac::CSPRNG& rng) {
    while (tree && tree_depth(*tree) > max_depth) {
        tree = random_tree(0, std::max<size_t>(2, max_depth / 2), n_vars, available_funcs, rng);
    }
    return tree;
}

size_t tournament_select(const std::vector<Algorithm>& population, size_t tournament_size, izaac::CSPRNG& rng) {
    if (population.empty()) {
        return 0;
    }
    const int pop_size = static_cast<int>(population.size());
    const int t_size = static_cast<int>(std::min(tournament_size, population.size()));
    int best_idx = random_int(rng, 0, pop_size);
    double best_fitness = population[static_cast<size_t>(best_idx)].fitness;
    for (int t = 1; t < t_size; ++t) {
        const int idx = random_int(rng, 0, pop_size);
        if (population[static_cast<size_t>(idx)].fitness > best_fitness) {
            best_fitness = population[static_cast<size_t>(idx)].fitness;
            best_idx = idx;
        }
    }
    return static_cast<size_t>(best_idx);
}

void assign_placeholder_fields(Algorithm& algo) {
    algo.evaluation = Sym("f(x)");
    algo.selection = Sym("tournament");
    algo.mutation = Sym("mutate");
}

} // namespace

PrimitiveRegistry PrimitiveRegistry::build_from_ms_namespace() {
    PrimitiveRegistry registry;
    registry.function_names = {"sin", "cos", "exp", "log", "sqrt", "tanh"};
    for (const auto& name : registry.function_names) {
        registry.function_symbols.emplace_back(name.c_str());
    }
    return registry;
}

Axiom::~Axiom() = default;

Axiom::Axiom(EvolutionConfig cfg, PrimitiveRegistry primitives)
    : cfg_(std::move(cfg)), primitives_(std::move(primitives)) {
    population_.resize(cfg_.population_size);
    gp_trees_.resize(cfg_.population_size);
    izaac::CSPRNG& rng = session_rng();

    const size_t init_depth = std::max<size_t>(2, cfg_.max_depth / 2);
    for (size_t i = 0; i < cfg_.population_size; ++i) {
        gp_trees_[i] = random_tree(0, init_depth, num_vars_, primitives_.function_names, rng);
        assign_placeholder_fields(population_[i]);
        sync_representation(i);
        population_[i].fitness = 0.0;
    }
}

void Axiom::sync_representation(size_t index) {
    const std::string expr = to_sym_string(*gp_trees_[index]);
    population_[index].representation = Sym(expr.c_str());
}

Result<Algorithm> Axiom::evolve(
    const std::function<double(const Algorithm&)>& objective,
    const std::function<bool(const Algorithm&)>& termination) {
    if (population_.empty()) {
        return std::unexpected(DomainError{"axiom", "empty population"});
    }

    izaac::CSPRNG& rng = session_rng();

    size_t best_idx = 0;
    Algorithm best = population_.front();
    best.fitness = objective(best);

    for (size_t i = 0; i < population_.size(); ++i) {
        population_[i].fitness = objective(population_[i]);
        if (population_[i].fitness > best.fitness) {
            best = population_[i];
            best_idx = i;
        }
    }

    for (size_t generation = 0; generation < cfg_.max_generations; ++generation) {
        if (termination(best)) {
            break;
        }

        std::vector<std::unique_ptr<GPNode>> next_trees;
        next_trees.reserve(cfg_.population_size);
        next_trees.push_back(clone_tree(*gp_trees_[best_idx]));

        for (size_t i = 1; i < cfg_.population_size; ++i) {
            const size_t parent_a = tournament_select(population_, cfg_.tournament_size, rng);
            std::unique_ptr<GPNode> child;
            if (rng.next_f64() < cfg_.crossover_rate) {
                const size_t parent_b = tournament_select(population_, cfg_.tournament_size, rng);
                child = crossover_offspring(*gp_trees_[parent_a], *gp_trees_[parent_b], rng);
            } else {
                child = clone_tree(*gp_trees_[parent_a]);
            }
            child = subtree_mutation(
                *child, cfg_.max_depth, num_vars_, primitives_.function_names, rng, cfg_.mutation_rate);
            child = enforce_max_depth(
                std::move(child), cfg_.max_depth, num_vars_, primitives_.function_names, rng);
            next_trees.push_back(std::move(child));
        }

        gp_trees_ = std::move(next_trees);

        for (size_t i = 0; i < cfg_.population_size; ++i) {
            assign_placeholder_fields(population_[i]);
            sync_representation(i);
            population_[i].fitness = objective(population_[i]);
            if (population_[i].fitness > best.fitness) {
                best = population_[i];
                best_idx = i;
            }
        }

        if (termination(best)) {
            break;
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

namespace {

// Large-but-finite penalty substituted for any row whose prediction (or squared
// error) is non-finite, so a single malformed individual cannot turn the whole
// aggregate fitness into NaN/Inf and destroy selection pressure. See mse_fitness's
// doc comment in axiom.hpp for the full rationale.
constexpr double kNonFinitePenalty = 1e12;

} // namespace

double Axiom::mse_fitness(
    const Algorithm& algo, const Matrix<double>& inputs, const std::vector<double>& targets) const {
    const size_t n = std::min(inputs.rows(), targets.size());
    if (n == 0) {
        return 0.0;
    }

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        std::map<std::string, double> env;
        for (size_t j = 0; j < inputs.cols(); ++j) {
            env["x" + std::to_string(j)] = inputs(i, j);
        }
        const double prediction = algo.representation.eval(env);

        double squared_error = kNonFinitePenalty;
        if (std::isfinite(prediction)) {
            const double error = prediction - targets[i];
            const double candidate = error * error;
            if (std::isfinite(candidate)) {
                squared_error = candidate;
            }
        }
        sum_squared_error += squared_error;
    }
    return sum_squared_error / static_cast<double>(n);
}

double Axiom::rmse_fitness(
    const Algorithm& algo, const Matrix<double>& inputs, const std::vector<double>& targets) const {
    return std::sqrt(mse_fitness(algo, inputs, targets));
}

} // namespace ms::axiom
