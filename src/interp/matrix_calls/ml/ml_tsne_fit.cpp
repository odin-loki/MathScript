#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_tsne_fit(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) { return ctx.resolve_operand(text); };
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    };
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    };
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<uint64_t>(value);
    };

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "ml_tsne_fit" &&
               (assign.args.size() >= 1 && assign.args.size() <= 4)) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        double perplexity = 30.0;
        int n_iter = 250;
        unsigned seed = 42;
        if (assign.args.size() >= 2) {
            auto p_arg = parse_scalar_arg(assign.args[1], "ml_tsne_fit");
            if (!p_arg) {
                return std::unexpected(p_arg.error());
            }
            perplexity = *p_arg;
        }
        if (assign.args.size() >= 3) {
            auto iter_arg = parse_scalar_arg(assign.args[2], "ml_tsne_fit");
            if (!iter_arg) {
                return std::unexpected(iter_arg.error());
            }
            n_iter = static_cast<int>(*iter_arg);
            if (n_iter < 1 || *iter_arg != n_iter) {
                return std::unexpected(
                    DomainError{"ml_tsne_fit", "expected positive integer n_iter"});
            }
        }
        if (assign.args.size() >= 4) {
            auto seed_val = parse_optional_seed(assign.args[3], "ml_tsne_fit");
            if (!seed_val) {
                return std::unexpected(seed_val.error());
            }
            seed = *seed_val;
        }
        auto embedding = eval_ml_tsne_fit(*X, perplexity, n_iter, seed);
        if (!embedding) {
            return std::unexpected(embedding.error());
        }
        result = *embedding;
    }

    return result;
}

void ms_register_matrix_call_ml_tsne_fit() {
    register_matrix_call("ml_tsne_fit", &handle_ml_tsne_fit);
}

} // namespace ms::interp
