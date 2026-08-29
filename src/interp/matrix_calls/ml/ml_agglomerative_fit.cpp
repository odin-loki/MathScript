#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_ml_agglomerative_fit(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "ml_agglomerative_fit" &&
               (assign.args.size() >= 1 && assign.args.size() <= 3)) {
        auto X = resolve_operand(assign.args[0]);
        if (!X) {
            return std::unexpected(X.error());
        }
        int n_clusters = 2;
        std::string linkage = "ward";
        if (assign.args.size() >= 2) {
            auto k_arg = parse_scalar_arg(assign.args[1], "ml_agglomerative_fit");
            if (!k_arg) {
                return std::unexpected(k_arg.error());
            }
            n_clusters = static_cast<int>(*k_arg);
            if (n_clusters < 1 || *k_arg != n_clusters) {
                return std::unexpected(
                    DomainError{"ml_agglomerative_fit", "expected integer n_clusters >= 1"});
            }
        }
        if (assign.args.size() >= 3) {
            auto linkage_arg = parse_ml_linkage(assign.args[2], "ml_agglomerative_fit");
            if (!linkage_arg) {
                return std::unexpected(linkage_arg.error());
            }
            linkage = *linkage_arg;
        }
        auto labels = eval_ml_agglomerative_fit(*X, n_clusters, linkage);
        if (!labels) {
            return std::unexpected(labels.error());
        }
        result = *labels;
    }

    return result;
}

void ms_register_matrix_call_ml_agglomerative_fit() {
    register_matrix_call("ml_agglomerative_fit", &handle_ml_agglomerative_fit);
}

} // namespace ms::interp
