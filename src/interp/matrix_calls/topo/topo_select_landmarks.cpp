#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_topo_select_landmarks(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "topo_select_landmarks" &&
               (assign.args.size() == 2 || assign.args.size() == 3)) {
        auto P = resolve_operand(assign.args[0]);
        if (!P) {
            return std::unexpected(P.error());
        }
        auto n_arg = parse_scalar_arg(assign.args[1], "topo_select_landmarks");
        if (!n_arg) {
            return std::unexpected(n_arg.error());
        }
        const int n_landmarks = static_cast<int>(*n_arg);
        if (n_landmarks < 1 || *n_arg != n_landmarks) {
            return std::unexpected(
                DomainError{"topo_select_landmarks", "expected positive integer n"});
        }
        int seed_index = 0;
        if (assign.args.size() == 3) {
            auto seed = parse_scalar_arg(assign.args[2], "topo_select_landmarks");
            if (!seed) {
                return std::unexpected(seed.error());
            }
            seed_index = static_cast<int>(*seed);
            if (*seed != seed_index) {
                return std::unexpected(
                    DomainError{"topo_select_landmarks", "expected integer seed_index"});
            }
        }
        result = eval_topo_select_landmarks(*P, n_landmarks, seed_index);
    }

    return result;
}

void ms_register_matrix_call_topo_select_landmarks() {
    register_matrix_call("topo_select_landmarks", &handle_topo_select_landmarks);
}

} // namespace ms::interp
