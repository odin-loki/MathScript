#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_izaac_fuzz_mutate(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "izaac_fuzz_mutate" &&
               (assign.args.size() == 1 || assign.args.size() == 2)) {
        auto input = resolve_operand(assign.args[0]);
        if (!input) {
            return std::unexpected(input.error());
        }
        size_t max_edits = 16;
        if (assign.args.size() == 2) {
            auto edits_val = parse_scalar_arg(assign.args[1], "izaac_fuzz_mutate");
            if (!edits_val) {
                return std::unexpected(edits_val.error());
            }
            const int edits_i = static_cast<int>(*edits_val);
            if (*edits_val != edits_i || edits_i < 0) {
                return std::unexpected(
                    DomainError{"izaac_fuzz_mutate", "expected non-negative integer max_edits"});
            }
            max_edits = static_cast<size_t>(edits_i);
        }
        result = eval_izaac_fuzz_mutate(*input, max_edits);
    }

    return result;
}

void ms_register_matrix_call_izaac_fuzz_mutate() {
    register_matrix_call("izaac_fuzz_mutate", &handle_izaac_fuzz_mutate);
}

} // namespace ms::interp
