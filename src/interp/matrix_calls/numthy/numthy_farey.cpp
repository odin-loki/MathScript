#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_numthy_farey(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "numthy_factor_exp" || assign.callee == "numthy_farey" ||
                assign.callee == "numthy_stern_brocot" ||
                assign.callee == "numthy_pell_solve" ||
                assign.callee == "numthy_quadratic_residues") &&
               assign.args.size() == 1) {
        double n_d = 0.0;
        if (!parse_number(assign.args[0], n_d)) {
            auto n_expr = eval_scalar_expr(ctx.state(), assign.args[0]);
            if (!n_expr) {
                return std::unexpected(DomainError{
                    assign.callee,
                    assign.callee == "numthy_factor_exp"
                        ? "expected numthy_factor_exp(n)"
                        : assign.callee == "numthy_farey"
                              ? "expected numthy_farey(n)"
                              : assign.callee == "numthy_stern_brocot"
                                    ? "expected numthy_stern_brocot(n)"
                                    : assign.callee == "numthy_pell_solve"
                                          ? "expected numthy_pell_solve(D)"
                                          : "expected numthy_quadratic_residues(p)"});
            }
            n_d = *n_expr;
        }
        const int n = static_cast<int>(n_d);
        if (assign.callee == "numthy_factor_exp") {
            if (n < 2 || n_d != n) {
                return std::unexpected(
                    DomainError{"numthy_factor_exp", "expected integer n >= 2"});
            }
            result = eval_numthy_factor_exp(n);
        } else if (assign.callee == "numthy_farey") {
            if (n < 1 || n_d != n) {
                return std::unexpected(
                    DomainError{"numthy_farey", "expected positive integer n"});
            }
            result = eval_numthy_farey(n);
        } else if (assign.callee == "numthy_stern_brocot") {
            if (n < 0 || n_d != n) {
                return std::unexpected(
                    DomainError{"numthy_stern_brocot", "expected non-negative integer n"});
            }
            result = eval_numthy_stern_brocot(n);
        } else if (assign.callee == "numthy_pell_solve") {
            if (n < 1 || n_d != n) {
                return std::unexpected(
                    DomainError{"numthy_pell_solve", "expected positive integer D"});
            }
            result = eval_numthy_pell_solve(n);
        } else {
            if (n < 3 || n_d != n) {
                return std::unexpected(
                    DomainError{"numthy_quadratic_residues", "expected odd prime p >= 3"});
            }
            result = eval_numthy_quadratic_residues(n);
        }
    }

    return result;
}

void ms_register_matrix_call_numthy_farey() {
    register_matrix_call("numthy_farey", &handle_numthy_farey);
}

} // namespace ms::interp
