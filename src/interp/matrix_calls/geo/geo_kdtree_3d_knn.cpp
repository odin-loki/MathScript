#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_geo_kdtree_3d_knn(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if ((assign.callee == "geo_kdtree_3d_knn" || assign.callee == "geo_kdtree_3d_range") &&
               assign.args.size() == 5) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        auto qx = parse_scalar_arg(assign.args[1], assign.callee.c_str());
        if (!qx) {
            return std::unexpected(qx.error());
        }
        auto qy = parse_scalar_arg(assign.args[2], assign.callee.c_str());
        if (!qy) {
            return std::unexpected(qy.error());
        }
        auto qz = parse_scalar_arg(assign.args[3], assign.callee.c_str());
        if (!qz) {
            return std::unexpected(qz.error());
        }
        auto arg4 = parse_scalar_arg(assign.args[4], assign.callee.c_str());
        if (!arg4) {
            return std::unexpected(arg4.error());
        }
        if (assign.callee == "geo_kdtree_3d_knn") {
            result = eval_geo_kdtree_3d_knn(*matrix, *qx, *qy, *qz, *arg4);
        } else {
            result = eval_geo_kdtree_3d_range(*matrix, *qx, *qy, *qz, *arg4);
        }
    }

    return result;
}

void ms_register_matrix_call_geo_kdtree_3d_knn() {
    register_matrix_call("geo_kdtree_3d_knn", &handle_geo_kdtree_3d_knn);
}

} // namespace ms::interp
