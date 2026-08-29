#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_upwind_step_3d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_upwind_step_3d" &&
               (assign.args.size() == 6 || assign.args.size() == 9)) {
        auto grid = resolve_operand(assign.args[0]);
        if (!grid) {
            return std::unexpected(grid.error());
        }
        auto u = resolve_operand(assign.args[1]);
        if (!u) {
            return std::unexpected(u.error());
        }
        auto vx = parse_scalar_arg(assign.args[2], "cfd_upwind_step_3d");
        if (!vx) {
            return std::unexpected(vx.error());
        }
        auto vy = parse_scalar_arg(assign.args[3], "cfd_upwind_step_3d");
        if (!vy) {
            return std::unexpected(vy.error());
        }
        auto vz = parse_scalar_arg(assign.args[4], "cfd_upwind_step_3d");
        if (!vz) {
            return std::unexpected(vz.error());
        }
        auto dt = parse_scalar_arg(assign.args[5], "cfd_upwind_step_3d");
        if (!dt) {
            return std::unexpected(dt.error());
        }
        cfd::BoundaryCondition bc_x = cfd::BoundaryCondition::Periodic;
        cfd::BoundaryCondition bc_y = cfd::BoundaryCondition::Periodic;
        cfd::BoundaryCondition bc_z = cfd::BoundaryCondition::Periodic;
        if (assign.args.size() == 9) {
            auto bcx = parse_scalar_arg(assign.args[6], "cfd_upwind_step_3d");
            if (!bcx) {
                return std::unexpected(bcx.error());
            }
            auto bcy = parse_scalar_arg(assign.args[7], "cfd_upwind_step_3d");
            if (!bcy) {
                return std::unexpected(bcy.error());
            }
            auto bcz = parse_scalar_arg(assign.args[8], "cfd_upwind_step_3d");
            if (!bcz) {
                return std::unexpected(bcz.error());
            }
            auto parsed_x = parse_cfd_bc(*bcx, "cfd_upwind_step_3d");
            if (!parsed_x) {
                return std::unexpected(parsed_x.error());
            }
            auto parsed_y = parse_cfd_bc(*bcy, "cfd_upwind_step_3d");
            if (!parsed_y) {
                return std::unexpected(parsed_y.error());
            }
            auto parsed_z = parse_cfd_bc(*bcz, "cfd_upwind_step_3d");
            if (!parsed_z) {
                return std::unexpected(parsed_z.error());
            }
            bc_x = *parsed_x;
            bc_y = *parsed_y;
            bc_z = *parsed_z;
        }
        result = eval_cfd_upwind_step_3d(*grid, *u, *vx, *vy, *vz, *dt, bc_x, bc_y, bc_z);
    }

    return result;
}

void ms_register_matrix_call_cfd_upwind_step_3d() {
    register_matrix_call("cfd_upwind_step_3d", &handle_cfd_upwind_step_3d);
}

} // namespace ms::interp
