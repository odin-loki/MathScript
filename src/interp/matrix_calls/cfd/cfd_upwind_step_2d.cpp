#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cfd_upwind_step_2d(Interpreter& interp, const MatrixCallAssign& assign) {
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
    if (assign.callee == "cfd_upwind_step_2d" &&
               (assign.args.size() >= 5 && assign.args.size() <= 8)) {
        auto first = resolve_operand(assign.args[0]);
        if (!first) {
            return std::unexpected(first.error());
        }
        if (cfd_grid2d_from_packed_matrix(*first, "cfd_upwind_step_2d") &&
            assign.args.size() <= 7) {
            auto grid = first;
            auto u = resolve_operand(assign.args[1]);
            if (!u) {
                return std::unexpected(u.error());
            }
            auto vx = parse_scalar_arg(assign.args[2], "cfd_upwind_step_2d");
            if (!vx) {
                return std::unexpected(vx.error());
            }
            auto vy = parse_scalar_arg(assign.args[3], "cfd_upwind_step_2d");
            if (!vy) {
                return std::unexpected(vy.error());
            }
            auto dt = parse_scalar_arg(assign.args[4], "cfd_upwind_step_2d");
            if (!dt) {
                return std::unexpected(dt.error());
            }
            cfd::BoundaryCondition bc_x = cfd::BoundaryCondition::Periodic;
            cfd::BoundaryCondition bc_y = cfd::BoundaryCondition::Periodic;
            if (assign.args.size() >= 6) {
                auto bcx = parse_scalar_arg(assign.args[5], "cfd_upwind_step_2d");
                if (!bcx) {
                    return std::unexpected(bcx.error());
                }
                auto parsed = parse_cfd_bc(*bcx, "cfd_upwind_step_2d");
                if (!parsed) {
                    return std::unexpected(parsed.error());
                }
                bc_x = *parsed;
            }
            if (assign.args.size() == 7) {
                auto bcy = parse_scalar_arg(assign.args[6], "cfd_upwind_step_2d");
                if (!bcy) {
                    return std::unexpected(bcy.error());
                }
                auto parsed = parse_cfd_bc(*bcy, "cfd_upwind_step_2d");
                if (!parsed) {
                    return std::unexpected(parsed.error());
                }
                bc_y = *parsed;
            }
            result = eval_cfd_upwind_step_2d_from_grid(*grid, *u, *vx, *vy, *dt, bc_x, bc_y);
        } else if (assign.args.size() >= 6 && assign.args.size() <= 8) {
            auto u = first;
            auto vx = parse_scalar_arg(assign.args[1], "cfd_upwind_step_2d");
            if (!vx) {
                return std::unexpected(vx.error());
            }
            auto vy = parse_scalar_arg(assign.args[2], "cfd_upwind_step_2d");
            if (!vy) {
                return std::unexpected(vy.error());
            }
            auto dt = parse_scalar_arg(assign.args[3], "cfd_upwind_step_2d");
            if (!dt) {
                return std::unexpected(dt.error());
            }
            auto dx = parse_scalar_arg(assign.args[4], "cfd_upwind_step_2d");
            if (!dx) {
                return std::unexpected(dx.error());
            }
            auto dy = parse_scalar_arg(assign.args[5], "cfd_upwind_step_2d");
            if (!dy) {
                return std::unexpected(dy.error());
            }
            cfd::BoundaryCondition bc_x = cfd::BoundaryCondition::Periodic;
            cfd::BoundaryCondition bc_y = cfd::BoundaryCondition::Periodic;
            if (assign.args.size() >= 7) {
                auto bcx = parse_scalar_arg(assign.args[6], "cfd_upwind_step_2d");
                if (!bcx) {
                    return std::unexpected(bcx.error());
                }
                auto parsed = parse_cfd_bc(*bcx, "cfd_upwind_step_2d");
                if (!parsed) {
                    return std::unexpected(parsed.error());
                }
                bc_x = *parsed;
            }
            if (assign.args.size() == 8) {
                auto bcy = parse_scalar_arg(assign.args[7], "cfd_upwind_step_2d");
                if (!bcy) {
                    return std::unexpected(bcy.error());
                }
                auto parsed = parse_cfd_bc(*bcy, "cfd_upwind_step_2d");
                if (!parsed) {
                    return std::unexpected(parsed.error());
                }
                bc_y = *parsed;
            }
            result = eval_cfd_upwind_step_2d(*u, *vx, *vy, *dt, *dx, *dy, bc_x, bc_y);
        }
    }

    return result;
}

void ms_register_matrix_call_cfd_upwind_step_2d() {
    register_matrix_call("cfd_upwind_step_2d", &handle_cfd_upwind_step_2d);
}

} // namespace ms::interp
