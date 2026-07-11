#include "ms/symbolic/symbolic.hpp"

#include <cmath>

namespace ms {

namespace {

SymExpr clone_expr(const SymExpr& expr) {
    SymExpr copy;
    copy.op = expr.op;
    copy.value = expr.value;
    copy.name = expr.name;
    if (expr.left) {
        copy.left = std::make_unique<SymExpr>(clone_expr(*expr.left));
    }
    if (expr.right) {
        copy.right = std::make_unique<SymExpr>(clone_expr(*expr.right));
    }
    return copy;
}

bool is_bare_var(const SymExpr& expr, const std::string& var) {
    return expr.op == SymOp::Var && expr.name == var;
}

bool is_const_zero(const SymExpr& expr) {
    return expr.op == SymOp::Const && expr.value == 0.0;
}

SymExpr sym_integrate_unsupported(const SymExpr& expr, const std::string& var) {
    return sym_deriv(clone_expr(expr), var);
}

} // namespace

SymExpr sym_const(double value) {
    SymExpr e;
    e.op = SymOp::Const;
    e.value = value;
    return e;
}

SymExpr sym_var(const std::string& name) {
    SymExpr e;
    e.op = SymOp::Var;
    e.name = name;
    return e;
}

SymExpr sym_add(SymExpr a, SymExpr b) {
    SymExpr e;
    e.op = SymOp::Add;
    e.left = std::make_unique<SymExpr>(std::move(a));
    e.right = std::make_unique<SymExpr>(std::move(b));
    return e;
}

SymExpr sym_mul(SymExpr a, SymExpr b) {
    SymExpr e;
    e.op = SymOp::Mul;
    e.left = std::make_unique<SymExpr>(std::move(a));
    e.right = std::make_unique<SymExpr>(std::move(b));
    return e;
}

SymExpr sym_sub(SymExpr a, SymExpr b) {
    SymExpr e;
    e.op = SymOp::Sub;
    e.left = std::make_unique<SymExpr>(std::move(a));
    e.right = std::make_unique<SymExpr>(std::move(b));
    return e;
}

SymExpr sym_div(SymExpr a, SymExpr b) {
    SymExpr e;
    e.op = SymOp::Div;
    e.left = std::make_unique<SymExpr>(std::move(a));
    e.right = std::make_unique<SymExpr>(std::move(b));
    return e;
}

SymExpr sym_unary(SymOp op, SymExpr arg) {
    SymExpr e;
    e.op = op;
    e.left = std::make_unique<SymExpr>(std::move(arg));
    return e;
}

SymExpr sym_neg(SymExpr arg) {
    return sym_unary(SymOp::Neg, std::move(arg));
}

SymExpr sym_sin(SymExpr arg) {
    return sym_unary(SymOp::Sin, std::move(arg));
}

SymExpr sym_cos(SymExpr arg) {
    return sym_unary(SymOp::Cos, std::move(arg));
}

SymExpr sym_tan(SymExpr arg) {
    return sym_unary(SymOp::Tan, std::move(arg));
}

SymExpr sym_exp(SymExpr arg) {
    return sym_unary(SymOp::Exp, std::move(arg));
}

SymExpr sym_log(SymExpr arg) {
    return sym_unary(SymOp::Log, std::move(arg));
}

SymExpr sym_sqrt(SymExpr arg) {
    return sym_unary(SymOp::Sqrt, std::move(arg));
}

SymExpr sym_pow(SymExpr base, SymExpr exponent) {
    SymExpr e;
    e.op = SymOp::Pow;
    e.left = std::make_unique<SymExpr>(std::move(base));
    e.right = std::make_unique<SymExpr>(std::move(exponent));
    return e;
}

SymExpr sym_deriv(SymExpr expr, const std::string& var) {
    SymExpr e;
    e.op = SymOp::Deriv;
    e.name = var;
    e.left = std::make_unique<SymExpr>(std::move(expr));
    return e;
}

SymExpr sym_diff(SymExpr expr, const std::string& var) {
    switch (expr.op) {
    case SymOp::Const:
        return sym_const(0.0);
    case SymOp::Var:
        return sym_const(expr.name == var ? 1.0 : 0.0);
    case SymOp::Add:
        return sym_add(sym_diff(clone_expr(*expr.left), var), sym_diff(clone_expr(*expr.right), var));
    case SymOp::Sub:
        return sym_sub(sym_diff(clone_expr(*expr.left), var), sym_diff(clone_expr(*expr.right), var));
    case SymOp::Mul:
        return sym_add(
            sym_mul(sym_diff(clone_expr(*expr.left), var), clone_expr(*expr.right)),
            sym_mul(clone_expr(*expr.left), sym_diff(clone_expr(*expr.right), var)));
    case SymOp::Div:
        return sym_div(
            sym_sub(
                sym_mul(sym_diff(clone_expr(*expr.left), var), clone_expr(*expr.right)),
                sym_mul(clone_expr(*expr.left), sym_diff(clone_expr(*expr.right), var))),
            sym_pow(clone_expr(*expr.right), sym_const(2.0)));
    case SymOp::Neg:
        return sym_neg(sym_diff(clone_expr(*expr.left), var));
    case SymOp::Sin:
        return sym_mul(sym_cos(clone_expr(*expr.left)), sym_diff(clone_expr(*expr.left), var));
    case SymOp::Cos:
        return sym_mul(
            sym_mul(sym_const(-1.0), sym_sin(clone_expr(*expr.left))),
            sym_diff(clone_expr(*expr.left), var));
    case SymOp::Tan:
        return sym_mul(
            sym_add(sym_const(1.0), sym_pow(sym_tan(clone_expr(*expr.left)), sym_const(2.0))),
            sym_diff(clone_expr(*expr.left), var));
    case SymOp::Exp:
        return sym_mul(clone_expr(expr), sym_diff(clone_expr(*expr.left), var));
    case SymOp::Log:
        return sym_mul(
            sym_diff(clone_expr(*expr.left), var),
            sym_pow(clone_expr(*expr.left), sym_const(-1.0)));
    case SymOp::Sqrt:
        return sym_mul(
            sym_div(sym_const(0.5), sym_sqrt(clone_expr(*expr.left))),
            sym_diff(clone_expr(*expr.left), var));
    case SymOp::Pow:
        if (expr.right->op == SymOp::Const) {
            const double n = expr.right->value;
            return sym_mul(
                sym_const(n),
                sym_mul(
                    sym_pow(clone_expr(*expr.left), sym_const(n - 1.0)),
                    sym_diff(clone_expr(*expr.left), var)));
        }
        if (expr.left->op == SymOp::Const && expr.left->value > 0.0) {
            return sym_mul(
                sym_pow(clone_expr(*expr.left), clone_expr(*expr.right)),
                sym_mul(sym_const(std::log(expr.left->value)), sym_diff(clone_expr(*expr.right), var)));
        }
        return sym_diff(
            sym_exp(sym_mul(clone_expr(*expr.right), sym_log(clone_expr(*expr.left)))),
            var);
    case SymOp::Deriv:
        return sym_deriv(clone_expr(*expr.left), var);
    }
    return sym_const(0.0);
}

SymExpr sym_simplify(SymExpr expr) {
    if (expr.left) {
        expr.left = std::make_unique<SymExpr>(sym_simplify(clone_expr(*expr.left)));
    }
    if (expr.right) {
        expr.right = std::make_unique<SymExpr>(sym_simplify(clone_expr(*expr.right)));
    }

    switch (expr.op) {
    case SymOp::Add:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const) {
            return sym_const(expr.left->value + expr.right->value);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return clone_expr(*expr.right);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 0.0) {
            return clone_expr(*expr.left);
        }
        break;
    case SymOp::Sub:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const) {
            return sym_const(expr.left->value - expr.right->value);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 0.0) {
            return clone_expr(*expr.left);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return sym_neg(clone_expr(*expr.right));
        }
        break;
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const) {
            return sym_const(expr.left->value * expr.right->value);
        }
        if ((expr.left->op == SymOp::Const && expr.left->value == 0.0) ||
            (expr.right->op == SymOp::Const && expr.right->value == 0.0)) {
            return sym_const(0.0);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 1.0) {
            return clone_expr(*expr.right);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 1.0) {
            return clone_expr(*expr.left);
        }
        break;
    case SymOp::Div:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const &&
            expr.right->value != 0.0) {
            return sym_const(expr.left->value / expr.right->value);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 1.0) {
            return clone_expr(*expr.left);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0 && !is_const_zero(*expr.right)) {
            return sym_const(0.0);
        }
        break;
    case SymOp::Neg:
        if (expr.left->op == SymOp::Const) {
            return sym_const(-expr.left->value);
        }
        if (expr.left->op == SymOp::Neg) {
            return clone_expr(*expr.left->left);
        }
        break;
    case SymOp::Sin:
        if (expr.left->op == SymOp::Const) {
            return sym_const(std::sin(expr.left->value));
        }
        break;
    case SymOp::Cos:
        if (expr.left->op == SymOp::Const) {
            return sym_const(std::cos(expr.left->value));
        }
        break;
    case SymOp::Tan:
        if (expr.left->op == SymOp::Const) {
            return sym_const(std::tan(expr.left->value));
        }
        break;
    case SymOp::Sqrt:
        if (expr.left->op == SymOp::Const && expr.left->value >= 0.0) {
            return sym_const(std::sqrt(expr.left->value));
        }
        break;
    case SymOp::Exp:
        if (expr.left->op == SymOp::Log) {
            return clone_expr(*expr.left->left);
        }
        if (expr.left->op == SymOp::Const) {
            return sym_const(std::exp(expr.left->value));
        }
        break;
    case SymOp::Log:
        if (expr.left->op == SymOp::Exp) {
            return clone_expr(*expr.left->left);
        }
        if (expr.left->op == SymOp::Const && expr.left->value > 0.0) {
            return sym_const(std::log(expr.left->value));
        }
        break;
    case SymOp::Pow:
        if (expr.right->op == SymOp::Const && expr.right->value == 0.0) {
            return sym_const(1.0);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 1.0) {
            return clone_expr(*expr.left);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 1.0) {
            return sym_const(1.0);
        }
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const) {
            return sym_const(std::pow(expr.left->value, expr.right->value));
        }
        break;
    default:
        break;
    }

    return expr;
}

// Supported forms:
//   Const c                          -> c * var
//   Var v (v == var)                 -> var^2 / 2
//   Var v (v != var)                 -> v * var  (treat as constant w.r.t. var)
//   Add / Sub / Neg                  -> integrate each child (linearity)
//   Mul with one Const factor        -> const * integrate(other)
//   Pow(var, Const n), n != -1       -> var^(n+1) / (n+1)
//   Sin(var) / Cos(var)              -> -Cos(var) / Sin(var)  (bare integration variable only)
// Unsupported forms (chain rule, general products, Pow with n == -1, etc.)
// return sym_deriv(expr, var) as an explicit unsupported sentinel.
SymExpr sym_integrate(const SymExpr& expr, const std::string& var) {
    switch (expr.op) {
    case SymOp::Const:
        return sym_mul(clone_expr(expr), sym_var(var));
    case SymOp::Var:
        if (expr.name == var) {
            return sym_div(sym_pow(sym_var(var), sym_const(2.0)), sym_const(2.0));
        }
        return sym_mul(clone_expr(expr), sym_var(var));
    case SymOp::Add:
        return sym_add(
            sym_integrate(*expr.left, var),
            sym_integrate(*expr.right, var));
    case SymOp::Sub:
        return sym_sub(
            sym_integrate(*expr.left, var),
            sym_integrate(*expr.right, var));
    case SymOp::Neg:
        return sym_neg(sym_integrate(*expr.left, var));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_integrate(*expr.right, var));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_integrate(*expr.left, var));
        }
        return sym_integrate_unsupported(expr, var);
    case SymOp::Pow:
        if (is_bare_var(*expr.left, var) && expr.right->op == SymOp::Const) {
            const double n = expr.right->value;
            if (n == -1.0) {
                return sym_integrate_unsupported(expr, var);
            }
            return sym_div(sym_pow(sym_var(var), sym_const(n + 1.0)), sym_const(n + 1.0));
        }
        return sym_integrate_unsupported(expr, var);
    case SymOp::Sin:
        if (is_bare_var(*expr.left, var)) {
            return sym_neg(sym_cos(sym_var(var)));
        }
        return sym_integrate_unsupported(expr, var);
    case SymOp::Cos:
        if (is_bare_var(*expr.left, var)) {
            return sym_sin(sym_var(var));
        }
        return sym_integrate_unsupported(expr, var);
    default:
        return sym_integrate_unsupported(expr, var);
    }
}

SymExpr sym_substitute(const SymExpr& expr, const std::string& var, const SymExpr& replacement) {
    if (expr.op == SymOp::Var && expr.name == var) {
        return clone_expr(replacement);
    }
    if (expr.op == SymOp::Var || expr.op == SymOp::Const) {
        return clone_expr(expr);
    }

    SymExpr result;
    result.op = expr.op;
    result.value = expr.value;
    result.name = expr.name;
    if (expr.left) {
        result.left = std::make_unique<SymExpr>(sym_substitute(*expr.left, var, replacement));
    }
    if (expr.right) {
        result.right = std::make_unique<SymExpr>(sym_substitute(*expr.right, var, replacement));
    }
    return result;
}

double sym_eval(const SymExpr& expr, const std::map<std::string, double>& env) {
    switch (expr.op) {
    case SymOp::Const:
        return expr.value;
    case SymOp::Var: {
        const auto it = env.find(expr.name);
        return it == env.end() ? 0.0 : it->second;
    }
    case SymOp::Add:
        return sym_eval(*expr.left, env) + sym_eval(*expr.right, env);
    case SymOp::Sub:
        return sym_eval(*expr.left, env) - sym_eval(*expr.right, env);
    case SymOp::Mul:
        return sym_eval(*expr.left, env) * sym_eval(*expr.right, env);
    case SymOp::Div:
        return sym_eval(*expr.left, env) / sym_eval(*expr.right, env);
    case SymOp::Neg:
        return -sym_eval(*expr.left, env);
    case SymOp::Sin:
        return std::sin(sym_eval(*expr.left, env));
    case SymOp::Cos:
        return std::cos(sym_eval(*expr.left, env));
    case SymOp::Tan:
        return std::tan(sym_eval(*expr.left, env));
    case SymOp::Exp:
        return std::exp(sym_eval(*expr.left, env));
    case SymOp::Log:
        return std::log(sym_eval(*expr.left, env));
    case SymOp::Sqrt:
        return std::sqrt(sym_eval(*expr.left, env));
    case SymOp::Pow:
        return std::pow(sym_eval(*expr.left, env), sym_eval(*expr.right, env));
    case SymOp::Deriv:
        return sym_eval(sym_diff(clone_expr(*expr.left), expr.name), env);
    }
    return 0.0;
}

std::string sym_to_string(const SymExpr& expr) {
    switch (expr.op) {
    case SymOp::Const:
        return std::to_string(expr.value);
    case SymOp::Var:
        return expr.name;
    case SymOp::Add:
        return "(" + sym_to_string(*expr.left) + " + " + sym_to_string(*expr.right) + ")";
    case SymOp::Sub:
        return "(" + sym_to_string(*expr.left) + " - " + sym_to_string(*expr.right) + ")";
    case SymOp::Mul:
        return "(" + sym_to_string(*expr.left) + " * " + sym_to_string(*expr.right) + ")";
    case SymOp::Div:
        return "(" + sym_to_string(*expr.left) + " / " + sym_to_string(*expr.right) + ")";
    case SymOp::Neg:
        return "(-" + sym_to_string(*expr.left) + ")";
    case SymOp::Sin:
        return "sin(" + sym_to_string(*expr.left) + ")";
    case SymOp::Cos:
        return "cos(" + sym_to_string(*expr.left) + ")";
    case SymOp::Tan:
        return "tan(" + sym_to_string(*expr.left) + ")";
    case SymOp::Exp:
        return "exp(" + sym_to_string(*expr.left) + ")";
    case SymOp::Log:
        return "log(" + sym_to_string(*expr.left) + ")";
    case SymOp::Sqrt:
        return "sqrt(" + sym_to_string(*expr.left) + ")";
    case SymOp::Pow:
        return "(" + sym_to_string(*expr.left) + " ^ " + sym_to_string(*expr.right) + ")";
    case SymOp::Deriv:
        return "d/d" + expr.name + "(" + sym_to_string(*expr.left) + ")";
    }
    return "?";
}

} // namespace ms
