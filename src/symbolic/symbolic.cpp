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

SymExpr sym_unary(SymOp op, SymExpr arg) {
    SymExpr e;
    e.op = op;
    e.left = std::make_unique<SymExpr>(std::move(arg));
    return e;
}

SymExpr sym_sin(SymExpr arg) {
    return sym_unary(SymOp::Sin, std::move(arg));
}

SymExpr sym_cos(SymExpr arg) {
    return sym_unary(SymOp::Cos, std::move(arg));
}

SymExpr sym_exp(SymExpr arg) {
    return sym_unary(SymOp::Exp, std::move(arg));
}

SymExpr sym_log(SymExpr arg) {
    return sym_unary(SymOp::Log, std::move(arg));
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
    case SymOp::Mul:
        return sym_add(
            sym_mul(sym_diff(clone_expr(*expr.left), var), clone_expr(*expr.right)),
            sym_mul(clone_expr(*expr.left), sym_diff(clone_expr(*expr.right), var)));
    case SymOp::Sin:
        return sym_mul(sym_cos(clone_expr(*expr.left)), sym_diff(clone_expr(*expr.left), var));
    case SymOp::Cos:
        return sym_mul(
            sym_mul(sym_const(-1.0), sym_sin(clone_expr(*expr.left))),
            sym_diff(clone_expr(*expr.left), var));
    case SymOp::Exp:
        return sym_mul(clone_expr(expr), sym_diff(clone_expr(*expr.left), var));
    case SymOp::Log:
        return sym_mul(
            sym_diff(clone_expr(*expr.left), var),
            sym_pow(clone_expr(*expr.left), sym_const(-1.0)));
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
    case SymOp::Sin:
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return sym_const(0.0);
        }
        break;
    case SymOp::Cos:
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return sym_const(1.0);
        }
        break;
    case SymOp::Exp:
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return sym_const(1.0);
        }
        break;
    case SymOp::Log:
        if (expr.left->op == SymOp::Const && expr.left->value == 1.0) {
            return sym_const(0.0);
        }
        if (expr.left->op == SymOp::Exp) {
            return clone_expr(*expr.left->left);
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
    case SymOp::Mul:
        return sym_eval(*expr.left, env) * sym_eval(*expr.right, env);
    case SymOp::Sin:
        return std::sin(sym_eval(*expr.left, env));
    case SymOp::Cos:
        return std::cos(sym_eval(*expr.left, env));
    case SymOp::Exp:
        return std::exp(sym_eval(*expr.left, env));
    case SymOp::Log:
        return std::log(sym_eval(*expr.left, env));
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
    case SymOp::Mul:
        return "(" + sym_to_string(*expr.left) + " * " + sym_to_string(*expr.right) + ")";
    case SymOp::Sin:
        return "sin(" + sym_to_string(*expr.left) + ")";
    case SymOp::Cos:
        return "cos(" + sym_to_string(*expr.left) + ")";
    case SymOp::Exp:
        return "exp(" + sym_to_string(*expr.left) + ")";
    case SymOp::Log:
        return "log(" + sym_to_string(*expr.left) + ")";
    case SymOp::Pow:
        return "(" + sym_to_string(*expr.left) + " ^ " + sym_to_string(*expr.right) + ")";
    case SymOp::Deriv:
        return "d/d" + expr.name + "(" + sym_to_string(*expr.left) + ")";
    }
    return "?";
}

} // namespace ms
