#include "ms/symbolic/symbolic.hpp"

#include <cmath>
#include <optional>
#include <algorithm>
#include <vector>
#include <numbers>

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

SymExpr take_child(std::unique_ptr<SymExpr>& child) {
    return std::move(*child);
}

bool is_bare_var(const SymExpr& expr, const std::string& var) {
    return expr.op == SymOp::Var && expr.name == var;
}

bool contains_var_name(const SymExpr& expr, const std::string& name);

bool is_const_zero(const SymExpr& expr) {
    return expr.op == SymOp::Const && expr.value == 0.0;
}

SymExpr sym_integrate_unsupported(const SymExpr& expr, const std::string& var) {
    return sym_deriv(clone_expr(expr), var);
}

constexpr int kMaxLaplacePower = 8;

bool try_get_const_value(const SymExpr& expr, double& out) {
    if (expr.op != SymOp::Const) {
        return false;
    }
    out = expr.value;
    return true;
}

double factorial_int(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= static_cast<double>(i);
    }
    return result;
}

bool is_small_nonneg_int(double value, int& out) {
    if (value < 0.0 || value > static_cast<double>(kMaxLaplacePower) || value != std::floor(value)) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool match_scaled_var(const SymExpr& expr, const std::string& var, double& scale) {
    if (is_bare_var(expr, var)) {
        scale = 1.0;
        return true;
    }
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const && is_bare_var(*expr.right, var)) {
            scale = expr.left->value;
            return true;
        }
        if (expr.right->op == SymOp::Const && is_bare_var(*expr.left, var)) {
            scale = expr.right->value;
            return true;
        }
    }
    return false;
}

SymExpr build_s2_plus_a2(const std::string& s, double a) {
    return sym_add(sym_pow(sym_var(s), sym_const(2.0)), sym_pow(sym_const(a), sym_const(2.0)));
}

bool is_var_pow(const SymExpr& expr, const std::string& var, double power) {
    return expr.op == SymOp::Pow && expr.left && expr.right && is_bare_var(*expr.left, var) &&
           expr.right->op == SymOp::Const && expr.right->value == power;
}

bool match_sub_var_minus_const(const SymExpr& expr, const std::string& var, double& a) {
    return expr.op == SymOp::Sub && expr.left && expr.right && is_bare_var(*expr.left, var) &&
           try_get_const_value(*expr.right, a);
}

bool match_add_var_plus_const(const SymExpr& expr, const std::string& var, double& a) {
    return expr.op == SymOp::Add && expr.left && expr.right && is_bare_var(*expr.left, var) &&
           try_get_const_value(*expr.right, a);
}

bool match_s2_plus_a2(const SymExpr& expr, const std::string& s, double& a) {
    if (expr.op != SymOp::Add || !expr.left || !expr.right) {
        return false;
    }
    const SymExpr* s_squared = nullptr;
    const SymExpr* a_squared = nullptr;
    if (is_var_pow(*expr.left, s, 2.0)) {
        s_squared = expr.left.get();
        a_squared = expr.right.get();
    } else if (is_var_pow(*expr.right, s, 2.0)) {
        s_squared = expr.right.get();
        a_squared = expr.left.get();
    } else {
        return false;
    }
    (void)s_squared;
    if (a_squared->op == SymOp::Pow && a_squared->left && a_squared->left->op == SymOp::Const &&
        a_squared->right && a_squared->right->op == SymOp::Const && a_squared->right->value == 2.0) {
        a = a_squared->left->value;
        return true;
    }
    if (a_squared->op == SymOp::Const && a_squared->value >= 0.0) {
        a = std::sqrt(a_squared->value);
        return true;
    }
    return false;
}

SymExpr sym_laplace_unsupported(const SymExpr& expr, const std::string& t) {
    return sym_deriv(clone_expr(expr), t);
}

SymExpr sym_ilaplace_unsupported(const SymExpr& expr, const std::string& s) {
    return sym_deriv(clone_expr(expr), s);
}

SymExpr sym_mellin_unsupported(const SymExpr& expr, const std::string& t) {
    return sym_deriv(clone_expr(expr), t);
}

SymExpr sym_imellin_unsupported(const SymExpr& expr, const std::string& s) {
    return sym_deriv(clone_expr(expr), s);
}

SymExpr sym_hankel_unsupported(const SymExpr& expr, const std::string& r) {
    return sym_deriv(clone_expr(expr), r);
}

SymExpr sym_ihankel_unsupported(const SymExpr& expr, const std::string& k) {
    return sym_deriv(clone_expr(expr), k);
}

constexpr int kMaxMellinPower = 8;
constexpr int kMaxHankelPower = 8;

double hankel_rpow_exp_scale(int n) {
    const double exponent = (static_cast<double>(n) + 3.0) / 2.0;
    return std::pow(2.0, n + 1) * std::tgamma(exponent) / std::sqrt(std::numbers::pi);
}

SymExpr build_k2_plus_a2(const std::string& k, double a) {
    return build_s2_plus_a2(k, a);
}

bool match_one_over_sqrt_r2_plus_a2(const SymExpr& expr, const std::string& r_var, double& a) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return false;
    }
    double numerator = 0.0;
    if (!try_get_const_value(*expr.left, numerator) || numerator != 1.0) {
        return false;
    }
    if (expr.right->op != SymOp::Sqrt || !expr.right->left) {
        return false;
    }
    return match_s2_plus_a2(*expr.right->left, r_var, a);
}

bool match_exp_neg_ak_over_k(const SymExpr& expr, const std::string& k_var, double& a) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right || !is_bare_var(*expr.right, k_var)) {
        return false;
    }
    if (expr.left->op != SymOp::Exp || !expr.left->left) {
        return false;
    }
    const SymExpr& inner = *expr.left->left;
    if (inner.op != SymOp::Neg || !inner.left) {
        return false;
    }
    return match_scaled_var(*inner.left, k_var, a) && a > 0.0;
}

std::optional<std::pair<int, double>> match_hankel_k_domain_rpow_exp(
    const SymExpr& expr, const std::string& k_var) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return std::nullopt;
    }
    if (expr.right->op != SymOp::Pow || !expr.right->left || !expr.right->right ||
        expr.right->right->op != SymOp::Const) {
        return std::nullopt;
    }
    double a = 0.0;
    if (!match_s2_plus_a2(*expr.right->left, k_var, a) || a <= 0.0) {
        return std::nullopt;
    }
    const double exp_power = expr.right->right->value;
    const int n = static_cast<int>(std::lround(2.0 * exp_power - 3.0));
    if (n < 0 || n > kMaxHankelPower ||
        exp_power != (static_cast<double>(n) + 3.0) / 2.0) {
        return std::nullopt;
    }
    double numerator = 0.0;
    if (try_get_const_value(*expr.left, numerator)) {
        if (numerator == hankel_rpow_exp_scale(n) * a) {
            return std::make_pair(n, a);
        }
        return std::nullopt;
    }
    if (expr.left->op != SymOp::Mul || !expr.left->left || !expr.left->right) {
        return std::nullopt;
    }
    double scale = 0.0;
    double a_factor = 0.0;
    if (expr.left->left->op == SymOp::Const && expr.left->right->op == SymOp::Const) {
        scale = expr.left->left->value;
        a_factor = expr.left->right->value;
    } else if (expr.left->right->op == SymOp::Const && expr.left->left->op == SymOp::Const) {
        scale = expr.left->right->value;
        a_factor = expr.left->left->value;
    } else {
        return std::nullopt;
    }
    if (scale == hankel_rpow_exp_scale(n) && a_factor == a) {
        return std::make_pair(n, a);
    }
    return std::nullopt;
}

SymExpr hankel_forward_rpow_exp_neg(int n, double a, const std::string& k) {
    const double exp_power = (static_cast<double>(n) + 3.0) / 2.0;
    return sym_div(
        sym_const(hankel_rpow_exp_scale(n) * a),
        sym_pow(build_k2_plus_a2(k, a), sym_const(exp_power)));
}

SymExpr ihankel_inverse_rpow_exp_neg(int n, double a, const std::string& r) {
    SymExpr decay = sym_exp(sym_neg(sym_mul(sym_const(a), sym_var(r))));
    if (n == 0) {
        return decay;
    }
    return sym_mul(sym_pow(sym_var(r), sym_const(static_cast<double>(n))), std::move(decay));
}

bool match_one_over_one_plus_t(const SymExpr& expr, const std::string& t_var) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return false;
    }
    double numerator = 0.0;
    if (!try_get_const_value(*expr.left, numerator) || numerator != 1.0) {
        return false;
    }
    if (expr.right->op != SymOp::Add || !expr.right->left || !expr.right->right) {
        return false;
    }
    double one = 0.0;
    const SymExpr& left = *expr.right->left;
    const SymExpr& right = *expr.right->right;
    if (try_get_const_value(left, one) && one == 1.0 && is_bare_var(right, t_var)) {
        return true;
    }
    if (try_get_const_value(right, one) && one == 1.0 && is_bare_var(left, t_var)) {
        return true;
    }
    return false;
}

bool match_exp_neg_at(const SymExpr& expr, const std::string& t_var, double& a) {
    if (expr.op != SymOp::Exp || !expr.left) {
        return false;
    }
    const SymExpr& inner = *expr.left;
    if (inner.op == SymOp::Neg && inner.left && match_scaled_var(*inner.left, t_var, a) && a > 0.0) {
        return true;
    }
    if (match_scaled_var(inner, t_var, a) && a < 0.0) {
        a = -a;
        return true;
    }
    return false;
}

std::optional<std::pair<int, double>> match_tpow_exp_neg(const SymExpr& expr, const std::string& t_var) {
    if (expr.op != SymOp::Mul || !expr.left || !expr.right) {
        return std::nullopt;
    }
    const SymExpr* pow_part = nullptr;
    const SymExpr* exp_part = nullptr;
    if (expr.left->op == SymOp::Pow && expr.right->op == SymOp::Exp) {
        pow_part = expr.left.get();
        exp_part = expr.right.get();
    } else if (expr.right->op == SymOp::Pow && expr.left->op == SymOp::Exp) {
        pow_part = expr.right.get();
        exp_part = expr.left.get();
    } else {
        return std::nullopt;
    }
    if (!is_bare_var(*pow_part->left, t_var) || pow_part->right->op != SymOp::Const) {
        return std::nullopt;
    }
    int n = 0;
    if (!is_small_nonneg_int(pow_part->right->value, n)) {
        return std::nullopt;
    }
    double a = 0.0;
    if (!match_exp_neg_at(*exp_part, t_var, a)) {
        return std::nullopt;
    }
    return std::make_pair(n, a);
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
        expr.left = std::make_unique<SymExpr>(sym_simplify(std::move(*expr.left)));
    }
    if (expr.right) {
        expr.right = std::make_unique<SymExpr>(sym_simplify(std::move(*expr.right)));
    }

    switch (expr.op) {
    case SymOp::Add:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const) {
            return sym_const(expr.left->value + expr.right->value);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return take_child(expr.right);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 0.0) {
            return take_child(expr.left);
        }
        break;
    case SymOp::Sub:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const) {
            return sym_const(expr.left->value - expr.right->value);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 0.0) {
            return take_child(expr.left);
        }
        if (expr.left->op == SymOp::Const && expr.left->value == 0.0) {
            return sym_neg(take_child(expr.right));
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
            return take_child(expr.right);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 1.0) {
            return take_child(expr.left);
        }
        break;
    case SymOp::Div:
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Const &&
            expr.right->value != 0.0) {
            return sym_const(expr.left->value / expr.right->value);
        }
        if (expr.right->op == SymOp::Const && expr.right->value == 1.0) {
            return take_child(expr.left);
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
            return take_child(expr.left->left);
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
            return take_child(expr.left->left);
        }
        if (expr.left->op == SymOp::Const) {
            return sym_const(std::exp(expr.left->value));
        }
        break;
    case SymOp::Log:
        if (expr.left->op == SymOp::Exp) {
            return take_child(expr.left->left);
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
            return take_child(expr.left);
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

namespace {

constexpr int kMaxExpandExponent = 8;

bool is_small_nonneg_integer_exponent(const SymExpr& expr, int& exponent) {
    if (expr.op != SymOp::Const) {
        return false;
    }
    const double value = expr.value;
    if (value < 0.0 || value > static_cast<double>(kMaxExpandExponent) || value != std::floor(value)) {
        return false;
    }
    exponent = static_cast<int>(value);
    return true;
}

} // namespace

SymExpr sym_expand(SymExpr expr) {
    if (expr.left) {
        expr.left = std::make_unique<SymExpr>(sym_expand(clone_expr(*expr.left)));
    }
    if (expr.right) {
        expr.right = std::make_unique<SymExpr>(sym_expand(clone_expr(*expr.right)));
    }

    switch (expr.op) {
    case SymOp::Pow: {
        int exponent = 0;
        if (!is_small_nonneg_integer_exponent(*expr.right, exponent)) {
            break;
        }
        if (exponent == 0) {
            return sym_simplify(sym_const(1.0));
        }
        if (exponent == 1) {
            return sym_simplify(clone_expr(*expr.left));
        }
        SymExpr result = clone_expr(*expr.left);
        for (int i = 1; i < exponent; ++i) {
            result = sym_mul(std::move(result), clone_expr(*expr.left));
        }
        return sym_simplify(sym_expand(std::move(result)));
    }
    case SymOp::Mul: {
        const SymExpr& left = *expr.left;
        const SymExpr& right = *expr.right;
        if (left.op == SymOp::Add) {
            return sym_simplify(sym_expand(sym_add(
                sym_mul(clone_expr(*left.left), clone_expr(right)),
                sym_mul(clone_expr(*left.right), clone_expr(right)))));
        }
        if (left.op == SymOp::Sub) {
            return sym_simplify(sym_expand(sym_sub(
                sym_mul(clone_expr(*left.left), clone_expr(right)),
                sym_mul(clone_expr(*left.right), clone_expr(right)))));
        }
        if (right.op == SymOp::Add) {
            return sym_simplify(sym_expand(sym_add(
                sym_mul(clone_expr(left), clone_expr(*right.left)),
                sym_mul(clone_expr(left), clone_expr(*right.right)))));
        }
        if (right.op == SymOp::Sub) {
            return sym_simplify(sym_expand(sym_sub(
                sym_mul(clone_expr(left), clone_expr(*right.left)),
                sym_mul(clone_expr(left), clone_expr(*right.right)))));
        }
        break;
    }
    default:
        break;
    }

    return sym_simplify(std::move(expr));
}

namespace {

bool extract_var_power_term(const SymExpr& term, const std::string& var, double& coef, double& power) {
    switch (term.op) {
    case SymOp::Const:
        coef = term.value;
        power = 0.0;
        return true;
    case SymOp::Var:
        if (term.name != var) {
            return false;
        }
        coef = 1.0;
        power = 1.0;
        return true;
    case SymOp::Pow:
        if (!term.left || !term.right) {
            return false;
        }
        if (term.left->op == SymOp::Var && term.left->name == var && term.right->op == SymOp::Const) {
            coef = 1.0;
            power = term.right->value;
            return true;
        }
        return false;
    case SymOp::Mul:
        if (!term.left || !term.right) {
            return false;
        }
        if (term.left->op == SymOp::Const) {
            double inner_coef = 0.0;
            double inner_power = 0.0;
            if (extract_var_power_term(*term.right, var, inner_coef, inner_power)) {
                coef = term.left->value * inner_coef;
                power = inner_power;
                return true;
            }
        }
        if (term.right->op == SymOp::Const) {
            double inner_coef = 0.0;
            double inner_power = 0.0;
            if (extract_var_power_term(*term.left, var, inner_coef, inner_power)) {
                coef = term.right->value * inner_coef;
                power = inner_power;
                return true;
            }
        }
        return false;
    case SymOp::Neg:
        if (!term.left) {
            return false;
        }
        if (extract_var_power_term(*term.left, var, coef, power)) {
            coef = -coef;
            return true;
        }
        return false;
    default:
        return false;
    }
}

void flatten_sum_terms(
    const SymExpr& expr,
    const std::string& var,
    bool negate,
    std::map<double, double>& by_power,
    std::vector<SymExpr>& other_terms) {
    switch (expr.op) {
    case SymOp::Add:
        if (expr.left) {
            flatten_sum_terms(*expr.left, var, negate, by_power, other_terms);
        }
        if (expr.right) {
            flatten_sum_terms(*expr.right, var, negate, by_power, other_terms);
        }
        return;
    case SymOp::Sub:
        if (expr.left) {
            flatten_sum_terms(*expr.left, var, negate, by_power, other_terms);
        }
        if (expr.right) {
            flatten_sum_terms(*expr.right, var, !negate, by_power, other_terms);
        }
        return;
    case SymOp::Neg:
        if (expr.left) {
            flatten_sum_terms(*expr.left, var, !negate, by_power, other_terms);
        }
        return;
    default: {
        SymExpr leaf = sym_simplify(clone_expr(expr));
        double coef = 0.0;
        double power = 0.0;
        if (extract_var_power_term(leaf, var, coef, power)) {
            by_power[power] += negate ? -coef : coef;
        } else if (negate) {
            other_terms.push_back(sym_simplify(sym_neg(std::move(leaf))));
        } else {
            other_terms.push_back(std::move(leaf));
        }
        return;
    }
    }
}

SymExpr build_var_power_term(double coef, double power, const std::string& var) {
    if (coef == 0.0) {
        return sym_const(0.0);
    }
    if (power == 0.0) {
        return sym_const(coef);
    }

    SymExpr base = (power == 1.0) ? sym_var(var) : sym_pow(sym_var(var), sym_const(power));
    if (coef == 1.0) {
        return sym_simplify(std::move(base));
    }
    if (coef == -1.0) {
        return sym_simplify(sym_neg(std::move(base)));
    }
    return sym_simplify(sym_mul(sym_const(coef), std::move(base)));
}

SymExpr rebuild_collected_sum(
    std::map<double, double> by_power, std::vector<SymExpr> other_terms, const std::string& var) {
    std::vector<SymExpr> parts;
    parts.reserve(by_power.size() + other_terms.size());

    for (const auto& [power, coef] : by_power) {
        if (coef == 0.0) {
            continue;
        }
        parts.push_back(build_var_power_term(coef, power, var));
    }
    for (auto& term : other_terms) {
        if (term.op == SymOp::Const && term.value == 0.0) {
            continue;
        }
        parts.push_back(std::move(term));
    }

    if (parts.empty()) {
        return sym_const(0.0);
    }

    SymExpr result = std::move(parts.front());
    for (std::size_t i = 1; i < parts.size(); ++i) {
        result = sym_add(std::move(result), std::move(parts[i]));
    }
    return sym_simplify(std::move(result));
}

SymExpr collect_in_sum(const SymExpr& expr, const std::string& var) {
    std::map<double, double> by_power;
    std::vector<SymExpr> other_terms;
    flatten_sum_terms(expr, var, false, by_power, other_terms);
    return rebuild_collected_sum(std::move(by_power), std::move(other_terms), var);
}

} // namespace

SymExpr sym_collect(const SymExpr& expr, const std::string& var) {
    if (var.empty()) {
        return sym_simplify(clone_expr(expr));
    }

    SymExpr e = sym_simplify(clone_expr(expr));

    switch (e.op) {
    case SymOp::Add:
    case SymOp::Sub:
    case SymOp::Neg:
        if (e.left) {
            e.left = std::make_unique<SymExpr>(sym_collect(*e.left, var));
        }
        if (e.right) {
            e.right = std::make_unique<SymExpr>(sym_collect(*e.right, var));
        }
        return collect_in_sum(e, var);
    default:
        break;
    }

    if (e.left) {
        e.left = std::make_unique<SymExpr>(sym_collect(*e.left, var));
    }
    if (e.right) {
        e.right = std::make_unique<SymExpr>(sym_collect(*e.right, var));
    }
    return sym_simplify(std::move(e));
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

SymExpr sym_dsolve_unsupported(const SymExpr& rhs, const std::string& indep_var) {
    return sym_deriv(clone_expr(rhs), indep_var);
}

bool sym_is_deriv_sentinel(const SymExpr& result, const std::string& var) {
    return result.op == SymOp::Deriv && result.name == var;
}

std::optional<SymExpr> sym_extract_y_multiplier(const SymExpr& expr, const std::string& dep_var) {
    if (is_bare_var(expr, dep_var)) {
        return sym_const(1.0);
    }
    if (expr.op == SymOp::Neg && expr.left && is_bare_var(*expr.left, dep_var)) {
        return sym_const(-1.0);
    }
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (is_bare_var(*expr.right, dep_var) && !contains_var_name(*expr.left, dep_var)) {
            return clone_expr(*expr.left);
        }
        if (is_bare_var(*expr.left, dep_var) && !contains_var_name(*expr.right, dep_var)) {
            return clone_expr(*expr.right);
        }
    }
    return std::nullopt;
}

struct SymAffineInDepVar {
    SymExpr coef;
    SymExpr intercept;
};

std::optional<SymAffineInDepVar> sym_extract_affine_in_dep_var(
    const SymExpr& expr, const std::string& dep_var) {
    if (is_bare_var(expr, dep_var)) {
        return SymAffineInDepVar{sym_const(1.0), sym_const(0.0)};
    }
    if (expr.op == SymOp::Neg && expr.left && is_bare_var(*expr.left, dep_var)) {
        return SymAffineInDepVar{sym_const(-1.0), sym_const(0.0)};
    }
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (is_bare_var(*expr.right, dep_var) && !contains_var_name(*expr.left, dep_var)) {
            return SymAffineInDepVar{clone_expr(*expr.left), sym_const(0.0)};
        }
        if (is_bare_var(*expr.left, dep_var) && !contains_var_name(*expr.right, dep_var)) {
            return SymAffineInDepVar{clone_expr(*expr.right), sym_const(0.0)};
        }
    }
    if ((expr.op == SymOp::Add || expr.op == SymOp::Sub) && expr.left && expr.right) {
        const auto left = sym_extract_affine_in_dep_var(*expr.left, dep_var);
        const auto right = sym_extract_affine_in_dep_var(*expr.right, dep_var);
        if (!left && contains_var_name(*expr.left, dep_var)) {
            return std::nullopt;
        }
        if (!right && contains_var_name(*expr.right, dep_var)) {
            return std::nullopt;
        }
        const SymExpr ca = left ? clone_expr(left->coef) : sym_const(0.0);
        const SymExpr cb = left ? clone_expr(left->intercept) : clone_expr(*expr.left);
        const SymExpr da = right ? clone_expr(right->coef) : sym_const(0.0);
        const SymExpr db = right ? clone_expr(right->intercept) : clone_expr(*expr.right);
        if (expr.op == SymOp::Add) {
            return SymAffineInDepVar{
                sym_add(clone_expr(ca), clone_expr(da)),
                sym_add(clone_expr(cb), clone_expr(db)),
            };
        }
        return SymAffineInDepVar{
            sym_sub(clone_expr(ca), clone_expr(da)),
            sym_sub(clone_expr(cb), clone_expr(db)),
        };
    }
    if (!contains_var_name(expr, dep_var)) {
        return SymAffineInDepVar{sym_const(0.0), clone_expr(expr)};
    }
    return std::nullopt;
}

SymExpr sym_add_integration_constant(SymExpr expr) {
    return sym_add(std::move(expr), sym_var("C"));
}

// Supported separable first-order ODE rules (table-driven MVP):
//   f(x)                     -> integrate(f, x) + C
//   y^n (n != 1)             -> ((1-n)*(x+C))^(1/(1-n))
//   a*y + b (const a,b)      -> -b/a + C*exp(a*x)  (a != 0); b*x + C (a == 0)
//   k(x)*y                   -> C*exp(integrate(k, x))
// Unsupported rhs forms return sym_deriv(rhs, indep_var) as an explicit sentinel.
SymExpr sym_dsolve(const SymExpr& rhs, const std::string& indep_var, const std::string& dep_var) {
    if (!contains_var_name(rhs, dep_var)) {
        SymExpr integrated = sym_integrate(rhs, indep_var);
        if (sym_is_deriv_sentinel(integrated, indep_var)) {
            return sym_dsolve_unsupported(rhs, indep_var);
        }
        return sym_add_integration_constant(std::move(integrated));
    }

    if (rhs.op == SymOp::Pow && rhs.left && rhs.right && is_bare_var(*rhs.left, dep_var) &&
        rhs.right->op == SymOp::Const) {
        const double n = rhs.right->value;
        if (n != 1.0) {
            const double one_minus_n = 1.0 - n;
            if (one_minus_n != 0.0) {
                SymExpr x_plus_c = sym_add(sym_var(indep_var), sym_var("C"));
                SymExpr inner = sym_mul(sym_const(one_minus_n), std::move(x_plus_c));
                return sym_pow(std::move(inner), sym_const(1.0 / one_minus_n));
            }
        }
    }

    if (const auto affine = sym_extract_affine_in_dep_var(rhs, dep_var)) {
        const SymExpr a_expr = sym_simplify(clone_expr(affine->coef));
        const SymExpr b_expr = sym_simplify(clone_expr(affine->intercept));
        double a = 0.0;
        double b = 0.0;
        if (try_get_const_value(a_expr, a) && try_get_const_value(b_expr, b) &&
            !contains_var_name(a_expr, indep_var) && !contains_var_name(b_expr, indep_var)) {
            if (a == 0.0) {
                return sym_add(sym_mul(sym_const(b), sym_var(indep_var)), sym_var("C"));
            }
            return sym_add(
                sym_neg(sym_div(sym_const(b), sym_const(a))),
                sym_mul(sym_var("C"), sym_exp(sym_mul(sym_const(a), sym_var(indep_var)))));
        }
    }

    if (auto multiplier = sym_extract_y_multiplier(rhs, dep_var)) {
        if (!contains_var_name(*multiplier, dep_var)) {
            SymExpr factor = std::move(*multiplier);
            SymExpr integrated = sym_integrate(factor, indep_var);
            if (sym_is_deriv_sentinel(integrated, indep_var)) {
                return sym_dsolve_unsupported(rhs, indep_var);
            }
            return sym_mul(sym_var("C"), sym_exp(std::move(integrated)));
        }
    }

    return sym_dsolve_unsupported(rhs, indep_var);
}

// Supported forward Laplace rules (table-driven MVP):
//   Const c              -> c / s
//   exp(a*t)             -> 1 / (s - a)
//   sin(a*t)             -> a / (s^2 + a^2)
//   cos(a*t)             -> s / (s^2 + a^2)
//   t^n (small int n)    -> n! / s^(n + 1)
// Linearity: Add, Sub, Neg, Mul with Const factor.
// Unsupported forms return sym_deriv(expr, t) as an explicit sentinel.
SymExpr sym_laplace(const SymExpr& expr, const std::string& t, const std::string& s) {
    switch (expr.op) {
    case SymOp::Const:
        return sym_div(clone_expr(expr), sym_var(s));
    case SymOp::Var:
        if (expr.name == t) {
            return sym_div(sym_const(1.0), sym_pow(sym_var(s), sym_const(2.0)));
        }
        return sym_laplace_unsupported(expr, t);
    case SymOp::Add:
        return sym_add(
            sym_laplace(*expr.left, t, s),
            sym_laplace(*expr.right, t, s));
    case SymOp::Sub:
        return sym_sub(
            sym_laplace(*expr.left, t, s),
            sym_laplace(*expr.right, t, s));
    case SymOp::Neg:
        return sym_neg(sym_laplace(*expr.left, t, s));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_laplace(*expr.right, t, s));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_laplace(*expr.left, t, s));
        }
        return sym_laplace_unsupported(expr, t);
    case SymOp::Pow:
        if (is_bare_var(*expr.left, t) && expr.right->op == SymOp::Const) {
            int n = 0;
            if (is_small_nonneg_int(expr.right->value, n)) {
                return sym_div(
                    sym_const(factorial_int(n)),
                    sym_pow(sym_var(s), sym_const(static_cast<double>(n + 1))));
            }
        }
        return sym_laplace_unsupported(expr, t);
    case SymOp::Exp: {
        double a = 0.0;
        if (match_scaled_var(*expr.left, t, a)) {
            return sym_div(sym_const(1.0), sym_sub(sym_var(s), sym_const(a)));
        }
        return sym_laplace_unsupported(expr, t);
    }
    case SymOp::Sin: {
        double a = 0.0;
        if (match_scaled_var(*expr.left, t, a)) {
            return sym_div(sym_const(a), build_s2_plus_a2(s, a));
        }
        return sym_laplace_unsupported(expr, t);
    }
    case SymOp::Cos: {
        double a = 0.0;
        if (match_scaled_var(*expr.left, t, a)) {
            return sym_div(sym_var(s), build_s2_plus_a2(s, a));
        }
        return sym_laplace_unsupported(expr, t);
    }
    default:
        return sym_laplace_unsupported(expr, t);
    }
}

// Supported inverse Laplace rules (table-driven MVP):
//   1 / (s - a)          -> exp(a*t)
//   1 / s^2              -> t
//   s / (s^2 + a^2)      -> cos(a*t)
//   a / (s^2 + a^2)      -> sin(a*t)
//   n! / s^(n + 1)       -> t^n   (paired with forward t^n rule)
// Linearity: Add, Sub, Neg, Mul with Const factor.
// Unsupported forms return sym_deriv(expr, s) as an explicit sentinel.
SymExpr sym_ilaplace(const SymExpr& expr, const std::string& s, const std::string& t) {
    switch (expr.op) {
    case SymOp::Const:
        if (expr.value == 0.0) {
            return sym_const(0.0);
        }
        return sym_ilaplace_unsupported(expr, s);
    case SymOp::Add:
        return sym_add(
            sym_ilaplace(*expr.left, s, t),
            sym_ilaplace(*expr.right, s, t));
    case SymOp::Sub:
        return sym_sub(
            sym_ilaplace(*expr.left, s, t),
            sym_ilaplace(*expr.right, s, t));
    case SymOp::Neg:
        return sym_neg(sym_ilaplace(*expr.left, s, t));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_ilaplace(*expr.right, s, t));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_ilaplace(*expr.left, s, t));
        }
        return sym_ilaplace_unsupported(expr, s);
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_ilaplace_unsupported(expr, s);
        }
        double numerator = 0.0;
        if (try_get_const_value(*expr.left, numerator)) {
            double a = 0.0;
            if (match_sub_var_minus_const(*expr.right, s, a)) {
                return sym_mul(
                    sym_const(numerator),
                    sym_exp(sym_mul(sym_const(a), sym_var(t))));
            }
            if (is_var_pow(*expr.right, s, 2.0) && numerator == 1.0) {
                return sym_var(t);
            }
            if (expr.right->op == SymOp::Pow && is_bare_var(*expr.right->left, s) &&
                expr.right->right->op == SymOp::Const) {
                const double power = expr.right->right->value;
                if (power >= 2.0 && power == std::floor(power)) {
                    const int n = static_cast<int>(power) - 1;
                    if (n >= 0 && n <= kMaxLaplacePower && numerator == factorial_int(n)) {
                        if (n == 0) {
                            return sym_const(1.0);
                        }
                        if (n == 1) {
                            return sym_var(t);
                        }
                        return sym_pow(sym_var(t), sym_const(static_cast<double>(n)));
                    }
                }
            }
            double denom_a = 0.0;
            if (match_s2_plus_a2(*expr.right, s, denom_a)) {
                if (numerator == denom_a) {
                    return sym_sin(sym_mul(sym_const(denom_a), sym_var(t)));
                }
                return sym_ilaplace_unsupported(expr, s);
            }
        }
        if (is_bare_var(*expr.left, s)) {
            double a = 0.0;
            if (match_s2_plus_a2(*expr.right, s, a)) {
                return sym_cos(sym_mul(sym_const(a), sym_var(t)));
            }
        }
        return sym_ilaplace_unsupported(expr, s);
    }
    default:
        return sym_ilaplace_unsupported(expr, s);
    }
}

// Supported Mellin rules (table-driven MVP, M{f}(s) = integral_0^inf t^{s-1} f(t) dt):
//   c                    -> c / s
//   t^a                  -> 1 / (s + a)
//   exp(-a*t)            -> n! / a^{n+s} with n = 0
//   t^n * exp(-a*t)      -> n! / a^{n+s}  (small int n)
//   1 / (1 + t)          -> pi / sin(pi * s)
// Linearity: Add, Sub, Neg, Mul with Const factor.
// Unsupported forms return sym_deriv(expr, t) as an explicit sentinel.
SymExpr sym_mellin(const SymExpr& expr, const std::string& t, const std::string& s) {
    switch (expr.op) {
    case SymOp::Const:
        return sym_div(clone_expr(expr), sym_var(s));
    case SymOp::Var:
        if (expr.name == t) {
            return sym_div(sym_const(1.0), sym_add(sym_var(s), sym_const(1.0)));
        }
        return sym_mellin_unsupported(expr, t);
    case SymOp::Add:
        return sym_add(
            sym_mellin(*expr.left, t, s),
            sym_mellin(*expr.right, t, s));
    case SymOp::Sub:
        return sym_sub(
            sym_mellin(*expr.left, t, s),
            sym_mellin(*expr.right, t, s));
    case SymOp::Neg:
        return sym_neg(sym_mellin(*expr.left, t, s));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_mellin(*expr.right, t, s));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_mellin(*expr.left, t, s));
        }
        if (const auto matched = match_tpow_exp_neg(expr, t)) {
            const int n = matched->first;
            const double a = matched->second;
            return sym_div(
                sym_const(factorial_int(n)),
                sym_pow(sym_const(a), sym_add(sym_var(s), sym_const(static_cast<double>(n)))));
        }
        return sym_mellin_unsupported(expr, t);
    case SymOp::Pow:
        if (is_bare_var(*expr.left, t) && expr.right->op == SymOp::Const) {
            return sym_div(
                sym_const(1.0),
                sym_add(sym_var(s), clone_expr(*expr.right)));
        }
        return sym_mellin_unsupported(expr, t);
    case SymOp::Exp: {
        double a = 0.0;
        if (match_exp_neg_at(expr, t, a)) {
            return sym_div(sym_const(1.0), sym_pow(sym_const(a), sym_var(s)));
        }
        return sym_mellin_unsupported(expr, t);
    }
    case SymOp::Div:
        if (match_one_over_one_plus_t(expr, t)) {
            return sym_div(
                sym_const(std::numbers::pi),
                sym_sin(sym_mul(sym_const(std::numbers::pi), sym_var(s))));
        }
        return sym_mellin_unsupported(expr, t);
    default:
        return sym_mellin_unsupported(expr, t);
    }
}

// Supported inverse Mellin rules (paired with forward table):
//   c / s                  -> c
//   1 / (s + a)            -> t^a
//   n! / a^{n+s}           -> t^n * exp(-a*t)
//   pi / sin(pi * s)       -> 1 / (1 + t)
// Linearity: Add, Sub, Neg, Mul with Const factor.
// Unsupported forms return sym_deriv(expr, s) as an explicit sentinel.
SymExpr sym_imellin(const SymExpr& expr, const std::string& s, const std::string& t) {
    switch (expr.op) {
    case SymOp::Const:
        if (expr.value == 0.0) {
            return sym_const(0.0);
        }
        return sym_imellin_unsupported(expr, s);
    case SymOp::Add:
        return sym_add(
            sym_imellin(*expr.left, s, t),
            sym_imellin(*expr.right, s, t));
    case SymOp::Sub:
        return sym_sub(
            sym_imellin(*expr.left, s, t),
            sym_imellin(*expr.right, s, t));
    case SymOp::Neg:
        return sym_neg(sym_imellin(*expr.left, s, t));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_imellin(*expr.right, s, t));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_imellin(*expr.left, s, t));
        }
        return sym_imellin_unsupported(expr, s);
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_imellin_unsupported(expr, s);
        }
        double numerator = 0.0;
        if (!try_get_const_value(*expr.left, numerator)) {
            return sym_imellin_unsupported(expr, s);
        }
        if (is_bare_var(*expr.right, s)) {
            return sym_const(numerator);
        }
        if (expr.right->op == SymOp::Sin && expr.right->left &&
            numerator == std::numbers::pi) {
            const SymExpr& sin_arg = *expr.right->left;
            if (sin_arg.op == SymOp::Mul && sin_arg.left && sin_arg.right &&
                sin_arg.left->op == SymOp::Const && sin_arg.left->value == std::numbers::pi &&
                is_bare_var(*sin_arg.right, s)) {
                return sym_div(sym_const(1.0), sym_add(sym_const(1.0), sym_var(t)));
            }
        }
        double a = 0.0;
        if (match_add_var_plus_const(*expr.right, s, a) && numerator == 1.0) {
            return sym_pow(sym_var(t), sym_const(a));
        }
        if (expr.right->op == SymOp::Pow && expr.right->left && expr.right->right &&
            expr.right->left->op == SymOp::Const) {
            const double base = expr.right->left->value;
            if (base > 0.0 && is_bare_var(*expr.right->right, s) && numerator == 1.0) {
                return sym_exp(sym_neg(sym_mul(sym_const(base), sym_var(t))));
            }
            if (expr.right->right->op == SymOp::Add && expr.right->right->left &&
                is_bare_var(*expr.right->right->left, s) && expr.right->right->right &&
                expr.right->right->right->op == SymOp::Const) {
                const int n = static_cast<int>(expr.right->right->right->value);
                if (n >= 0 && n <= kMaxMellinPower && numerator == factorial_int(n)) {
                    SymExpr decay = sym_exp(sym_neg(sym_mul(sym_const(base), sym_var(t))));
                    if (n == 0) {
                        return decay;
                    }
                    return sym_mul(
                        sym_pow(sym_var(t), sym_const(static_cast<double>(n))), std::move(decay));
                }
            }
        }
        return sym_imellin_unsupported(expr, s);
    }
    default:
        return sym_imellin_unsupported(expr, s);
    }
}

// Supported Hankel rules (order 0, r-domain -> k-domain):
//   exp(-a*r)            -> a / (a^2 + k^2)^(3/2)
//   r^n * exp(-a*r)      -> 2^{n+1} Gamma((n+3)/2) a / (sqrt(pi) (a^2 + k^2)^{(n+3)/2})
//   1 / sqrt(r^2 + a^2)  -> exp(-a*k) / k
// Linearity: Add, Sub, Neg, Mul with Const factor.
// Unsupported forms return sym_deriv(expr, r) as an explicit sentinel.
SymExpr sym_hankel(const SymExpr& expr, const std::string& r, const std::string& k) {
    switch (expr.op) {
    case SymOp::Const:
        return sym_hankel_unsupported(expr, r);
    case SymOp::Var:
        return sym_hankel_unsupported(expr, r);
    case SymOp::Add:
        return sym_add(
            sym_hankel(*expr.left, r, k),
            sym_hankel(*expr.right, r, k));
    case SymOp::Sub:
        return sym_sub(
            sym_hankel(*expr.left, r, k),
            sym_hankel(*expr.right, r, k));
    case SymOp::Neg:
        return sym_neg(sym_hankel(*expr.left, r, k));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_hankel(*expr.right, r, k));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_hankel(*expr.left, r, k));
        }
        if (const auto matched = match_tpow_exp_neg(expr, r)) {
            return hankel_forward_rpow_exp_neg(matched->first, matched->second, k);
        }
        return sym_hankel_unsupported(expr, r);
    case SymOp::Exp: {
        double a = 0.0;
        if (match_exp_neg_at(expr, r, a)) {
            return hankel_forward_rpow_exp_neg(0, a, k);
        }
        return sym_hankel_unsupported(expr, r);
    }
    case SymOp::Div: {
        double a = 0.0;
        if (match_one_over_sqrt_r2_plus_a2(expr, r, a)) {
            return sym_div(
                sym_exp(sym_neg(sym_mul(sym_const(a), sym_var(k)))),
                sym_var(k));
        }
        return sym_hankel_unsupported(expr, r);
    }
    default:
        return sym_hankel_unsupported(expr, r);
    }
}

// Supported inverse Hankel rules (paired with forward table):
//   a / (a^2 + k^2)^{(n+3)/2} scaled form -> r^n * exp(-a*r)
//   exp(-a*k) / k                        -> 1 / sqrt(r^2 + a^2)
// Linearity: Add, Sub, Neg, Mul with Const factor.
// Unsupported forms return sym_deriv(expr, k) as an explicit sentinel.
SymExpr sym_ihankel(const SymExpr& expr, const std::string& k, const std::string& r) {
    switch (expr.op) {
    case SymOp::Const:
        if (expr.value == 0.0) {
            return sym_const(0.0);
        }
        return sym_ihankel_unsupported(expr, k);
    case SymOp::Add:
        return sym_add(
            sym_ihankel(*expr.left, k, r),
            sym_ihankel(*expr.right, k, r));
    case SymOp::Sub:
        return sym_sub(
            sym_ihankel(*expr.left, k, r),
            sym_ihankel(*expr.right, k, r));
    case SymOp::Neg:
        return sym_neg(sym_ihankel(*expr.left, k, r));
    case SymOp::Mul:
        if (expr.left->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.left), sym_ihankel(*expr.right, k, r));
        }
        if (expr.right->op == SymOp::Const) {
            return sym_mul(clone_expr(*expr.right), sym_ihankel(*expr.left, k, r));
        }
        return sym_ihankel_unsupported(expr, k);
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_ihankel_unsupported(expr, k);
        }
        double a = 0.0;
        if (match_exp_neg_ak_over_k(expr, k, a)) {
            return sym_div(
                sym_const(1.0),
                sym_sqrt(build_k2_plus_a2(r, a)));
        }
        if (const auto matched = match_hankel_k_domain_rpow_exp(expr, k)) {
            return ihankel_inverse_rpow_exp_neg(matched->first, matched->second, r);
        }
        return sym_ihankel_unsupported(expr, k);
    }
    default:
        return sym_ihankel_unsupported(expr, k);
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
    case SymOp::Add: {
        const std::string left = sym_to_string(*expr.left);
        const std::string right = sym_to_string(*expr.right);
        std::string out;
        out.reserve(left.size() + right.size() + 7);
        out.push_back('(');
        out += left;
        out += " + ";
        out += right;
        out.push_back(')');
        return out;
    }
    case SymOp::Sub: {
        const std::string left = sym_to_string(*expr.left);
        const std::string right = sym_to_string(*expr.right);
        std::string out;
        out.reserve(left.size() + right.size() + 7);
        out.push_back('(');
        out += left;
        out += " - ";
        out += right;
        out.push_back(')');
        return out;
    }
    case SymOp::Mul: {
        const std::string left = sym_to_string(*expr.left);
        const std::string right = sym_to_string(*expr.right);
        std::string out;
        out.reserve(left.size() + right.size() + 7);
        out.push_back('(');
        out += left;
        out += " * ";
        out += right;
        out.push_back(')');
        return out;
    }
    case SymOp::Div: {
        const std::string left = sym_to_string(*expr.left);
        const std::string right = sym_to_string(*expr.right);
        std::string out;
        out.reserve(left.size() + right.size() + 7);
        out.push_back('(');
        out += left;
        out += " / ";
        out += right;
        out.push_back(')');
        return out;
    }
    case SymOp::Neg: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 3);
        out += "(-";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Sin: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 5);
        out += "sin(";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Cos: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 5);
        out += "cos(";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Tan: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 5);
        out += "tan(";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Exp: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 5);
        out += "exp(";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Log: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 5);
        out += "log(";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Sqrt: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(inner.size() + 7);
        out += "sqrt(";
        out += inner;
        out.push_back(')');
        return out;
    }
    case SymOp::Pow: {
        const std::string left = sym_to_string(*expr.left);
        const std::string right = sym_to_string(*expr.right);
        std::string out;
        out.reserve(left.size() + right.size() + 7);
        out.push_back('(');
        out += left;
        out += " ^ ";
        out += right;
        out.push_back(')');
        return out;
    }
    case SymOp::Deriv: {
        const std::string inner = sym_to_string(*expr.left);
        std::string out;
        out.reserve(expr.name.size() + inner.size() + 6);
        out += "d/d";
        out += expr.name;
        out.push_back('(');
        out += inner;
        out.push_back(')');
        return out;
    }
    }
    return "?";
}

double sym_limit(const SymExpr& expr, const std::string& var, double point) {
    const auto eval_at = [&](double x) {
        return sym_eval(expr, {{var, x}});
    };

    const double direct = eval_at(point);
    if (std::isfinite(direct)) {
        const double probe = 1e-8;
        const double left = eval_at(point - probe);
        const double right = eval_at(point + probe);
        if (std::isfinite(left) && std::isfinite(right) &&
            std::abs(left - direct) < 1e-6 && std::abs(right - direct) < 1e-6) {
            return direct;
        }
    }

    double h = 1e-4;
    double estimate = 0.0;
    double prev = 0.0;
    for (int step = 0; step < 12; ++step) {
        const double left = eval_at(point - h);
        const double right = eval_at(point + h);
        if (std::isfinite(left) && std::isfinite(right)) {
            estimate = 0.5 * (left + right);
            if (step > 0) {
                const double scale = std::max(1.0, std::abs(estimate));
                if (std::abs(estimate - prev) < 1e-10 * scale) {
                    return estimate;
                }
            }
            prev = estimate;
        }
        h *= 0.1;
    }
    return estimate;
}

SymExpr sym_series(const SymExpr& expr, const std::string& var, double point, int order) {
    if (order <= 0) {
        return sym_const(0.0);
    }

    SymExpr result = sym_const(0.0);
    const SymExpr x_shift = sym_sub(sym_var(var), sym_const(point));
    SymExpr deriv = clone_expr(expr);
    double factorial = 1.0;

    for (int n = 0; n < order; ++n) {
        const double coeff = sym_eval(sym_simplify(clone_expr(deriv)), {{var, point}}) / factorial;
        if (coeff != 0.0) {
            SymExpr term = sym_const(coeff);
            if (n > 0) {
                term = sym_mul(
                    sym_const(coeff),
                    sym_pow(clone_expr(x_shift), sym_const(static_cast<double>(n))));
            }
            result = sym_add(std::move(result), std::move(term));
        }
        if (n + 1 < order) {
            deriv = sym_diff(std::move(deriv), var);
            factorial *= static_cast<double>(n + 1);
        }
    }

    return sym_simplify(std::move(result));
}

namespace {

SymExpr scale_expr(SymExpr expr, double sign) {
    if (sign == 1.0) {
        return expr;
    }
    if (sign == -1.0) {
        return sym_neg(std::move(expr));
    }
    return sym_mul(sym_const(sign), std::move(expr));
}

bool contains_var_name(const SymExpr& expr, const std::string& name) {
    if (expr.op == SymOp::Var) {
        return expr.name == name;
    }
    if (expr.left && contains_var_name(*expr.left, name)) {
        return true;
    }
    if (expr.right && contains_var_name(*expr.right, name)) {
        return true;
    }
    return false;
}

bool is_allowed_other_var(const SymExpr& expr, const std::vector<std::string>& vars) {
    if (expr.op == SymOp::Var) {
        return std::find(vars.begin(), vars.end(), expr.name) != vars.end();
    }
    if (expr.left && !is_allowed_other_var(*expr.left, vars)) {
        return false;
    }
    if (expr.right && !is_allowed_other_var(*expr.right, vars)) {
        return false;
    }
    return true;
}

void gather_mul_factors(const SymExpr& expr, std::vector<SymExpr>& factors) {
    if (expr.op == SymOp::Mul) {
        if (expr.left) {
            gather_mul_factors(*expr.left, factors);
        }
        if (expr.right) {
            gather_mul_factors(*expr.right, factors);
        }
        return;
    }
    factors.push_back(clone_expr(expr));
}

std::optional<SymExpr> extract_linear_term(
    const SymExpr& term, const std::vector<std::string>& vars, std::string& matched_var) {
    SymExpr leaf = sym_simplify(clone_expr(term));
    matched_var.clear();

    if (leaf.op == SymOp::Const) {
        return leaf;
    }
    if (leaf.op == SymOp::Var) {
        if (std::find(vars.begin(), vars.end(), leaf.name) != vars.end()) {
            matched_var = leaf.name;
            return sym_const(1.0);
        }
        return clone_expr(leaf);
    }
    if (leaf.op == SymOp::Pow) {
        if (leaf.left && leaf.left->op == SymOp::Var && leaf.right && leaf.right->op == SymOp::Const) {
            if (leaf.right->value == 1.0 &&
                std::find(vars.begin(), vars.end(), leaf.left->name) != vars.end()) {
                matched_var = leaf.left->name;
                return sym_const(1.0);
            }
            if (std::find(vars.begin(), vars.end(), leaf.left->name) == vars.end()) {
                return clone_expr(leaf);
            }
        }
        return std::nullopt;
    }
    if (leaf.op == SymOp::Mul) {
        SymExpr coef = sym_const(1.0);
        bool have_var = false;
        std::vector<SymExpr> factors;
        gather_mul_factors(leaf, factors);

        for (SymExpr& factor : factors) {
            factor = sym_simplify(std::move(factor));
            if (factor.op == SymOp::Const) {
                coef = sym_mul(std::move(coef), std::move(factor));
                continue;
            }
            if (factor.op == SymOp::Var) {
                if (std::find(vars.begin(), vars.end(), factor.name) != vars.end()) {
                    if (have_var) {
                        return std::nullopt;
                    }
                    have_var = true;
                    matched_var = factor.name;
                    continue;
                }
                coef = sym_mul(std::move(coef), std::move(factor));
                continue;
            }
            if (factor.op == SymOp::Pow && factor.left && factor.left->op == SymOp::Var && factor.right &&
                factor.right->op == SymOp::Const && factor.right->value == 1.0) {
                if (std::find(vars.begin(), vars.end(), factor.left->name) != vars.end()) {
                    if (have_var) {
                        return std::nullopt;
                    }
                    have_var = true;
                    matched_var = factor.left->name;
                    continue;
                }
                coef = sym_mul(std::move(coef), std::move(factor));
                continue;
            }
            bool uses_any_solve_var = false;
            for (const auto& solve_var : vars) {
                if (contains_var_name(factor, solve_var)) {
                    uses_any_solve_var = true;
                    break;
                }
            }
            if (!uses_any_solve_var) {
                coef = sym_mul(std::move(coef), std::move(factor));
                continue;
            }
            return std::nullopt;
        }

        if (!have_var) {
            return sym_simplify(std::move(coef));
        }
        return sym_simplify(std::move(coef));
    }
    if (leaf.op == SymOp::Neg && leaf.left) {
        auto inner = extract_linear_term(*leaf.left, vars, matched_var);
        if (!inner) {
            return std::nullopt;
        }
        return sym_neg(std::move(*inner));
    }

    if (!is_allowed_other_var(leaf, vars)) {
        return std::nullopt;
    }
    return std::nullopt;
}

void flatten_linear_sum(
    const SymExpr& expr,
    double sign,
    const std::vector<std::string>& vars,
    std::map<std::string, SymExpr>& var_coeffs,
    SymExpr& constant) {
    switch (expr.op) {
    case SymOp::Add:
        if (expr.left) {
            flatten_linear_sum(*expr.left, sign, vars, var_coeffs, constant);
        }
        if (expr.right) {
            flatten_linear_sum(*expr.right, sign, vars, var_coeffs, constant);
        }
        return;
    case SymOp::Sub:
        if (expr.left) {
            flatten_linear_sum(*expr.left, sign, vars, var_coeffs, constant);
        }
        if (expr.right) {
            flatten_linear_sum(*expr.right, -sign, vars, var_coeffs, constant);
        }
        return;
    case SymOp::Neg:
        if (expr.left) {
            flatten_linear_sum(*expr.left, -sign, vars, var_coeffs, constant);
        }
        return;
    default: {
        std::string matched_var;
        auto parsed = extract_linear_term(expr, vars, matched_var);
        if (!parsed) {
            return;
        }
        SymExpr scaled = scale_expr(std::move(*parsed), sign);
        if (matched_var.empty()) {
            constant = sym_simplify(sym_add(std::move(constant), std::move(scaled)));
        } else {
            var_coeffs[matched_var] =
                sym_simplify(sym_add(std::move(var_coeffs[matched_var]), std::move(scaled)));
        }
        return;
    }
    }
}

struct LinearRow {
    std::map<std::string, SymExpr> var_coeffs;
    SymExpr constant = sym_const(0.0);
};

std::optional<LinearRow> extract_linear_row(const SymExpr& equation, const std::vector<std::string>& vars) {
    LinearRow row;
    for (const auto& v : vars) {
        row.var_coeffs[v] = sym_const(0.0);
    }
    flatten_linear_sum(sym_simplify(clone_expr(equation)), 1.0, vars, row.var_coeffs, row.constant);
    return row;
}

bool try_eval_const(const SymExpr& expr, double& out) {
    const SymExpr simplified = sym_simplify(clone_expr(expr));
    if (simplified.op != SymOp::Const) {
        return false;
    }
    out = simplified.value;
    return true;
}

std::expected<std::map<std::string, SymExpr>, SymSolveError> solve_numeric_system(
    std::vector<std::vector<double>> a, std::vector<double> b, const std::vector<std::string>& vars) {
    const int n = static_cast<int>(vars.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(a[static_cast<size_t>(row)][static_cast<size_t>(col)]) >
                std::abs(a[static_cast<size_t>(pivot)][static_cast<size_t>(col)])) {
                pivot = row;
            }
        }
        std::swap(a[static_cast<size_t>(col)], a[static_cast<size_t>(pivot)]);
        std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot)]);
        const double diag = a[static_cast<size_t>(col)][static_cast<size_t>(col)];
        if (std::abs(diag) < 1e-14) {
            return std::unexpected(SymSolveError{"singular linear system"});
        }
        for (int row = col + 1; row < n; ++row) {
            const double factor = a[static_cast<size_t>(row)][static_cast<size_t>(col)] / diag;
            for (int k = col; k < n; ++k) {
                a[static_cast<size_t>(row)][static_cast<size_t>(k)] -=
                    factor * a[static_cast<size_t>(col)][static_cast<size_t>(k)];
            }
            b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
        }
    }

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= a[static_cast<size_t>(i)][static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
        }
        const double diag = a[static_cast<size_t>(i)][static_cast<size_t>(i)];
        if (std::abs(diag) < 1e-14) {
            return std::unexpected(SymSolveError{"singular linear system"});
        }
        x[static_cast<size_t>(i)] = sum / diag;
    }

    std::map<std::string, SymExpr> solution;
    for (int i = 0; i < n; ++i) {
        solution[vars[static_cast<size_t>(i)]] = sym_const(x[static_cast<size_t>(i)]);
    }
    return solution;
}

} // namespace

std::expected<std::map<std::string, SymExpr>, SymSolveError> sym_solve_linear(
    const std::vector<SymExpr>& equations, const std::vector<std::string>& vars) {
    if (vars.empty()) {
        return std::unexpected(SymSolveError{"no variables specified"});
    }
    if (equations.size() != vars.size()) {
        return std::unexpected(SymSolveError{"equation count must match variable count"});
    }

    std::vector<LinearRow> rows;
    rows.reserve(equations.size());
    for (const auto& equation : equations) {
        auto row = extract_linear_row(equation, vars);
        if (!row) {
            return std::unexpected(SymSolveError{"failed to extract linear form"});
        }
        rows.push_back(std::move(*row));
    }

    bool all_numeric = true;
    std::vector<std::vector<double>> a(
        equations.size(), std::vector<double>(vars.size(), 0.0));
    std::vector<double> b(equations.size(), 0.0);

    for (std::size_t i = 0; i < equations.size(); ++i) {
        for (std::size_t j = 0; j < vars.size(); ++j) {
            double value = 0.0;
            if (!try_eval_const(rows[i].var_coeffs.at(vars[j]), value)) {
                all_numeric = false;
                break;
            }
            a[i][j] = value;
        }
        if (!all_numeric) {
            break;
        }
        double constant = 0.0;
        if (!try_eval_const(rows[i].constant, constant)) {
            all_numeric = false;
            break;
        }
        b[i] = -constant;
    }

    if (all_numeric) {
        return solve_numeric_system(std::move(a), std::move(b), vars);
    }

    if (vars.size() == 1) {
        SymExpr coeff = std::move(rows.front().var_coeffs.at(vars.front()));
        SymExpr constant = std::move(rows.front().constant);
        std::map<std::string, SymExpr> solution;
        solution.emplace(
            vars.front(), sym_simplify(sym_div(sym_neg(std::move(constant)), std::move(coeff))));
        return solution;
    }

    if (vars.size() == 2) {
        const std::string& x = vars[0];
        const std::string& y = vars[1];
        SymExpr a_coef = std::move(rows[0].var_coeffs.at(x));
        SymExpr b_coef = std::move(rows[0].var_coeffs.at(y));
        SymExpr e_rhs = sym_neg(std::move(rows[0].constant));
        SymExpr c_coef = std::move(rows[1].var_coeffs.at(x));
        SymExpr d_coef = std::move(rows[1].var_coeffs.at(y));
        SymExpr f_rhs = sym_neg(std::move(rows[1].constant));
        const SymExpr det = sym_sub(
            sym_mul(clone_expr(a_coef), clone_expr(d_coef)),
            sym_mul(clone_expr(b_coef), clone_expr(c_coef)));
        std::map<std::string, SymExpr> solution;
        solution.emplace(
            x,
            sym_simplify(sym_div(
                sym_sub(
                    sym_mul(clone_expr(e_rhs), clone_expr(d_coef)),
                    sym_mul(clone_expr(b_coef), clone_expr(f_rhs))),
                clone_expr(det))));
        solution.emplace(
            y,
            sym_simplify(sym_div(
                sym_sub(
                    sym_mul(clone_expr(a_coef), clone_expr(f_rhs)),
                    sym_mul(clone_expr(e_rhs), clone_expr(c_coef))),
                clone_expr(det))));
        return solution;
    }

    return std::unexpected(SymSolveError{"symbolic solve supports 1 or 2 variables"});
}

namespace {

SymExpr sym_transform_unsupported(const SymExpr& expr, const std::string& var) {
    return sym_deriv(clone_expr(expr), var);
}

bool is_named_var(const SymExpr& expr, const std::string& name) {
    return expr.op == SymOp::Var && expr.name == name;
}

std::optional<double> as_const(const SymExpr& expr) {
    if (expr.op == SymOp::Const) {
        return expr.value;
    }
    return std::nullopt;
}

std::optional<double> extract_positive_scale_of_var(const SymExpr& expr, const std::string& var) {
    if (is_named_var(expr, var)) {
        return 1.0;
    }
    if (expr.op == SymOp::Neg && expr.left && is_named_var(*expr.left, var)) {
        return -1.0;
    }
    if (expr.op == SymOp::Mul) {
        if (expr.left && expr.left->op == SymOp::Const && expr.right && is_named_var(*expr.right, var)) {
            return expr.left->value;
        }
        if (expr.right && expr.right->op == SymOp::Const && expr.left && is_named_var(*expr.left, var)) {
            return expr.right->value;
        }
        if (expr.left && expr.left->op == SymOp::Neg && expr.left->left &&
            expr.left->left->op == SymOp::Const && expr.right && is_named_var(*expr.right, var)) {
            return -expr.left->left->value;
        }
        if (expr.right && expr.right->op == SymOp::Neg && expr.right->left &&
            expr.right->left->op == SymOp::Const && expr.left && is_named_var(*expr.left, var)) {
            return -expr.right->left->value;
        }
    }
    return std::nullopt;
}

std::optional<double> match_exp_neg_linear(const SymExpr& expr, const std::string& t_var) {
    if (expr.op != SymOp::Exp || !expr.left) {
        return std::nullopt;
    }
    const SymExpr& inner = *expr.left;
    if (inner.op == SymOp::Neg && inner.left) {
        return extract_positive_scale_of_var(*inner.left, t_var);
    }
    if (inner.op == SymOp::Sub && inner.left && inner.right && is_const_zero(*inner.left)) {
        return extract_positive_scale_of_var(*inner.right, t_var);
    }
    if (const auto scale = extract_positive_scale_of_var(inner, t_var)) {
        if (*scale < 0.0) {
            return -*scale;
        }
    }
    return std::nullopt;
}

std::optional<double> match_exp_neg_quadratic(const SymExpr& expr, const std::string& t_var) {
    if (expr.op != SymOp::Exp || !expr.left) {
        return std::nullopt;
    }
    const SymExpr& inner = *expr.left;
    const SymExpr* negated = nullptr;
    if (inner.op == SymOp::Neg && inner.left) {
        negated = inner.left.get();
    } else if (inner.op == SymOp::Sub && inner.left && inner.right && is_const_zero(*inner.left)) {
        negated = inner.right.get();
    } else {
        return std::nullopt;
    }
    if (!negated || negated->op != SymOp::Mul) {
        return std::nullopt;
    }
    const SymExpr* coef_expr = nullptr;
    const SymExpr* power_expr = nullptr;
    if (negated->left && negated->left->op == SymOp::Const && negated->right) {
        coef_expr = negated->left.get();
        power_expr = negated->right.get();
    } else if (negated->right && negated->right->op == SymOp::Const && negated->left) {
        coef_expr = negated->right.get();
        power_expr = negated->left.get();
    } else {
        return std::nullopt;
    }
    if (!coef_expr || !power_expr || coef_expr->value <= 0.0) {
        return std::nullopt;
    }
    if (power_expr->op == SymOp::Pow && power_expr->left && is_named_var(*power_expr->left, t_var) &&
        power_expr->right && power_expr->right->op == SymOp::Const && power_expr->right->value == 2.0) {
        return coef_expr->value;
    }
    return std::nullopt;
}

std::optional<double> match_rational_decay_form(const SymExpr& expr, const std::string& omega_var) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return std::nullopt;
    }
    double numerator = 0.0;
    if (expr.left->op == SymOp::Const) {
        numerator = expr.left->value;
    } else if (
        expr.left->op == SymOp::Mul && expr.left->left && expr.left->left->op == SymOp::Const &&
        expr.left->right && expr.left->right->op == SymOp::Const) {
        numerator = expr.left->left->value * expr.left->right->value;
    } else {
        return std::nullopt;
    }
    const SymExpr& den = *expr.right;
    if (den.op != SymOp::Add || !den.left || !den.right) {
        return std::nullopt;
    }

    auto match_omega_squared = [&](const SymExpr& term) {
        return term.op == SymOp::Pow && term.left && is_named_var(*term.left, omega_var) &&
               term.right && term.right->op == SymOp::Const && term.right->value == 2.0;
    };

    auto match_a_squared = [&](const SymExpr& term, double& a_out) {
        if (term.op == SymOp::Pow && term.left && term.left->op == SymOp::Const && term.right &&
            term.right->op == SymOp::Const && term.right->value == 2.0 && term.left->value > 0.0) {
            a_out = term.left->value;
            return true;
        }
        if (term.op == SymOp::Const && term.value > 0.0) {
            a_out = std::sqrt(term.value);
            return true;
        }
        return false;
    };

    double a = 0.0;
    if (match_a_squared(*den.left, a) && match_omega_squared(*den.right)) {
        // matched
    } else if (match_a_squared(*den.right, a) && match_omega_squared(*den.left)) {
        // matched
    } else {
        return std::nullopt;
    }
    if (a <= 0.0 || std::abs(numerator - 2.0 * a) > 1e-9) {
        return std::nullopt;
    }
    return a;
}

std::optional<double> match_gaussian_spectrum_form(const SymExpr& expr, const std::string& omega_var) {
    if (expr.op != SymOp::Mul || !expr.left || !expr.right) {
        return std::nullopt;
    }
    const SymExpr* scale_expr = nullptr;
    const SymExpr* gaussian = nullptr;
    if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Exp) {
        scale_expr = expr.left.get();
        gaussian = expr.right.get();
    } else if (expr.right->op == SymOp::Const && expr.left->op == SymOp::Exp) {
        scale_expr = expr.right.get();
        gaussian = expr.left.get();
    } else {
        return std::nullopt;
    }
    if (!scale_expr || !gaussian || !gaussian->left) {
        return std::nullopt;
    }
    const double scale = scale_expr->value;
    if (scale <= 0.0) {
        return std::nullopt;
    }
    const double a = std::numbers::pi / (scale * scale);
    const double expected_scale = std::sqrt(std::numbers::pi / a);
    if (std::abs(scale - expected_scale) > 1e-9) {
        return std::nullopt;
    }
    const SymExpr& inner = *gaussian->left;
    const SymExpr* negated = nullptr;
    if (inner.op == SymOp::Neg && inner.left) {
        negated = inner.left.get();
    } else if (inner.op == SymOp::Sub && inner.left && inner.right && is_const_zero(*inner.left)) {
        negated = inner.right.get();
    } else {
        return std::nullopt;
    }
    if (!negated || negated->op != SymOp::Div || !negated->left || !negated->right) {
        return std::nullopt;
    }
    if (negated->left->op != SymOp::Pow || !negated->left->left || !is_named_var(*negated->left->left, omega_var) ||
        !negated->left->right || negated->left->right->op != SymOp::Const ||
        negated->left->right->value != 2.0) {
        return std::nullopt;
    }
    if (negated->right->op != SymOp::Const || std::abs(negated->right->value - 4.0 * a) > 1e-9) {
        return std::nullopt;
    }
    return a;
}

std::optional<SymExpr> match_geometric_sequence(const SymExpr& expr, const std::string& n_var) {
    if (expr.op == SymOp::Pow && expr.left && expr.right && is_named_var(*expr.right, n_var)) {
        return clone_expr(*expr.left);
    }
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Pow && expr.right->left && expr.right->right &&
            is_named_var(*expr.right->right, n_var)) {
            return sym_mul(clone_expr(*expr.left), clone_expr(*expr.right->left));
        }
        if (expr.right->op == SymOp::Const && expr.left->op == SymOp::Pow && expr.left->left && expr.left->right &&
            is_named_var(*expr.left->right, n_var)) {
            return sym_mul(clone_expr(*expr.right), clone_expr(*expr.left->left));
        }
    }
    return std::nullopt;
}

std::optional<SymExpr> match_z_over_z_minus_a(const SymExpr& expr, const std::string& z_var) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return std::nullopt;
    }
    if (!is_named_var(*expr.left, z_var)) {
        return std::nullopt;
    }
    const SymExpr& den = *expr.right;
    if (den.op != SymOp::Sub || !den.left || !den.right || !is_named_var(*den.left, z_var)) {
        return std::nullopt;
    }
    return clone_expr(*den.right);
}

} // namespace

// Canonical pairs (MVP):
//   exp(-a|t|) ~ exp(-a*t)  <->  2a/(a^2 + omega^2)
//   exp(-a*t^2)             <->  sqrt(pi/a)*exp(-omega^2/(4a))
//   a^n                     <->  z/(z-a)
SymExpr sym_fourier(const SymExpr& expr, const std::string& t_var, const std::string& omega_var) {
    if (const auto a = match_exp_neg_linear(expr, t_var)) {
        if (*a > 0.0) {
            SymExpr a2 = sym_pow(sym_const(*a), sym_const(2.0));
            SymExpr omega2 = sym_pow(sym_var(omega_var), sym_const(2.0));
            return sym_simplify(sym_div(sym_const(2.0 * *a), sym_add(std::move(a2), std::move(omega2))));
        }
    }
    if (const auto a = match_exp_neg_quadratic(expr, t_var)) {
        SymExpr scale = sym_const(std::sqrt(std::numbers::pi / *a));
        SymExpr omega2 = sym_pow(sym_var(omega_var), sym_const(2.0));
        SymExpr exponent = sym_neg(sym_div(std::move(omega2), sym_const(4.0 * *a)));
        return sym_simplify(sym_mul(std::move(scale), sym_exp(std::move(exponent))));
    }
    return sym_transform_unsupported(expr, t_var);
}

SymExpr sym_ifourier(const SymExpr& expr, const std::string& omega_var, const std::string& t_var) {
    if (const auto a = match_rational_decay_form(expr, omega_var)) {
        return sym_simplify(sym_exp(sym_neg(sym_mul(sym_const(*a), sym_var(t_var)))));
    }
    if (const auto a = match_gaussian_spectrum_form(expr, omega_var)) {
        return sym_simplify(sym_exp(sym_neg(sym_mul(sym_const(*a), sym_pow(sym_var(t_var), sym_const(2.0))))));
    }
    return sym_transform_unsupported(expr, omega_var);
}

SymExpr sym_ztransform(const SymExpr& expr, const std::string& n_var, const std::string& z_var) {
    if (auto base = match_geometric_sequence(expr, n_var)) {
        return sym_simplify(sym_div(sym_var(z_var), sym_sub(sym_var(z_var), std::move(*base))));
    }
    if (expr.op == SymOp::Const) {
        return sym_simplify(sym_mul(clone_expr(expr), sym_div(sym_var(z_var), sym_sub(sym_var(z_var), sym_const(1.0)))));
    }
    return sym_transform_unsupported(expr, n_var);
}

SymExpr sym_iztransform(const SymExpr& expr, const std::string& z_var, const std::string& n_var) {
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const && expr.right->op == SymOp::Div) {
            if (auto base = match_z_over_z_minus_a(*expr.right, z_var)) {
                return sym_simplify(sym_mul(clone_expr(*expr.left), sym_pow(std::move(*base), sym_var(n_var))));
            }
        }
        if (expr.right->op == SymOp::Const && expr.left->op == SymOp::Div) {
            if (auto base = match_z_over_z_minus_a(*expr.left, z_var)) {
                return sym_simplify(sym_mul(clone_expr(*expr.right), sym_pow(std::move(*base), sym_var(n_var))));
            }
        }
    }
    if (auto base = match_z_over_z_minus_a(expr, z_var)) {
        return sym_simplify(sym_pow(std::move(*base), sym_var(n_var)));
    }
    if (expr.op == SymOp::Div && expr.left && is_named_var(*expr.left, z_var) && expr.right &&
        expr.right->op == SymOp::Sub && expr.right->left && is_named_var(*expr.right->left, z_var) &&
        expr.right->right && expr.right->right->op == SymOp::Const && expr.right->right->value == 1.0) {
        return sym_const(1.0);
    }
    return sym_transform_unsupported(expr, z_var);
}

} // namespace ms
