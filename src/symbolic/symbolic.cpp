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
    // Keep n=0 exact so inverse matching of a/(a^2+k^2)^{3/2} succeeds with == on doubles.
    if (n == 0) {
        return 1.0;
    }
    const double exponent = (static_cast<double>(n) + 3.0) / 2.0;
    return std::pow(2.0, n + 1) * std::tgamma(exponent) / std::sqrt(std::numbers::pi);
}

bool hankel_scale_matches(double numerator, int n, double a) {
    const double expected = hankel_rpow_exp_scale(n) * a;
    const double tol = 1e-9 * std::max(1.0, std::abs(expected));
    return std::abs(numerator - expected) <= tol;
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
        if (hankel_scale_matches(numerator, n, a)) {
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
    if (std::abs(a_factor - a) <= 1e-12 && hankel_scale_matches(scale * a_factor, n, a)) {
        return std::make_pair(n, a);
    }
    return std::nullopt;
}

SymExpr hankel_forward_rpow_exp_neg(int n, double a, const std::string& k) {
    // Exponent (n+3)/2 for the Fourier–Bessel table entry.
    const double exp_power = 0.5 * (static_cast<double>(n) + 3.0);
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
    // Bare t * exp(-a*t) is the n=1 case (common AST shape instead of t^1 * exp).
    if (is_bare_var(*expr.left, t_var) && expr.right->op == SymOp::Exp) {
        double a = 0.0;
        if (match_exp_neg_at(*expr.right, t_var, a)) {
            return std::make_pair(1, a);
        }
    }
    if (is_bare_var(*expr.right, t_var) && expr.left->op == SymOp::Exp) {
        double a = 0.0;
        if (match_exp_neg_at(*expr.left, t_var, a)) {
            return std::make_pair(1, a);
        }
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
std::optional<SymExpr> ode_try_legacy_separable(const SymExpr& rhs, const std::string& indep_var,
                                                const std::string& dep_var) {
    if (!contains_var_name(rhs, dep_var)) {
        SymExpr integrated = sym_integrate(rhs, indep_var);
        if (sym_is_deriv_sentinel(integrated, indep_var)) {
            return std::nullopt;
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
                return std::nullopt;
            }
            return sym_mul(sym_var("C"), sym_exp(std::move(integrated)));
        }
    }

    return std::nullopt;
}

SymExpr sym_dsolve(const SymExpr& rhs, const std::string& indep_var, const std::string& dep_var) {
    if (auto separable = ode_try_legacy_separable(rhs, indep_var, dep_var)) {
        return std::move(*separable);
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


// ---------------------------------------------------------------------------
// General first-order and second-order ODE solving (sym_dsolve_ode and friends)
//
// Everything below is private to this translation unit and prefixed "ode_" so it
// cannot collide with the older transform helpers that share the file. The public
// entry points are defined at the end of the block.
// ---------------------------------------------------------------------------
namespace {

constexpr int kOdeMaxPolyDegree = 8;      // matches kMaxExpandExponent / kMaxLaplacePower
constexpr int kOdeMaxDepth = 12;          // ode_integrate recursion guard
constexpr double kOdeIdentityTol = 1e-9;  // sampled identity / homogeneity tests
constexpr double kOdeVerifyTol = 1e-6;    // solution residual verification
constexpr double kOdeZeroTol = 1e-12;     // "this coefficient is zero"

// Fixed, deterministic probe grids. The coordinates are mutually distinct, non-zero,
// non-reciprocal and not small integers, so 1/x, log(x), x^n, y/x and x*y are all
// finite and no two grid points coincide under the substitutions made below.
constexpr double kOdeSampleX[3] = {0.37, 1.23, 2.71};
constexpr double kOdeSampleY[3] = {0.53, 1.61, 3.14};
constexpr double kOdeSampleScale[3] = {1.7, 0.43, 3.1};
constexpr double kOdeVerifyX[3] = {1.25, 0.7, 2.3};
constexpr double kOdeVerifyC[3] = {2.0, -1.5, 0.5};

using OdeCoeffs = std::vector<SymExpr>;

// SymExpr is move-only, so a coefficient vector needs an explicit deep-copy loop.
OdeCoeffs ode_poly_clone(const OdeCoeffs& coeffs) {
    OdeCoeffs out;
    out.reserve(coeffs.size());
    for (const SymExpr& c : coeffs) {
        out.push_back(clone_expr(c));
    }
    return out;
}

// try_get_const_value does not simplify first; this does.
bool ode_const_value(const SymExpr& expr, double& out) {
    const SymExpr folded = sym_simplify(clone_expr(expr));
    if (folded.op != SymOp::Const) {
        return false;
    }
    out = folded.value;
    return true;
}

bool ode_coeff_is_zero(const SymExpr& expr) {
    double value = 0.0;
    return ode_const_value(expr, value) && value == 0.0;
}

bool ode_poly_all_zero(const OdeCoeffs& coeffs) {
    for (const SymExpr& c : coeffs) {
        if (!ode_coeff_is_zero(c)) {
            return false;
        }
    }
    return true;
}

void ode_poly_trim(OdeCoeffs& coeffs) {
    while (coeffs.size() > 1 && ode_coeff_is_zero(coeffs.back())) {
        coeffs.pop_back();
    }
    if (coeffs.empty()) {
        coeffs.push_back(sym_const(0.0));
    }
}

OdeCoeffs ode_poly_add(const OdeCoeffs& a, const OdeCoeffs& b, bool subtract) {
    OdeCoeffs out;
    const std::size_t n = std::max(a.size(), b.size());
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool has_a = i < a.size();
        const bool has_b = i < b.size();
        if (has_a && has_b) {
            out.push_back(sym_simplify(subtract ? sym_sub(clone_expr(a[i]), clone_expr(b[i]))
                                                : sym_add(clone_expr(a[i]), clone_expr(b[i]))));
        } else if (has_a) {
            out.push_back(clone_expr(a[i]));
        } else {
            out.push_back(subtract ? sym_simplify(sym_neg(clone_expr(b[i]))) : clone_expr(b[i]));
        }
    }
    ode_poly_trim(out);
    return out;
}

std::optional<OdeCoeffs> ode_poly_mul(const OdeCoeffs& a, const OdeCoeffs& b) {
    if (a.empty() || b.empty()) {
        return std::nullopt;
    }
    const std::size_t degree = a.size() + b.size() - 2;
    if (degree > static_cast<std::size_t>(kOdeMaxPolyDegree)) {
        return std::nullopt;
    }
    OdeCoeffs out;
    out.reserve(degree + 1);
    for (std::size_t i = 0; i <= degree; ++i) {
        out.push_back(sym_const(0.0));
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] = sym_simplify(
                sym_add(std::move(out[i + j]), sym_mul(clone_expr(a[i]), clone_expr(b[j]))));
        }
    }
    ode_poly_trim(out);
    return out;
}

OdeCoeffs ode_poly_deriv(const OdeCoeffs& coeffs);
SymExpr ode_scale(SymExpr expr, double factor);

// Coefficients c[0..d] (each free of `var`) of sum c[k]*var^k, or nullopt when the
// expression is not polynomial in `var`.
std::optional<OdeCoeffs> ode_poly_coeffs(const SymExpr& expr, const std::string& var) {
    if (!contains_var_name(expr, var)) {
        OdeCoeffs out;
        out.push_back(sym_simplify(clone_expr(expr)));
        return out;
    }
    switch (expr.op) {
    case SymOp::Var: {
        OdeCoeffs out;
        out.push_back(sym_const(0.0));
        out.push_back(sym_const(1.0));
        return out;
    }
    case SymOp::Add:
    case SymOp::Sub: {
        if (!expr.left || !expr.right) {
            return std::nullopt;
        }
        auto lhs = ode_poly_coeffs(*expr.left, var);
        if (!lhs) {
            return std::nullopt;
        }
        auto rhs = ode_poly_coeffs(*expr.right, var);
        if (!rhs) {
            return std::nullopt;
        }
        return ode_poly_add(*lhs, *rhs, expr.op == SymOp::Sub);
    }
    case SymOp::Neg: {
        if (!expr.left) {
            return std::nullopt;
        }
        auto inner = ode_poly_coeffs(*expr.left, var);
        if (!inner) {
            return std::nullopt;
        }
        OdeCoeffs out;
        out.reserve(inner->size());
        for (SymExpr& c : *inner) {
            out.push_back(sym_simplify(sym_neg(std::move(c))));
        }
        ode_poly_trim(out);
        return out;
    }
    case SymOp::Mul: {
        if (!expr.left || !expr.right) {
            return std::nullopt;
        }
        auto lhs = ode_poly_coeffs(*expr.left, var);
        if (!lhs) {
            return std::nullopt;
        }
        auto rhs = ode_poly_coeffs(*expr.right, var);
        if (!rhs) {
            return std::nullopt;
        }
        return ode_poly_mul(*lhs, *rhs);
    }
    case SymOp::Div: {
        if (!expr.left || !expr.right || contains_var_name(*expr.right, var)) {
            return std::nullopt;
        }
        auto lhs = ode_poly_coeffs(*expr.left, var);
        if (!lhs) {
            return std::nullopt;
        }
        OdeCoeffs out;
        out.reserve(lhs->size());
        for (SymExpr& c : *lhs) {
            out.push_back(sym_simplify(sym_div(std::move(c), clone_expr(*expr.right))));
        }
        ode_poly_trim(out);
        return out;
    }
    case SymOp::Pow: {
        if (!expr.left || !expr.right) {
            return std::nullopt;
        }
        double exponent = 0.0;
        if (!ode_const_value(*expr.right, exponent)) {
            return std::nullopt;
        }
        if (exponent < 0.0 || exponent != std::floor(exponent) ||
            exponent > static_cast<double>(kOdeMaxPolyDegree)) {
            return std::nullopt;
        }
        auto base = ode_poly_coeffs(*expr.left, var);
        if (!base) {
            return std::nullopt;
        }
        const int power = static_cast<int>(exponent);
        OdeCoeffs acc;
        acc.push_back(sym_const(1.0));
        for (int i = 0; i < power; ++i) {
            auto product = ode_poly_mul(acc, *base);
            if (!product) {
                return std::nullopt;
            }
            acc = std::move(*product);
        }
        return acc;
    }
    default:
        return std::nullopt;
    }
}

std::optional<std::vector<double>> ode_poly_numeric(const SymExpr& expr, const std::string& var) {
    auto coeffs = ode_poly_coeffs(expr, var);
    if (!coeffs) {
        return std::nullopt;
    }
    std::vector<double> out;
    out.reserve(coeffs->size());
    for (const SymExpr& c : *coeffs) {
        double value = 0.0;
        if (!ode_const_value(c, value)) {
            return std::nullopt;
        }
        out.push_back(value);
    }
    return out;
}

// {a, b} for a*var + b.
std::optional<std::pair<double, double>> ode_linear_arg(const SymExpr& expr,
                                                        const std::string& var) {
    auto coeffs = ode_poly_numeric(expr, var);
    if (!coeffs || coeffs->size() > 2) {
        return std::nullopt;
    }
    const double b = (*coeffs)[0];
    const double a = coeffs->size() == 2 ? (*coeffs)[1] : 0.0;
    return std::make_pair(a, b);
}

OdeCoeffs ode_poly_deriv(const OdeCoeffs& coeffs) {
    OdeCoeffs out;
    if (coeffs.size() <= 1) {
        out.push_back(sym_const(0.0));
        return out;
    }
    out.reserve(coeffs.size() - 1);
    for (std::size_t k = 1; k < coeffs.size(); ++k) {
        out.push_back(ode_scale(clone_expr(coeffs[k]), static_cast<double>(k)));
    }
    ode_poly_trim(out);
    return out;
}

SymExpr ode_build_poly(const OdeCoeffs& coeffs, const std::string& var) {
    SymExpr acc = sym_const(0.0);
    bool first = true;
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        if (ode_coeff_is_zero(coeffs[k])) {
            continue;
        }
        SymExpr term;
        if (k == 0) {
            term = sym_simplify(clone_expr(coeffs[k]));
        } else if (k == 1) {
            term = sym_simplify(sym_mul(clone_expr(coeffs[k]), sym_var(var)));
        } else {
            term = sym_simplify(sym_mul(clone_expr(coeffs[k]),
                                        sym_pow(sym_var(var), sym_const(static_cast<double>(k)))));
        }
        acc = first ? std::move(term) : sym_simplify(sym_add(std::move(acc), std::move(term)));
        first = false;
    }
    return acc;
}

OdeCoeffs ode_poly_integrate(const OdeCoeffs& coeffs) {
    OdeCoeffs out;
    out.reserve(coeffs.size() + 1);
    out.push_back(sym_const(0.0));
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        out.push_back(ode_scale(clone_expr(coeffs[k]), 1.0 / static_cast<double>(k + 1)));
    }
    ode_poly_trim(out);
    return out;
}

// --- structural normalisers -------------------------------------------------
// sym_simplify cannot reassociate, so multiplying by a constant naively buries the
// constant one level deeper on every stage. These keep the emitted solutions readable
// and, more importantly, ode_exp_of turns integrating factors into powers, which makes
// strictly more integrands reachable by ode_integrate.

SymExpr ode_scale(SymExpr expr, double factor) {
    if (factor == 1.0) {
        return expr;
    }
    if (expr.op == SymOp::Const) {
        return sym_const(expr.value * factor);
    }
    if (expr.op == SymOp::Neg && expr.left) {
        return ode_scale(take_child(expr.left), -factor);
    }
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const) {
            const double k = expr.left->value;
            return sym_simplify(sym_mul(sym_const(k * factor), take_child(expr.right)));
        }
        if (expr.right->op == SymOp::Const) {
            const double k = expr.right->value;
            return sym_simplify(sym_mul(sym_const(k * factor), take_child(expr.left)));
        }
    }
    if (expr.op == SymOp::Div && expr.left && expr.right && expr.right->op == SymOp::Const &&
        expr.right->value != 0.0) {
        const double k = expr.right->value;
        return ode_scale(take_child(expr.left), factor / k);
    }
    return sym_simplify(sym_mul(sym_const(factor), std::move(expr)));
}

SymExpr ode_reciprocal(SymExpr expr) {
    if (expr.op == SymOp::Const && expr.value != 0.0) {
        return sym_const(1.0 / expr.value);
    }
    if (expr.op == SymOp::Pow && expr.left && expr.right && expr.right->op == SymOp::Const) {
        const double k = expr.right->value;
        return sym_simplify(sym_pow(take_child(expr.left), sym_const(-k)));
    }
    if (expr.op == SymOp::Exp && expr.left) {
        return sym_exp(sym_simplify(sym_neg(take_child(expr.left))));
    }
    return sym_simplify(sym_pow(std::move(expr), sym_const(-1.0)));
}

bool ode_is_log_shaped(const SymExpr& expr) {
    if (expr.op == SymOp::Log) {
        return true;
    }
    if (expr.op == SymOp::Neg && expr.left) {
        return ode_is_log_shaped(*expr.left);
    }
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const) {
            return ode_is_log_shaped(*expr.right);
        }
        if (expr.right->op == SymOp::Const) {
            return ode_is_log_shaped(*expr.left);
        }
    }
    if (expr.op == SymOp::Div && expr.left && expr.right && expr.right->op == SymOp::Const) {
        return ode_is_log_shaped(*expr.left);
    }
    return false;
}

// exp(w), rewritten to a power whenever w is log-shaped.
SymExpr ode_exp_of(SymExpr arg) {
    SymExpr w = sym_simplify(std::move(arg));
    if (w.op == SymOp::Const) {
        return sym_const(std::exp(w.value));
    }
    if (w.op == SymOp::Log && w.left) {
        return take_child(w.left);
    }
    if (w.op == SymOp::Neg && w.left) {
        return ode_reciprocal(ode_exp_of(take_child(w.left)));
    }
    if (w.op == SymOp::Mul && w.left && w.right) {
        if (w.left->op == SymOp::Const && w.right->op == SymOp::Log && w.right->left) {
            const double k = w.left->value;
            return sym_simplify(sym_pow(take_child(w.right->left), sym_const(k)));
        }
        if (w.right->op == SymOp::Const && w.left->op == SymOp::Log && w.left->left) {
            const double k = w.right->value;
            return sym_simplify(sym_pow(take_child(w.left->left), sym_const(k)));
        }
    }
    if (w.op == SymOp::Div && w.left && w.right && w.right->op == SymOp::Const &&
        w.right->value != 0.0 && w.left->op == SymOp::Log && w.left->left) {
        const double k = w.right->value;
        return sym_simplify(sym_pow(take_child(w.left->left), sym_const(1.0 / k)));
    }
    if (w.op == SymOp::Add && w.left && w.right &&
        (ode_is_log_shaped(*w.left) || ode_is_log_shaped(*w.right))) {
        SymExpr lhs = ode_exp_of(take_child(w.left));
        SymExpr rhs = ode_exp_of(take_child(w.right));
        return sym_simplify(sym_mul(std::move(lhs), std::move(rhs)));
    }
    return sym_exp(std::move(w));
}

SymExpr ode_div_by(SymExpr numerator, const SymExpr& divisor) {
    if (divisor.op == SymOp::Const && divisor.value != 0.0) {
        return ode_scale(std::move(numerator), 1.0 / divisor.value);
    }
    return sym_simplify(sym_mul(std::move(numerator), ode_reciprocal(clone_expr(divisor))));
}

// --- numeric polynomial vectors and rational functions ----------------------

std::vector<double> ode_pv_trim(std::vector<double> coeffs) {
    while (coeffs.size() > 1 && std::abs(coeffs.back()) <= kOdeZeroTol) {
        coeffs.pop_back();
    }
    if (coeffs.empty()) {
        coeffs.push_back(0.0);
    }
    return coeffs;
}

bool ode_pv_is_zero(const std::vector<double>& coeffs) {
    for (const double c : coeffs) {
        if (std::abs(c) > kOdeZeroTol) {
            return false;
        }
    }
    return true;
}

std::vector<double> ode_pv_add(const std::vector<double>& a, const std::vector<double>& b,
                               bool subtract) {
    std::vector<double> out(std::max(a.size(), b.size()), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] += a[i];
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        out[i] += subtract ? -b[i] : b[i];
    }
    return ode_pv_trim(std::move(out));
}

std::vector<double> ode_pv_mul(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty()) {
        return {0.0};
    }
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] += a[i] * b[j];
        }
    }
    return ode_pv_trim(std::move(out));
}

OdeCoeffs ode_pv_to_coeffs(const std::vector<double>& values) {
    OdeCoeffs out;
    out.reserve(values.size());
    for (const double v : values) {
        out.push_back(sym_const(v));
    }
    return out;
}

struct OdeRational {
    std::vector<double> num;
    std::vector<double> den;
};

bool ode_rational_in_range(const OdeRational& r) {
    const std::size_t cap = static_cast<std::size_t>(kOdeMaxPolyDegree) + 1;
    return r.num.size() <= cap && r.den.size() <= cap;
}

std::optional<OdeRational> ode_rational_of(const SymExpr& expr, const std::string& var) {
    if (!contains_var_name(expr, var)) {
        double value = 0.0;
        if (!ode_const_value(expr, value)) {
            return std::nullopt;
        }
        return OdeRational{{value}, {1.0}};
    }
    switch (expr.op) {
    case SymOp::Var:
        return OdeRational{{0.0, 1.0}, {1.0}};
    case SymOp::Add:
    case SymOp::Sub:
    case SymOp::Mul:
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return std::nullopt;
        }
        auto lhs = ode_rational_of(*expr.left, var);
        if (!lhs) {
            return std::nullopt;
        }
        auto rhs = ode_rational_of(*expr.right, var);
        if (!rhs) {
            return std::nullopt;
        }
        OdeRational out;
        if (expr.op == SymOp::Add || expr.op == SymOp::Sub) {
            out.num = ode_pv_add(ode_pv_mul(lhs->num, rhs->den), ode_pv_mul(rhs->num, lhs->den),
                                 expr.op == SymOp::Sub);
            out.den = ode_pv_mul(lhs->den, rhs->den);
        } else if (expr.op == SymOp::Mul) {
            out.num = ode_pv_mul(lhs->num, rhs->num);
            out.den = ode_pv_mul(lhs->den, rhs->den);
        } else {
            out.num = ode_pv_mul(lhs->num, rhs->den);
            out.den = ode_pv_mul(lhs->den, rhs->num);
        }
        if (ode_pv_is_zero(out.den) || !ode_rational_in_range(out)) {
            return std::nullopt;
        }
        return out;
    }
    case SymOp::Neg: {
        if (!expr.left) {
            return std::nullopt;
        }
        auto inner = ode_rational_of(*expr.left, var);
        if (!inner) {
            return std::nullopt;
        }
        for (double& c : inner->num) {
            c = -c;
        }
        return inner;
    }
    case SymOp::Pow: {
        if (!expr.left || !expr.right) {
            return std::nullopt;
        }
        double exponent = 0.0;
        if (!ode_const_value(*expr.right, exponent) || exponent != std::floor(exponent) ||
            std::abs(exponent) > static_cast<double>(kOdeMaxPolyDegree)) {
            return std::nullopt;
        }
        auto base = ode_rational_of(*expr.left, var);
        if (!base) {
            return std::nullopt;
        }
        const int power = static_cast<int>(std::abs(exponent));
        OdeRational out{{1.0}, {1.0}};
        for (int i = 0; i < power; ++i) {
            out.num = ode_pv_mul(out.num, base->num);
            out.den = ode_pv_mul(out.den, base->den);
            if (!ode_rational_in_range(out)) {
                return std::nullopt;
            }
        }
        if (exponent < 0.0) {
            std::swap(out.num, out.den);
        }
        if (ode_pv_is_zero(out.den)) {
            return std::nullopt;
        }
        return out;
    }
    default:
        return std::nullopt;
    }
}

// --- the ODE-private integrator ---------------------------------------------
// Returns nullopt (not the Deriv sentinel) on failure; callers translate that.

std::optional<SymExpr> ode_integrate(const SymExpr& expr, const std::string& var, int depth = 0);

// Closed-form integration by parts for polynomial x {exp, sin, cos} with a linear argument.
//   int P e^u      = e^u  sum_k (-1)^k P^(k) / a^(k+1)
//   int P sin(u)   = -cos(u) E + sin(u) O
//   int P cos(u)   =  sin(u) E + cos(u) O
// with E = sum_j (-1)^j P^(2j) / a^(2j+1) and O = sum_j (-1)^j P^(2j+1) / a^(2j+2).
std::optional<SymExpr> ode_integrate_poly_kernel(const OdeCoeffs& poly, const SymExpr& kernel,
                                                 const std::string& var) {
    if (!kernel.left) {
        return std::nullopt;
    }
    const auto linear = ode_linear_arg(*kernel.left, var);
    if (!linear || linear->first == 0.0) {
        return std::nullopt;
    }
    const double a = linear->first;
    SymExpr arg = sym_simplify(clone_expr(*kernel.left));

    if (kernel.op == SymOp::Exp) {
        SymExpr acc = sym_const(0.0);
        OdeCoeffs d = ode_poly_clone(poly);
        double apow = a;
        double sign = 1.0;
        for (int k = 0; k <= kOdeMaxPolyDegree + 1; ++k) {
            if (ode_poly_all_zero(d)) {
                break;
            }
            acc = sym_simplify(
                sym_add(std::move(acc), ode_scale(ode_build_poly(d, var), sign / apow)));
            d = ode_poly_deriv(d);
            apow *= a;
            sign = -sign;
        }
        return sym_simplify(sym_mul(std::move(acc), sym_exp(std::move(arg))));
    }

    if (kernel.op != SymOp::Sin && kernel.op != SymOp::Cos) {
        return std::nullopt;
    }
    SymExpr even = sym_const(0.0);
    SymExpr odd = sym_const(0.0);
    OdeCoeffs d = ode_poly_clone(poly);
    double apow = a;
    double sign = 1.0;
    for (int k = 0; k <= kOdeMaxPolyDegree + 1; ++k) {
        if (ode_poly_all_zero(d)) {
            break;
        }
        SymExpr term = ode_scale(ode_build_poly(d, var), sign / apow);
        if (k % 2 == 0) {
            even = sym_simplify(sym_add(std::move(even), std::move(term)));
        } else {
            odd = sym_simplify(sym_add(std::move(odd), std::move(term)));
            sign = -sign;
        }
        d = ode_poly_deriv(d);
        apow *= a;
    }
    if (kernel.op == SymOp::Sin) {
        SymExpr first = sym_simplify(sym_mul(sym_simplify(sym_neg(std::move(even))),
                                             sym_cos(clone_expr(arg))));
        SymExpr second = sym_simplify(sym_mul(std::move(odd), sym_sin(std::move(arg))));
        return sym_simplify(sym_add(std::move(first), std::move(second)));
    }
    SymExpr first = sym_simplify(sym_mul(std::move(even), sym_sin(clone_expr(arg))));
    SymExpr second = sym_simplify(sym_mul(std::move(odd), sym_cos(std::move(arg))));
    return sym_simplify(sym_add(std::move(first), std::move(second)));
}

//   int e^(ax+p) sin(bx+q) = e^(ax+p) (a sin(bx+q) - b cos(bx+q)) / (a^2 + b^2)
//   int e^(ax+p) cos(bx+q) = e^(ax+p) (a cos(bx+q) + b sin(bx+q)) / (a^2 + b^2)
std::optional<SymExpr> ode_integrate_exp_trig(const SymExpr& exponential, const SymExpr& trig,
                                              const std::string& var) {
    if (!exponential.left || !trig.left) {
        return std::nullopt;
    }
    if (trig.op != SymOp::Sin && trig.op != SymOp::Cos) {
        return std::nullopt;
    }
    const auto exp_arg = ode_linear_arg(*exponential.left, var);
    const auto trig_arg = ode_linear_arg(*trig.left, var);
    if (!exp_arg || !trig_arg) {
        return std::nullopt;
    }
    const double a = exp_arg->first;
    const double b = trig_arg->first;
    const double denominator = a * a + b * b;
    if (denominator == 0.0) {
        return std::nullopt;
    }
    const SymExpr kernel_arg = sym_simplify(clone_expr(*trig.left));
    SymExpr inner;
    if (trig.op == SymOp::Sin) {
        inner = sym_simplify(sym_sub(ode_scale(sym_sin(clone_expr(kernel_arg)), a),
                                     ode_scale(sym_cos(clone_expr(kernel_arg)), b)));
    } else {
        inner = sym_simplify(sym_add(ode_scale(sym_cos(clone_expr(kernel_arg)), a),
                                     ode_scale(sym_sin(clone_expr(kernel_arg)), b)));
    }
    SymExpr product =
        sym_mul(sym_exp(sym_simplify(clone_expr(*exponential.left))), std::move(inner));
    return sym_simplify(sym_div(std::move(product), sym_const(denominator)));
}

bool ode_is_kernel(const SymExpr& expr) {
    return expr.op == SymOp::Exp || expr.op == SymOp::Sin || expr.op == SymOp::Cos;
}

std::optional<SymExpr> ode_integrate(const SymExpr& expr, const std::string& var, int depth) {
    if (depth > kOdeMaxDepth) {
        return std::nullopt;
    }
    if (!contains_var_name(expr, var)) {
        return sym_simplify(sym_mul(sym_simplify(clone_expr(expr)), sym_var(var)));
    }
    switch (expr.op) {
    case SymOp::Add:
    case SymOp::Sub: {
        if (!expr.left || !expr.right) {
            return std::nullopt;
        }
        auto lhs = ode_integrate(*expr.left, var, depth + 1);
        if (!lhs) {
            return std::nullopt;
        }
        auto rhs = ode_integrate(*expr.right, var, depth + 1);
        if (!rhs) {
            return std::nullopt;
        }
        return sym_simplify(expr.op == SymOp::Add ? sym_add(std::move(*lhs), std::move(*rhs))
                                                  : sym_sub(std::move(*lhs), std::move(*rhs)));
    }
    case SymOp::Neg: {
        if (!expr.left) {
            return std::nullopt;
        }
        auto inner = ode_integrate(*expr.left, var, depth + 1);
        if (!inner) {
            return std::nullopt;
        }
        return sym_simplify(sym_neg(std::move(*inner)));
    }
    default:
        break;
    }

    if (auto poly = ode_poly_coeffs(expr, var)) {
        return ode_build_poly(ode_poly_integrate(*poly), var);
    }

    if (ode_is_kernel(expr)) {
        OdeCoeffs unit;
        unit.push_back(sym_const(1.0));
        return ode_integrate_poly_kernel(unit, expr, var);
    }

    if (expr.op == SymOp::Div && expr.left && expr.right) {
        if (!contains_var_name(*expr.right, var)) {
            auto inner = ode_integrate(*expr.left, var, depth + 1);
            if (!inner) {
                return std::nullopt;
            }
            return sym_simplify(sym_div(std::move(*inner), sym_simplify(clone_expr(*expr.right))));
        }
        if (!contains_var_name(*expr.left, var)) {
            if (const auto linear = ode_linear_arg(*expr.right, var)) {
                if (linear->first != 0.0) {
                    return sym_simplify(
                        sym_div(sym_mul(sym_simplify(clone_expr(*expr.left)),
                                        sym_log(sym_simplify(clone_expr(*expr.right)))),
                                sym_const(linear->first)));
                }
            }
            const SymExpr& den = *expr.right;
            if (den.op == SymOp::Pow && den.left && den.right) {
                double n = 0.0;
                const auto linear = ode_linear_arg(*den.left, var);
                if (linear && linear->first != 0.0 && ode_const_value(*den.right, n) && n != 1.0) {
                    return sym_simplify(sym_div(
                        sym_mul(sym_simplify(clone_expr(*expr.left)),
                                sym_pow(sym_simplify(clone_expr(*den.left)), sym_const(1.0 - n))),
                        sym_const(linear->first * (1.0 - n))));
                }
            }
        }
        return std::nullopt;
    }

    if (expr.op == SymOp::Pow && expr.left && expr.right) {
        double n = 0.0;
        if (ode_const_value(*expr.right, n) && !contains_var_name(*expr.right, var)) {
            const auto linear = ode_linear_arg(*expr.left, var);
            if (linear && linear->first != 0.0) {
                SymExpr base = sym_simplify(clone_expr(*expr.left));
                if (n == -1.0) {
                    return sym_simplify(
                        sym_div(sym_log(std::move(base)), sym_const(linear->first)));
                }
                return sym_simplify(sym_div(sym_pow(std::move(base), sym_const(n + 1.0)),
                                            sym_const(linear->first * (n + 1.0))));
            }
        }
        double base_value = 0.0;
        if (ode_const_value(*expr.left, base_value) && base_value > 0.0) {
            const auto linear = ode_linear_arg(*expr.right, var);
            if (linear && linear->first != 0.0) {
                return sym_simplify(sym_div(sym_simplify(clone_expr(expr)),
                                            sym_const(linear->first * std::log(base_value))));
            }
        }
        return std::nullopt;
    }

    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        const SymExpr& lhs = *expr.left;
        const SymExpr& rhs = *expr.right;
        if (!contains_var_name(lhs, var)) {
            auto inner = ode_integrate(rhs, var, depth + 1);
            if (!inner) {
                return std::nullopt;
            }
            return sym_simplify(sym_mul(sym_simplify(clone_expr(lhs)), std::move(*inner)));
        }
        if (!contains_var_name(rhs, var)) {
            auto inner = ode_integrate(lhs, var, depth + 1);
            if (!inner) {
                return std::nullopt;
            }
            return sym_simplify(sym_mul(sym_simplify(clone_expr(rhs)), std::move(*inner)));
        }
        if (lhs.op == SymOp::Exp && rhs.op == SymOp::Exp && lhs.left && rhs.left) {
            SymExpr merged = sym_exp(sym_simplify(
                sym_add(clone_expr(*lhs.left), clone_expr(*rhs.left))));
            return ode_integrate(merged, var, depth + 1);
        }
        if (lhs.op == SymOp::Exp && (rhs.op == SymOp::Sin || rhs.op == SymOp::Cos)) {
            return ode_integrate_exp_trig(lhs, rhs, var);
        }
        if (rhs.op == SymOp::Exp && (lhs.op == SymOp::Sin || lhs.op == SymOp::Cos)) {
            return ode_integrate_exp_trig(rhs, lhs, var);
        }
        if (ode_is_kernel(lhs)) {
            if (auto poly = ode_poly_coeffs(rhs, var)) {
                return ode_integrate_poly_kernel(*poly, lhs, var);
            }
        }
        if (ode_is_kernel(rhs)) {
            if (auto poly = ode_poly_coeffs(lhs, var)) {
                return ode_integrate_poly_kernel(*poly, rhs, var);
            }
        }
        return std::nullopt;
    }

    return std::nullopt;
}

// One-variable rational integration; used by the homogeneous stage. Denominators of
// degree > 2, and irreducible quadratics (which would need an arctangent the AST has
// no node for), are declined.
std::optional<SymExpr> ode_integrate_rational(std::vector<double> num, std::vector<double> den,
                                              const std::string& var) {
    num = ode_pv_trim(std::move(num));
    den = ode_pv_trim(std::move(den));
    if (ode_pv_is_zero(den)) {
        return std::nullopt;
    }
    if (den.size() == 1) {
        std::vector<double> scaled = num;
        for (double& c : scaled) {
            c /= den[0];
        }
        return ode_build_poly(ode_poly_integrate(ode_pv_to_coeffs(scaled)), var);
    }
    if (den.size() == 2) {
        const double b1 = den[1];
        const double b0 = den[0];
        std::vector<double> quotient(num.size() >= 2 ? num.size() - 1 : 1, 0.0);
        std::vector<double> remainder = num;
        for (std::size_t i = num.size(); i-- > 1;) {
            const double q = remainder[i] / b1;
            quotient[i - 1] = q;
            remainder[i] = 0.0;
            remainder[i - 1] -= q * b0;
        }
        SymExpr acc = ode_build_poly(ode_poly_integrate(ode_pv_to_coeffs(quotient)), var);
        const double r = remainder[0];
        if (std::abs(r) > kOdeZeroTol) {
            SymExpr linear = ode_build_poly(ode_pv_to_coeffs(den), var);
            SymExpr term = ode_scale(sym_log(std::move(linear)), r / b1);
            acc = sym_simplify(sym_add(std::move(acc), std::move(term)));
        }
        return acc;
    }
    if (den.size() == 3 && num.size() <= 2) {
        const double a2 = den[2];
        const double a1 = den[1];
        const double a0 = den[0];
        const double n1 = num.size() > 1 ? num[1] : 0.0;
        const double n0 = num[0];
        const double disc = a1 * a1 - 4.0 * a2 * a0;
        const double disc_scale = std::max(1.0, std::abs(a1 * a1) + std::abs(4.0 * a2 * a0));
        if (std::abs(disc) <= kOdeZeroTol * disc_scale) {
            const double r = -a1 / (2.0 * a2);
            SymExpr shifted = sym_simplify(sym_sub(sym_var(var), sym_const(r)));
            SymExpr acc = sym_const(0.0);
            bool first = true;
            if (std::abs(n1) > kOdeZeroTol) {
                acc = ode_scale(sym_log(clone_expr(shifted)), n1 / a2);
                first = false;
            }
            const double k = -(n1 * r + n0) / a2;
            if (std::abs(k) > kOdeZeroTol) {
                SymExpr term = ode_scale(
                    sym_simplify(sym_pow(std::move(shifted), sym_const(-1.0))), k);
                acc = first ? std::move(term)
                            : sym_simplify(sym_add(std::move(acc), std::move(term)));
            }
            return acc;
        }
        if (disc < 0.0) {
            return std::nullopt;
        }
        const double root = std::sqrt(disc);
        const double r1 = (-a1 + root) / (2.0 * a2);
        const double r2 = (-a1 - root) / (2.0 * a2);
        const double coef_a = (n1 * r1 + n0) / (a2 * (r1 - r2));
        const double coef_b = (n1 * r2 + n0) / (a2 * (r2 - r1));
        SymExpr acc = sym_const(0.0);
        bool first = true;
        if (std::abs(coef_a) > kOdeZeroTol) {
            acc = ode_scale(sym_log(sym_simplify(sym_sub(sym_var(var), sym_const(r1)))), coef_a);
            first = false;
        }
        if (std::abs(coef_b) > kOdeZeroTol) {
            SymExpr term =
                ode_scale(sym_log(sym_simplify(sym_sub(sym_var(var), sym_const(r2)))), coef_b);
            acc = first ? std::move(term) : sym_simplify(sym_add(std::move(acc), std::move(term)));
        }
        return acc;
    }
    return std::nullopt;
}

// --- sampled numeric predicates ---------------------------------------------
// These can only cause a candidate solution to be CONSTRUCTED. Every candidate is then
// run through ode_verify_solution, which differentiates it symbolically and exactly with
// sym_diff, so a sampling false positive cannot escape as a wrong answer - the worst case
// is a discarded candidate and a fall-through to the sentinel.

bool ode_same_function(const SymExpr& a, const SymExpr& b, const std::string& x,
                       const std::string& y) {
    if (sym_to_string(sym_simplify(sym_expand(sym_sub(clone_expr(a), clone_expr(b))))) ==
        "0.000000") {
        return true;
    }
    int usable = 0;
    for (const double xv : kOdeSampleX) {
        for (const double yv : kOdeSampleY) {
            const std::map<std::string, double> env{{x, xv}, {y, yv}};
            const double av = sym_eval(a, env);
            const double bv = sym_eval(b, env);
            if (!std::isfinite(av) || !std::isfinite(bv)) {
                continue;
            }
            ++usable;
            if (std::abs(av - bv) > kOdeIdentityTol * (1.0 + std::abs(av) + std::abs(bv))) {
                return false;
            }
        }
    }
    return usable >= 6;
}

bool ode_numeric_zero(const SymExpr& expr, const std::string& x, const std::string& y) {
    int usable = 0;
    for (const double xv : kOdeSampleX) {
        for (const double yv : kOdeSampleY) {
            const double value = sym_eval(expr, {{x, xv}, {y, yv}});
            if (!std::isfinite(value)) {
                continue;
            }
            ++usable;
            if (std::abs(value) > kOdeIdentityTol) {
                return false;
            }
        }
    }
    return usable >= 6;
}

bool ode_is_homogeneous_degree0(const SymExpr& rhs, const std::string& x, const std::string& y) {
    int usable = 0;
    for (const double xv : kOdeSampleX) {
        for (const double yv : kOdeSampleY) {
            const double base = sym_eval(rhs, {{x, xv}, {y, yv}});
            if (!std::isfinite(base)) {
                continue;
            }
            for (const double t : kOdeSampleScale) {
                const double scaled = sym_eval(rhs, {{x, t * xv}, {y, t * yv}});
                if (!std::isfinite(scaled)) {
                    continue;
                }
                ++usable;
                if (std::abs(scaled - base) >
                    kOdeIdentityTol * (1.0 + std::abs(base) + std::abs(scaled))) {
                    return false;
                }
            }
        }
    }
    return usable >= 9;
}

// The safety net: differentiate the candidate exactly and substitute it back into the ODE.
bool ode_verify_solution(const SymExpr& rhs, const SymExpr& solution, const std::string& x,
                         const std::string& y) {
    const SymExpr derivative = sym_simplify(sym_diff(clone_expr(solution), x));
    int usable = 0;
    for (const double xv : kOdeVerifyX) {
        for (const double cv : kOdeVerifyC) {
            const std::map<std::string, double> env{{x, xv}, {"C", cv}};
            const double yv = sym_eval(solution, env);
            if (!std::isfinite(yv)) {
                continue;
            }
            const double dv = sym_eval(derivative, env);
            std::map<std::string, double> full = env;
            full[y] = yv;
            const double rv = sym_eval(rhs, full);
            if (!std::isfinite(dv) || !std::isfinite(rv)) {
                continue;
            }
            ++usable;
            if (std::abs(dv - rv) > kOdeVerifyTol * (1.0 + std::abs(dv) + std::abs(rv))) {
                return false;
            }
        }
    }
    return usable >= 5;
}

// --- decomposition into powers of the dependent variable --------------------

bool ode_gather_term(const SymExpr& expr, const std::string& dep, double& power, SymExpr& coef) {
    switch (expr.op) {
    case SymOp::Mul:
        if (!expr.left || !expr.right) {
            return false;
        }
        return ode_gather_term(*expr.left, dep, power, coef) &&
               ode_gather_term(*expr.right, dep, power, coef);
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return false;
        }
        if (!ode_gather_term(*expr.left, dep, power, coef)) {
            return false;
        }
        const SymExpr& den = *expr.right;
        if (is_bare_var(den, dep)) {
            power -= 1.0;
            return true;
        }
        if (den.op == SymOp::Pow && den.left && den.right && is_bare_var(*den.left, dep) &&
            den.right->op == SymOp::Const) {
            power -= den.right->value;
            return true;
        }
        if (contains_var_name(den, dep)) {
            return false;
        }
        coef = sym_simplify(sym_div(std::move(coef), clone_expr(den)));
        return true;
    }
    case SymOp::Neg:
        if (!expr.left) {
            return false;
        }
        coef = sym_simplify(sym_neg(std::move(coef)));
        return ode_gather_term(*expr.left, dep, power, coef);
    case SymOp::Pow: {
        if (!expr.left || !expr.right) {
            return false;
        }
        if (!contains_var_name(expr, dep)) {
            coef = sym_simplify(sym_mul(std::move(coef), clone_expr(expr)));
            return true;
        }
        if (is_bare_var(*expr.left, dep) && expr.right->op == SymOp::Const) {
            power += expr.right->value;
            return true;
        }
        return false;
    }
    default:
        break;
    }
    if (is_bare_var(expr, dep)) {
        power += 1.0;
        return true;
    }
    if (contains_var_name(expr, dep)) {
        return false;
    }
    coef = sym_simplify(sym_mul(std::move(coef), clone_expr(expr)));
    return true;
}

void ode_merge_power(std::map<double, SymExpr>& out, double power, SymExpr coef) {
    const auto it = out.find(power);
    if (it == out.end()) {
        out.emplace(power, sym_simplify(std::move(coef)));
        return;
    }
    it->second = sym_simplify(sym_add(std::move(it->second), std::move(coef)));
}

// exponent -> coefficient (each coefficient free of dep_var). false on any shape that is
// not a sum of coefficient*dep^k terms, which is exactly what keeps sin(y), y*sin(y) and
// friends out of the linear and Bernoulli stages.
bool ode_collect_powers(const SymExpr& expr, const std::string& dep_var, bool negate,
                        std::map<double, SymExpr>& out) {
    switch (expr.op) {
    case SymOp::Add:
        if (!expr.left || !expr.right) {
            return false;
        }
        return ode_collect_powers(*expr.left, dep_var, negate, out) &&
               ode_collect_powers(*expr.right, dep_var, negate, out);
    case SymOp::Sub:
        if (!expr.left || !expr.right) {
            return false;
        }
        return ode_collect_powers(*expr.left, dep_var, negate, out) &&
               ode_collect_powers(*expr.right, dep_var, !negate, out);
    case SymOp::Neg:
        if (!expr.left) {
            return false;
        }
        return ode_collect_powers(*expr.left, dep_var, !negate, out);
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            break;
        }
        if (contains_var_name(*expr.right, dep_var)) {
            break;
        }
        const SymOp inner = expr.left->op;
        if (inner != SymOp::Add && inner != SymOp::Sub && inner != SymOp::Neg) {
            break;
        }
        std::map<double, SymExpr> nested;
        if (!ode_collect_powers(*expr.left, dep_var, negate, nested)) {
            return false;
        }
        for (auto& entry : nested) {
            ode_merge_power(out, entry.first, ode_div_by(std::move(entry.second), *expr.right));
        }
        return true;
    }
    default:
        break;
    }
    double power = 0.0;
    SymExpr coef = sym_const(1.0);
    if (!ode_gather_term(expr, dep_var, power, coef)) {
        return false;
    }
    if (negate) {
        coef = sym_simplify(sym_neg(std::move(coef)));
    }
    ode_merge_power(out, power, std::move(coef));
    return true;
}

SymExpr ode_power_coefficient(const std::map<double, SymExpr>& powers, double exponent) {
    const auto it = powers.find(exponent);
    return it == powers.end() ? sym_const(0.0) : clone_expr(it->second);
}

// --- stage 3: first-order linear --------------------------------------------
// y' = A(x) y + B(x)  <=>  y' + p y = q with p = -A, q = B.
// mu = exp(integral p), y = (integral mu*q + C) / mu.
std::optional<SymExpr> ode_stage_linear(const std::map<double, SymExpr>& powers,
                                        const std::string& indep_var) {
    for (const auto& entry : powers) {
        if (entry.first != 0.0 && entry.first != 1.0) {
            return std::nullopt;
        }
    }
    const SymExpr a_coef = ode_power_coefficient(powers, 1.0);
    const SymExpr b_coef = ode_power_coefficient(powers, 0.0);
    auto integral_p = ode_integrate(sym_simplify(sym_neg(clone_expr(a_coef))), indep_var);
    if (!integral_p) {
        return std::nullopt;
    }
    SymExpr mu = ode_exp_of(std::move(*integral_p));
    auto integral_q =
        ode_integrate(sym_simplify(sym_mul(clone_expr(mu), clone_expr(b_coef))), indep_var);
    if (!integral_q) {
        return std::nullopt;
    }
    SymExpr numerator = sym_simplify(sym_add(std::move(*integral_q), sym_var("C")));
    return ode_div_by(std::move(numerator), mu);
}

// --- stage 4: Bernoulli ------------------------------------------------------
// y' = A(x) y + B(x) y^n with n not in {0, 1}. v = y^(1-n) turns it linear:
// v' + m p v = m q with m = 1 - n, p = -A, q = B; then y = v^(1/m).
std::optional<SymExpr> ode_stage_bernoulli(const std::map<double, SymExpr>& powers,
                                           const std::string& indep_var) {
    if (powers.size() > 2 || powers.find(0.0) != powers.end()) {
        return std::nullopt;
    }
    double exponent = 0.0;
    int found = 0;
    for (const auto& entry : powers) {
        if (entry.first != 1.0) {
            exponent = entry.first;
            ++found;
        }
    }
    if (found != 1) {
        return std::nullopt;
    }
    const double m = 1.0 - exponent;
    if (m == 0.0) {
        return std::nullopt;
    }
    const SymExpr a_coef = ode_power_coefficient(powers, 1.0);
    const SymExpr b_coef = ode_power_coefficient(powers, exponent);
    SymExpr scaled_p =
        sym_simplify(sym_mul(sym_const(m), sym_simplify(sym_neg(clone_expr(a_coef)))));
    SymExpr scaled_q = sym_simplify(sym_mul(sym_const(m), clone_expr(b_coef)));
    auto integral_p = ode_integrate(scaled_p, indep_var);
    if (!integral_p) {
        return std::nullopt;
    }
    SymExpr mu = ode_exp_of(std::move(*integral_p));
    auto integral_q = ode_integrate(sym_simplify(sym_mul(clone_expr(mu), std::move(scaled_q))),
                                    indep_var);
    if (!integral_q) {
        return std::nullopt;
    }
    SymExpr numerator = sym_simplify(sym_add(std::move(*integral_q), sym_var("C")));
    SymExpr v = ode_div_by(std::move(numerator), mu);
    return sym_simplify(sym_pow(std::move(v), sym_const(1.0 / m)));
}

// --- stage 5: exact equation, made explicit in the dependent variable --------

std::optional<SymExpr> ode_solve_implicit_for_dep(const SymExpr& relation,
                                                  const std::string& dep_var) {
    auto coeffs = ode_poly_coeffs(relation, dep_var);
    if (!coeffs) {
        return std::nullopt;
    }
    if (coeffs->size() == 2) {
        return sym_simplify(
            sym_div(sym_neg(clone_expr((*coeffs)[0])), clone_expr((*coeffs)[1])));
    }
    if (coeffs->size() == 3) {
        const SymExpr& p0 = (*coeffs)[0];
        const SymExpr& p1 = (*coeffs)[1];
        const SymExpr& p2 = (*coeffs)[2];
        SymExpr discriminant = sym_sub(
            sym_pow(clone_expr(p1), sym_const(2.0)),
            sym_mul(sym_const(4.0), sym_mul(clone_expr(p2), clone_expr(p0))));
        return sym_simplify(sym_div(
            sym_add(sym_neg(clone_expr(p1)), sym_sqrt(std::move(discriminant))),
            sym_mul(sym_const(2.0), clone_expr(p2))));
    }
    return std::nullopt;
}

std::optional<SymExpr> ode_stage_exact_explicit(const SymExpr& rhs, const std::string& indep_var,
                                                const std::string& dep_var) {
    const SymExpr* quotient = nullptr;
    bool negated = false;
    if (rhs.op == SymOp::Div) {
        quotient = &rhs;
    } else if (rhs.op == SymOp::Neg && rhs.left && rhs.left->op == SymOp::Div) {
        quotient = rhs.left.get();
        negated = true;
    }
    if (quotient == nullptr || !quotient->left || !quotient->right) {
        return std::nullopt;
    }
    SymExpr m_part = negated ? clone_expr(*quotient->left)
                             : sym_simplify(sym_neg(clone_expr(*quotient->left)));
    SymExpr n_part = clone_expr(*quotient->right);
    const SymExpr relation = sym_dsolve_exact(m_part, n_part, indep_var, dep_var);
    if (relation.op == SymOp::Deriv) {
        return std::nullopt;
    }
    return ode_solve_implicit_for_dep(relation, dep_var);
}

// --- stage 6: homogeneous of degree zero -------------------------------------

struct OdePowerMatch {
    double coef = 1.0;
    double exponent = 0.0;
};

std::optional<OdePowerMatch> ode_match_pure_power(const SymExpr& expr, const std::string& var) {
    const SymExpr* core = &expr;
    double coef = 1.0;
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const) {
            coef = expr.left->value;
            core = expr.right.get();
        } else if (expr.right->op == SymOp::Const) {
            coef = expr.right->value;
            core = expr.left.get();
        }
    } else if (expr.op == SymOp::Div && expr.left && expr.right &&
               expr.right->op == SymOp::Const && expr.right->value != 0.0) {
        coef = 1.0 / expr.right->value;
        core = expr.left.get();
    }
    if (coef == 0.0 || core->op != SymOp::Pow || !core->left || !core->right) {
        return std::nullopt;
    }
    if (!is_bare_var(*core->left, var) || core->right->op != SymOp::Const) {
        return std::nullopt;
    }
    const double exponent = core->right->value;
    if (exponent == 0.0) {
        return std::nullopt;
    }
    return OdePowerMatch{coef, exponent};
}

std::optional<double> ode_match_pure_log(const SymExpr& expr, const std::string& var) {
    const SymExpr* core = &expr;
    double coef = 1.0;
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const) {
            coef = expr.left->value;
            core = expr.right.get();
        } else if (expr.right->op == SymOp::Const) {
            coef = expr.right->value;
            core = expr.left.get();
        }
    }
    if (coef == 0.0 || core->op != SymOp::Log || !core->left || !is_bare_var(*core->left, var)) {
        return std::nullopt;
    }
    return coef;
}

std::optional<SymExpr> ode_stage_homogeneous(const SymExpr& rhs, const std::string& indep_var,
                                             const std::string& dep_var) {
    if (!contains_var_name(rhs, indep_var) || !contains_var_name(rhs, dep_var)) {
        return std::nullopt;
    }
    if (!ode_is_homogeneous_degree0(rhs, indep_var, dep_var)) {
        return std::nullopt;
    }
    const std::string v_name = contains_var_name(rhs, "v") ? "v__ode" : "v";
    const SymExpr reduced = sym_simplify(sym_substitute(
        sym_substitute(rhs, dep_var, sym_var(v_name)), indep_var, sym_const(1.0)));
    const auto g = ode_rational_of(reduced, v_name);
    if (!g) {
        return std::nullopt;
    }
    // x v' = G(v) - v, so dv / (G(v) - v) = dx / x with G = num/den.
    const std::vector<double> delta =
        ode_pv_add(g->num, ode_pv_mul(std::vector<double>{0.0, 1.0}, g->den), true);
    if (ode_pv_is_zero(delta)) {
        return sym_simplify(sym_mul(sym_var("C"), sym_var(indep_var)));
    }
    const auto potential = ode_integrate_rational(g->den, delta, v_name);
    if (!potential) {
        return std::nullopt;
    }
    // H(v) = log(x) + C; invert H.
    const SymExpr target = sym_add(sym_log(sym_var(indep_var)), sym_var("C"));
    SymExpr v_solution;
    if (const auto linear = ode_poly_numeric(*potential, v_name);
        linear && linear->size() == 2 && (*linear)[1] != 0.0) {
        v_solution = sym_simplify(sym_div(sym_sub(clone_expr(target), sym_const((*linear)[0])),
                                          sym_const((*linear)[1])));
    } else if (const auto power = ode_match_pure_power(*potential, v_name)) {
        v_solution = sym_simplify(sym_pow(ode_scale(clone_expr(target), 1.0 / power->coef),
                                          sym_const(1.0 / power->exponent)));
    } else if (const auto log_coef = ode_match_pure_log(*potential, v_name)) {
        v_solution = ode_exp_of(ode_scale(clone_expr(target), 1.0 / *log_coef));
    } else {
        return std::nullopt;
    }
    return sym_simplify(sym_mul(sym_var(indep_var), std::move(v_solution)));
}

// --- second-order constant-coefficient helpers ------------------------------

struct OdeTerm {
    double sign = 1.0;
    SymExpr expr;
};

void ode_split_terms(const SymExpr& expr, double sign, std::vector<OdeTerm>& out) {
    if (expr.op == SymOp::Add && expr.left && expr.right) {
        ode_split_terms(*expr.left, sign, out);
        ode_split_terms(*expr.right, sign, out);
        return;
    }
    if (expr.op == SymOp::Sub && expr.left && expr.right) {
        ode_split_terms(*expr.left, sign, out);
        ode_split_terms(*expr.right, -sign, out);
        return;
    }
    if (expr.op == SymOp::Neg && expr.left) {
        ode_split_terms(*expr.left, -sign, out);
        return;
    }
    out.push_back(OdeTerm{sign, sym_simplify(clone_expr(expr))});
}

// Peels a leading constant factor; `core` is left pointing into `expr` (a borrowed
// observer, never stored).
double ode_peel_const_factor(const SymExpr& expr, const SymExpr*& core) {
    core = &expr;
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (expr.left->op == SymOp::Const) {
            core = expr.right.get();
            return expr.left->value;
        }
        if (expr.right->op == SymOp::Const) {
            core = expr.left.get();
            return expr.right->value;
        }
    }
    return 1.0;
}

SymExpr ode_exp_basis(double rate, const std::string& var) {
    return sym_simplify(sym_exp(sym_mul(sym_const(rate), sym_var(var))));
}

// --- initial-value helpers ---------------------------------------------------

bool ode_ivp_accept(double residual, double y0) {
    return std::isfinite(residual) && std::abs(residual) <= 1e-9 * std::max(1.0, std::abs(y0));
}

} // namespace


SymExpr sym_dsolve_ode(const SymExpr& rhs, const std::string& indep_var,
                       const std::string& dep_var) {
    // Stage 1: the frozen separable table. Returned unverified and unmodified so that
    // sym_dsolve_ode is expression-identical to sym_dsolve wherever sym_dsolve succeeds.
    if (auto separable = ode_try_legacy_separable(rhs, indep_var, dep_var)) {
        return std::move(*separable);
    }

    // Stage 2: quadrature with the ODE-private integrator (strictly stronger than the
    // frozen sym_integrate the legacy table uses).
    if (!contains_var_name(rhs, dep_var)) {
        if (auto integral = ode_integrate(rhs, indep_var)) {
            SymExpr candidate = sym_simplify(sym_add(std::move(*integral), sym_var("C")));
            if (ode_verify_solution(rhs, candidate, indep_var, dep_var)) {
                return candidate;
            }
        }
        return sym_dsolve_unsupported(rhs, indep_var);
    }

    // Stages 3 and 4 share one decomposition of the rhs into powers of the dependent
    // variable; the same map detects the linear and the Bernoulli form.
    std::map<double, SymExpr> powers;
    if (ode_collect_powers(rhs, dep_var, false, powers)) {
        if (auto linear = ode_stage_linear(powers, indep_var)) {
            if (ode_verify_solution(rhs, *linear, indep_var, dep_var)) {
                return std::move(*linear);
            }
        }
        if (auto bernoulli = ode_stage_bernoulli(powers, indep_var)) {
            if (ode_verify_solution(rhs, *bernoulli, indep_var, dep_var)) {
                return std::move(*bernoulli);
            }
        }
    }

    if (auto exact = ode_stage_exact_explicit(rhs, indep_var, dep_var)) {
        if (ode_verify_solution(rhs, *exact, indep_var, dep_var)) {
            return std::move(*exact);
        }
    }

    if (auto homogeneous = ode_stage_homogeneous(rhs, indep_var, dep_var)) {
        if (ode_verify_solution(rhs, *homogeneous, indep_var, dep_var)) {
            return std::move(*homogeneous);
        }
    }

    return sym_dsolve_unsupported(rhs, indep_var);
}

SymExpr sym_dsolve_exact(const SymExpr& m, const SymExpr& n, const std::string& indep_var,
                         const std::string& dep_var) {
    const SymExpr dm_dy = sym_simplify(sym_diff(clone_expr(m), dep_var));
    const SymExpr dn_dx = sym_simplify(sym_diff(clone_expr(n), indep_var));
    if (!ode_same_function(dm_dy, dn_dx, indep_var, dep_var)) {
        return sym_dsolve_unsupported(m, indep_var);
    }

    auto partial = ode_integrate(m, indep_var);
    if (!partial) {
        return sym_dsolve_unsupported(m, indep_var);
    }
    const SymExpr fx = std::move(*partial);

    // n - d(fx)/dy must be a function of the dependent variable alone. sym_simplify cannot
    // cancel A - A for compound A, so test it numerically rather than structurally.
    const SymExpr remainder =
        sym_simplify(sym_sub(clone_expr(n), sym_simplify(sym_diff(clone_expr(fx), dep_var))));
    SymExpr correction = sym_const(0.0);
    if (!ode_numeric_zero(remainder, indep_var, dep_var)) {
        static constexpr double kProbes[3] = {1.0, 1.3, 2.7};
        bool resolved = false;
        for (const double probe : kProbes) {
            const SymExpr frozen =
                sym_simplify(sym_substitute(remainder, indep_var, sym_const(probe)));
            if (!ode_numeric_zero(sym_simplify(sym_sub(clone_expr(frozen), clone_expr(remainder))),
                                  indep_var, dep_var)) {
                continue;
            }
            auto integrated = ode_integrate(frozen, dep_var);
            if (!integrated) {
                continue;
            }
            correction = std::move(*integrated);
            resolved = true;
            break;
        }
        if (!resolved) {
            return sym_dsolve_unsupported(m, indep_var);
        }
    }

    const SymExpr potential = sym_simplify(sym_add(clone_expr(fx), std::move(correction)));
    if (!ode_same_function(sym_simplify(sym_diff(clone_expr(potential), indep_var)), m, indep_var,
                           dep_var) ||
        !ode_same_function(sym_simplify(sym_diff(clone_expr(potential), dep_var)), n, indep_var,
                           dep_var)) {
        return sym_dsolve_unsupported(m, indep_var);
    }
    return sym_simplify(sym_sub(clone_expr(potential), sym_var("C")));
}

SymExpr sym_dsolve_linear2(double a, double b, double c, const SymExpr& forcing,
                           const std::string& indep_var) {
    const double magnitude = std::max(1.0, std::abs(a) + std::abs(b) + std::abs(c));
    const bool a_zero = std::abs(a) <= kOdeZeroTol * magnitude;
    if (a_zero) {
        return sym_dsolve_unsupported(forcing, indep_var);
    }
    const bool b_zero = std::abs(b) <= kOdeZeroTol * magnitude;
    const bool c_zero = std::abs(c) <= kOdeZeroTol * magnitude;

    // Homogeneous part from the characteristic roots of a r^2 + b r + c.
    const double discriminant = b * b - 4.0 * a * c;
    const double discriminant_scale = std::max(1.0, std::abs(b * b) + std::abs(4.0 * a * c));
    SymExpr homogeneous;
    if (std::abs(discriminant) <= kOdeZeroTol * discriminant_scale) {
        const double root = -b / (2.0 * a);
        homogeneous = sym_simplify(
            sym_mul(sym_add(sym_var("C1"), sym_mul(sym_var("C2"), sym_var(indep_var))),
                    ode_exp_basis(root, indep_var)));
    } else if (discriminant > 0.0) {
        const double spread = std::sqrt(discriminant);
        const double r1 = (-b + spread) / (2.0 * a);
        const double r2 = (-b - spread) / (2.0 * a);
        homogeneous = sym_simplify(
            sym_add(sym_mul(sym_var("C1"), ode_exp_basis(r1, indep_var)),
                    sym_mul(sym_var("C2"), ode_exp_basis(r2, indep_var))));
    } else {
        const double alpha = -b / (2.0 * a);
        const double beta = std::sqrt(-discriminant) / (2.0 * std::abs(a));
        SymExpr oscillation = sym_add(
            sym_mul(sym_var("C1"), sym_cos(sym_simplify(
                                       sym_mul(sym_const(beta), sym_var(indep_var))))),
            sym_mul(sym_var("C2"), sym_sin(sym_simplify(
                                       sym_mul(sym_const(beta), sym_var(indep_var))))));
        homogeneous = sym_simplify(
            sym_mul(ode_exp_basis(alpha, indep_var), std::move(oscillation)));
    }

    const SymExpr folded_forcing = sym_simplify(clone_expr(forcing));
    if (folded_forcing.op == SymOp::Const && folded_forcing.value == 0.0) {
        return homogeneous;
    }

    // Particular solution by undetermined coefficients, term by term.
    std::vector<OdeTerm> terms;
    ode_split_terms(folded_forcing, 1.0, terms);
    std::vector<double> poly_forcing;
    SymExpr particular = sym_const(0.0);
    bool has_particular = false;

    for (const OdeTerm& term : terms) {
        const SymExpr* core = nullptr;
        const double amplitude = term.sign * ode_peel_const_factor(term.expr, core);

        if (core->op == SymOp::Exp && core->left) {
            const auto arg = ode_linear_arg(*core->left, indep_var);
            if (!arg) {
                return sym_dsolve_unsupported(forcing, indep_var);
            }
            const double gamma = arg->first;
            const double amp = amplitude * std::exp(arg->second);
            if (gamma == 0.0) {
                if (poly_forcing.empty()) {
                    poly_forcing.push_back(0.0);
                }
                poly_forcing[0] += amp;
                continue;
            }
            const double chi = a * gamma * gamma + b * gamma + c;
            const double chi_scale = std::max(
                1.0, std::abs(a * gamma * gamma) + std::abs(b * gamma) + std::abs(c));
            const double chi_prime = 2.0 * a * gamma + b;
            const double chi_prime_scale = std::max(1.0, std::abs(2.0 * a * gamma) + std::abs(b));
            SymExpr shape;
            double coefficient = 0.0;
            if (std::abs(chi) > kOdeZeroTol * chi_scale) {
                shape = ode_exp_basis(gamma, indep_var);
                coefficient = amp / chi;
            } else if (std::abs(chi_prime) > kOdeZeroTol * chi_prime_scale) {
                shape = sym_mul(sym_var(indep_var), ode_exp_basis(gamma, indep_var));
                coefficient = amp / chi_prime;
            } else {
                shape = sym_mul(sym_pow(sym_var(indep_var), sym_const(2.0)),
                                ode_exp_basis(gamma, indep_var));
                coefficient = amp / (2.0 * a);
            }
            particular = has_particular
                             ? sym_simplify(sym_add(std::move(particular),
                                                    ode_scale(std::move(shape), coefficient)))
                             : ode_scale(std::move(shape), coefficient);
            has_particular = true;
            continue;
        }

        if ((core->op == SymOp::Sin || core->op == SymOp::Cos) && core->left) {
            const auto arg = ode_linear_arg(*core->left, indep_var);
            if (!arg || arg->first == 0.0) {
                return sym_dsolve_unsupported(forcing, indep_var);
            }
            const double omega = arg->first;
            const double phase = arg->second;
            const double k1 = core->op == SymOp::Cos ? amplitude * std::cos(phase)
                                                     : amplitude * std::sin(phase);
            const double k2 = core->op == SymOp::Cos ? -amplitude * std::sin(phase)
                                                     : amplitude * std::cos(phase);
            const double real_part = c - a * omega * omega;
            const double imag_part = b * omega;
            const double real_scale = std::max(1.0, std::abs(c) + std::abs(a * omega * omega));
            const double imag_scale = std::max(1.0, std::abs(b * omega));
            SymExpr cosine =
                sym_cos(sym_simplify(sym_mul(sym_const(omega), sym_var(indep_var))));
            SymExpr sine = sym_sin(sym_simplify(sym_mul(sym_const(omega), sym_var(indep_var))));
            double p_coef = 0.0;
            double q_coef = 0.0;
            const bool resonant = std::abs(real_part) <= kOdeZeroTol * real_scale &&
                                  std::abs(imag_part) <= kOdeZeroTol * imag_scale;
            if (resonant) {
                q_coef = k1 / (2.0 * a * omega);
                p_coef = -k2 / (2.0 * a * omega);
            } else {
                const double delta = real_part * real_part + imag_part * imag_part;
                p_coef = (real_part * k1 - imag_part * k2) / delta;
                q_coef = (imag_part * k1 + real_part * k2) / delta;
            }
            SymExpr combination = sym_const(0.0);
            bool first = true;
            if (p_coef != 0.0) {
                combination = ode_scale(std::move(cosine), p_coef);
                first = false;
            }
            if (q_coef != 0.0) {
                SymExpr sine_term = ode_scale(std::move(sine), q_coef);
                combination = first ? std::move(sine_term)
                                    : sym_simplify(sym_add(std::move(combination),
                                                           std::move(sine_term)));
                first = false;
            }
            if (first) {
                continue;
            }
            SymExpr shape = resonant
                                ? sym_simplify(sym_mul(sym_var(indep_var), std::move(combination)))
                                : std::move(combination);
            particular = has_particular
                             ? sym_simplify(sym_add(std::move(particular), std::move(shape)))
                             : std::move(shape);
            has_particular = true;
            continue;
        }

        const auto polynomial = ode_poly_numeric(term.expr, indep_var);
        if (!polynomial) {
            return sym_dsolve_unsupported(forcing, indep_var);
        }
        if (poly_forcing.size() < polynomial->size()) {
            poly_forcing.resize(polynomial->size(), 0.0);
        }
        for (std::size_t i = 0; i < polynomial->size(); ++i) {
            poly_forcing[i] += term.sign * (*polynomial)[i];
        }
    }

    if (!poly_forcing.empty()) {
        poly_forcing = ode_pv_trim(std::move(poly_forcing));
        if (!ode_pv_is_zero(poly_forcing)) {
            const std::size_t degree = poly_forcing.size() - 1;
            const std::size_t lift = c_zero ? (b_zero ? 2u : 1u) : 0u;
            std::vector<double> ansatz(degree + lift + 3, 0.0);
            for (std::size_t j = degree + 1; j-- > 0;) {
                const double jd = static_cast<double>(j);
                const double f = poly_forcing[j];
                if (lift == 0) {
                    ansatz[j] = (f - b * (jd + 1.0) * ansatz[j + 1] -
                                 a * (jd + 2.0) * (jd + 1.0) * ansatz[j + 2]) /
                                c;
                } else if (lift == 1) {
                    ansatz[j + 1] =
                        (f - a * (jd + 2.0) * (jd + 1.0) * ansatz[j + 2]) / (b * (jd + 1.0));
                } else {
                    ansatz[j + 2] = f / (a * (jd + 2.0) * (jd + 1.0));
                }
            }
            SymExpr poly_part = ode_build_poly(ode_pv_to_coeffs(ansatz), indep_var);
            particular = has_particular
                             ? sym_simplify(sym_add(std::move(particular), std::move(poly_part)))
                             : std::move(poly_part);
            has_particular = true;
        }
    }

    if (!has_particular) {
        return homogeneous;
    }
    return sym_simplify(sym_add(sym_simplify(std::move(particular)), std::move(homogeneous)));
}

SymExpr sym_dsolve_ivp(const SymExpr& general_solution, const std::string& indep_var, double x0,
                       double y0) {
    if (general_solution.op == SymOp::Deriv) {
        return clone_expr(general_solution);
    }
    if (!contains_var_name(general_solution, "C")) {
        return sym_dsolve_unsupported(general_solution, indep_var);
    }

    // Affine shortcut: two probes pin C exactly whenever the solution is affine in C.
    const double at_zero = sym_eval(general_solution, {{indep_var, x0}, {"C", 0.0}});
    const double at_one = sym_eval(general_solution, {{indep_var, x0}, {"C", 1.0}});
    const double slope = at_one - at_zero;
    if (std::isfinite(at_zero) && std::isfinite(at_one) && std::abs(slope) > 1e-14) {
        const double candidate = (y0 - at_zero) / slope;
        const double residual =
            sym_eval(general_solution, {{indep_var, x0}, {"C", candidate}}) - y0;
        if (ode_ivp_accept(residual, y0)) {
            return sym_simplify(
                sym_substitute(general_solution, "C", sym_const(candidate)));
        }
    }

    // Otherwise bracket a sign change on a fixed ladder and bisect.
    static constexpr double kLadder[] = {0.0,  0.125, -0.125, 0.25, -0.25, 0.5,   -0.5,
                                         1.0,  -1.0,  2.0,    -2.0, 4.0,   -4.0,  8.0,
                                         -8.0, 16.0,  -16.0,  32.0, -32.0, 64.0,  -64.0,
                                         128.0, -128.0};
    double low = 0.0;
    double high = 0.0;
    double low_value = 0.0;
    double previous_c = 0.0;
    double previous_value = 0.0;
    bool have_previous = false;
    bool have_bracket = false;
    for (const double trial : kLadder) {
        const double value = sym_eval(general_solution, {{indep_var, x0}, {"C", trial}}) - y0;
        if (!std::isfinite(value)) {
            have_previous = false;
            continue;
        }
        if (ode_ivp_accept(value, y0)) {
            return sym_simplify(sym_substitute(general_solution, "C", sym_const(trial)));
        }
        if (have_previous && ((previous_value < 0.0 && value > 0.0) ||
                              (previous_value > 0.0 && value < 0.0))) {
            low = previous_c;
            low_value = previous_value;
            high = trial;
            have_bracket = true;
            break;
        }
        previous_c = trial;
        previous_value = value;
        have_previous = true;
    }
    if (!have_bracket) {
        return sym_dsolve_unsupported(general_solution, indep_var);
    }
    for (int step = 0; step < 200; ++step) {
        const double middle = 0.5 * (low + high);
        const double value = sym_eval(general_solution, {{indep_var, x0}, {"C", middle}}) - y0;
        if (!std::isfinite(value)) {
            break;
        }
        if ((value < 0.0) == (low_value < 0.0)) {
            low = middle;
            low_value = value;
        } else {
            high = middle;
        }
    }
    const double solved = 0.5 * (low + high);
    const double residual = sym_eval(general_solution, {{indep_var, x0}, {"C", solved}}) - y0;
    if (!ode_ivp_accept(residual, y0)) {
        return sym_dsolve_unsupported(general_solution, indep_var);
    }
    return sym_simplify(sym_substitute(general_solution, "C", sym_const(solved)));
}

SymExpr sym_dsolve_ivp2(const SymExpr& general_solution, const std::string& indep_var, double x0,
                        double y0, double yp0) {
    if (general_solution.op == SymOp::Deriv) {
        return clone_expr(general_solution);
    }
    const SymExpr derivative = sym_simplify(sym_diff(clone_expr(general_solution), indep_var));

    const std::map<std::string, double> base{{indep_var, x0}, {"C1", 0.0}, {"C2", 0.0}};
    const std::map<std::string, double> first{{indep_var, x0}, {"C1", 1.0}, {"C2", 0.0}};
    const std::map<std::string, double> second{{indep_var, x0}, {"C1", 0.0}, {"C2", 1.0}};
    const double u0 = sym_eval(general_solution, base);
    const double u1 = sym_eval(general_solution, first) - u0;
    const double u2 = sym_eval(general_solution, second) - u0;
    const double v0 = sym_eval(derivative, base);
    const double v1 = sym_eval(derivative, first) - v0;
    const double v2 = sym_eval(derivative, second) - v0;
    const double determinant = u1 * v2 - u2 * v1;
    if (!std::isfinite(determinant) ||
        std::abs(determinant) <= 1e-12 * std::max(1.0, std::abs(u1 * v2) + std::abs(u2 * v1))) {
        return sym_dsolve_unsupported(general_solution, indep_var);
    }
    const double c1 = ((y0 - u0) * v2 - u2 * (yp0 - v0)) / determinant;
    const double c2 = (u1 * (yp0 - v0) - (y0 - u0) * v1) / determinant;

    SymExpr pinned = sym_simplify(sym_substitute(
        sym_substitute(general_solution, "C1", sym_const(c1)), "C2", sym_const(c2)));
    const SymExpr pinned_derivative = sym_simplify(sym_diff(clone_expr(pinned), indep_var));
    const double value_residual = sym_eval(pinned, {{indep_var, x0}}) - y0;
    const double slope_residual = sym_eval(pinned_derivative, {{indep_var, x0}}) - yp0;
    if (!ode_ivp_accept(value_residual, y0) || !ode_ivp_accept(slope_residual, yp0)) {
        return sym_dsolve_unsupported(general_solution, indep_var);
    }
    return pinned;
}
} // namespace ms
