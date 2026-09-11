// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/symbolic/symbolic.hpp"

#include "ms/core/format.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <functional>
#include <optional>
#include <algorithm>
#include <vector>
#include <numbers>
#include <string>
#include <unordered_map>

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
bool try_eval_const(const SymExpr& expr, double& out);
std::optional<double> match_quadratic_coefficient(const SymExpr& expr, const std::string& var);
bool match_exp_neg_at(const SymExpr& expr, const std::string& t_var, double& a);

bool is_const_zero(const SymExpr& expr) {
    return expr.op == SymOp::Const && expr.value == 0.0;
}

// Structural comparison used by sym_is_unsupported. Recursive rather than a hash
// so it stays exact: two expressions are equal only if every node matches.
bool sym_equal_impl(const SymExpr& a, const SymExpr& b) {
    if (a.op != b.op || a.name != b.name) {
        return false;
    }
    if (a.op == SymOp::Const && a.value != b.value) {
        return false;
    }
    if ((a.left == nullptr) != (b.left == nullptr) ||
        (a.right == nullptr) != (b.right == nullptr)) {
        return false;
    }
    if (a.left && !sym_equal_impl(*a.left, *b.left)) {
        return false;
    }
    return !a.right || sym_equal_impl(*a.right, *b.right);
}

SymExpr sym_integrate_unsupported(const SymExpr& expr, const std::string& var) {
    return sym_deriv(clone_expr(expr), var);
}

// True when `expr` carries the "no closed form" sentinel for `var` anywhere in its
// tree, not merely at the root. SymOp::Deriv is constructed in this translation
// unit only, and only as that sentinel -- no parser and no public constructor can
// produce one -- so the scan is exact rather than a heuristic.
bool contains_unsupported_sentinel(const SymExpr& expr, const std::string& var) {
    if (expr.op == SymOp::Deriv && expr.name == var) {
        return true;
    }
    if (expr.left && contains_unsupported_sentinel(*expr.left, var)) {
        return true;
    }
    return expr.right && contains_unsupported_sentinel(*expr.right, var);
}

// True when `var` appears anywhere in `expr`. Used to check that the other factor of
// a product really is a constant with respect to the transform variable.
bool expression_uses_var(const SymExpr& expr, const std::string& var) {
    if (expr.op == SymOp::Var && expr.name == var) {
        return true;
    }
    if (expr.left && expression_uses_var(*expr.left, var)) {
        return true;
    }
    return expr.right && expression_uses_var(*expr.right, var);
}

// Guard for the linearity rules. Each of them recurses into the operands and
// reassembles whatever comes back, so a single operand that declines yields a tree
// that is part answer and part sentinel:
//
//     sym_laplace("t + t*exp(2*t)")  ->  (1/s^2) + d/dt(t*exp(2*t))
//
// which the root-only unsupported check reads as a success. Wrapping the assembled
// result here turns any operand's refusal into a refusal of the whole expression,
// so the sentinel arrives at the caller as one unambiguous marker naming what was
// actually asked for.
SymExpr decline_if_unsupported(SymExpr built, const SymExpr& whole, const std::string& var) {
    if (contains_unsupported_sentinel(built, var)) {
        return sym_deriv(clone_expr(whole), var);
    }
    return built;
}

constexpr int kMaxLaplacePower = 8;

bool try_get_const_value(const SymExpr& expr, double& out) {
    if (expr.op == SymOp::Const) {
        out = expr.value;
        return true;
    }
    // The parser builds a negative literal as Neg(Const(3)), never Const(-3):
    //
    //     exp(-3*t)  ->  Exp(Mul(Neg(Const(3)), Var(t)))
    //     exp( 2*t)  ->  Exp(Mul(    Const(2),  Var(t)))
    //
    // Every transform table below matches constants through this function or an
    // `op == SymOp::Const` test, so before this case existed no table entry could
    // match a negative coefficient at all. L{exp(-3t)} = 1/(s+3) -- the decaying
    // exponential, which is most of the practical use of a Laplace transform --
    // returned the unsupported sentinel, while L{exp(3t)} worked.
    if (expr.op == SymOp::Neg && expr.left) {
        double inner = 0.0;
        if (try_get_const_value(*expr.left, inner)) {
            out = -inner;
            return true;
        }
    }
    // Constant arithmetic. The parser folds nothing, so 2*3 arrives as a Mul of two
    // Const nodes and 1/2 as a Div of two; every caller here is asking about the
    // value of a subexpression, not its shape, so a fully constant subtree is a
    // constant however it was written. Folding cannot misfire on a subtree that
    // mentions a variable, because such a subtree never reaches the arithmetic below.
    double lhs = 0.0;
    double rhs = 0.0;
    if (expr.left && expr.right && try_get_const_value(*expr.left, lhs) &&
        try_get_const_value(*expr.right, rhs)) {
        double folded = 0.0;
        switch (expr.op) {
        case SymOp::Add:
            folded = lhs + rhs;
            break;
        case SymOp::Sub:
            folded = lhs - rhs;
            break;
        case SymOp::Mul:
            folded = lhs * rhs;
            break;
        case SymOp::Div:
            if (rhs == 0.0) {
                return false;
            }
            folded = lhs / rhs;
            break;
        case SymOp::Pow:
            folded = std::pow(lhs, rhs);
            break;
        default:
            return false;
        }
        // A fold that is not finite -- a real root of a negative number, an overflow
        // -- is not a constant a table entry can use.
        if (!std::isfinite(folded)) {
            return false;
        }
        out = folded;
        return true;
    }
    return false;
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
        // try_get_const_value rather than an op test, so a negated literal counts.
        if (is_bare_var(*expr.right, var) && try_get_const_value(*expr.left, scale)) {
            return true;
        }
        if (is_bare_var(*expr.left, var) && try_get_const_value(*expr.right, scale)) {
            return true;
        }
    }
    // A bare negated variable: exp(-t) is the a = -1 case, and parses as
    // Neg(Var(t)) with no Mul node at all.
    if (expr.op == SymOp::Neg && expr.left && is_bare_var(*expr.left, var)) {
        scale = -1.0;
        return true;
    }
    return false;
}

// a*var + b, in every spelling the parser produces, with a != 0 required by the
// callers rather than here. This is the argument shape that every linear-substitution
// table entry needs -- integral f(a*x+b) dx = F(a*x+b)/a -- and the one the table
// used to reject: only a *bare* variable argument was matched, so sin(x) integrated
// and sin(2*x) did not, though the second is the first chain-rule row of the table.
bool match_affine_in_var(const SymExpr& expr, const std::string& var, double& a, double& b) {
    if (match_scaled_var(expr, var, a)) {
        b = 0.0;
        return true;
    }
    if (expr.op == SymOp::Add && expr.left && expr.right) {
        if (match_scaled_var(*expr.left, var, a) && try_get_const_value(*expr.right, b)) {
            return true;
        }
        return match_scaled_var(*expr.right, var, a) && try_get_const_value(*expr.left, b);
    }
    if (expr.op == SymOp::Sub && expr.left && expr.right) {
        if (match_scaled_var(*expr.left, var, a) && try_get_const_value(*expr.right, b)) {
            b = -b;
            return true;
        }
        double lead = 0.0;
        if (try_get_const_value(*expr.left, lead) && match_scaled_var(*expr.right, var, a)) {
            a = -a;
            b = lead;
            return true;
        }
        return false;
    }
    return false;
}

// Divide an antiderivative by the inner linear coefficient, leaving a == 1 alone so
// that the bare-argument entries keep printing exactly as they always have
// (-cos(x), not (-cos(x)) / 1.000000).
SymExpr scale_antiderivative(SymExpr anti, double a) {
    if (a == 1.0) {
        return anti;
    }
    return sym_div(std::move(anti), sym_const(a));
}

SymExpr build_s2_plus_a2(const std::string& s, double a) {
    return sym_add(sym_pow(sym_var(s), sym_const(2.0)), sym_pow(sym_const(a), sym_const(2.0)));
}

// s - a, spelled as s + |a| when a is negative so that the first shifting theorem
// prints 1/(s + 3) rather than 1/(s - -3.000000).
SymExpr build_s_minus_a(const std::string& s, double a) {
    if (a < 0.0) {
        return sym_add(sym_var(s), sym_const(-a));
    }
    return sym_sub(sym_var(s), sym_const(a));
}

// exp(a*var), the factor the first shifting theorem keys on.
bool match_exp_scaled_var(const SymExpr& expr, const std::string& var, double& a) {
    return expr.op == SymOp::Exp && expr.left && match_scaled_var(*expr.left, var, a);
}

// s - a with unit coefficient, which is what the first shifting theorem leaves in a
// denominator. A bare s is the a = 0 case, so every rule keyed on this matcher covers
// the unshifted row of the table and its shifted companion at the same time.
bool match_shifted_var(const SymExpr& expr, const std::string& s, double& a) {
    if (is_bare_var(expr, s)) {
        a = 0.0;
        return true;
    }
    double coeff = 0.0;
    double constant = 0.0;
    if (match_affine_in_var(expr, s, coeff, constant) && coeff == 1.0) {
        a = -constant;
        return true;
    }
    return false;
}

bool is_square(const SymExpr& expr) {
    return expr.op == SymOp::Pow && expr.left && expr.right && expr.right->op == SymOp::Const &&
           expr.right->value == 2.0;
}

// A positive constant written either as b^2 or as the number b^2 itself.
bool match_positive_square(const SymExpr& expr, double& b) {
    double value = 0.0;
    if (is_square(expr) && try_get_const_value(*expr.left, value) && value > 0.0) {
        b = value;
        return true;
    }
    if (try_get_const_value(expr, value) && value > 0.0) {
        b = std::sqrt(value);
        return true;
    }
    return false;
}

// (s - a)^2 + b^2, b > 0.
bool match_shifted_quadratic(const SymExpr& expr, const std::string& s, double& a, double& b) {
    if (expr.op != SymOp::Add || !expr.left || !expr.right) {
        return false;
    }
    if (is_square(*expr.left) && match_shifted_var(*expr.left->left, s, a) &&
        match_positive_square(*expr.right, b)) {
        return true;
    }
    return is_square(*expr.right) && match_shifted_var(*expr.right->left, s, a) &&
           match_positive_square(*expr.left, b);
}

// (s - a)^n for a small n >= 1, with a bare s or s - a as the n = 1 case.
bool match_shifted_power(const SymExpr& expr, const std::string& s, double& a, int& n) {
    double power = 0.0;
    if (expr.op == SymOp::Pow && expr.left && expr.right &&
        try_get_const_value(*expr.right, power) && match_shifted_var(*expr.left, s, a)) {
        int as_int = 0;
        if (is_small_nonneg_int(power, as_int) && as_int >= 1) {
            n = as_int;
            return true;
        }
        return false;
    }
    if (match_shifted_var(expr, s, a)) {
        n = 1;
        return true;
    }
    return false;
}

// s^2 - a^2, a > 0 -- the hyperbolic row, printed on the same table page as the
// sine row and previously absent.
bool match_difference_of_squares(const SymExpr& expr, const std::string& s, double& a) {
    double shift = 0.0;
    return expr.op == SymOp::Sub && expr.left && expr.right && is_square(*expr.left) &&
           match_shifted_var(*expr.left->left, s, shift) && shift == 0.0 &&
           match_positive_square(*expr.right, a);
}

// c * f(t), dropping a unit factor.
SymExpr attach_scale(double c, SymExpr f) {
    if (c == 1.0) {
        return f;
    }
    return sym_mul(sym_const(c), std::move(f));
}

// f(t) * exp(a*t), dropping the exponential when a == 0 and dropping f when it is
// the constant 1, so the unshifted rows keep printing as they always have.
SymExpr attach_exp(SymExpr f, double a, const std::string& t) {
    if (a == 0.0) {
        return f;
    }
    SymExpr factor = sym_exp(sym_mul(sym_const(a), sym_var(t)));
    if (f.op == SymOp::Const && f.value == 1.0) {
        return factor;
    }
    return sym_mul(std::move(f), std::move(factor));
}

// c * t^n, collapsing to c alone when n == 0.
SymExpr build_monomial(double c, int n, const std::string& t) {
    if (n == 0) {
        return sym_const(c);
    }
    SymExpr power =
        n == 1 ? sym_var(t) : sym_pow(sym_var(t), sym_const(static_cast<double>(n)));
    return attach_scale(c, std::move(power));
}

// var or var^n for a small positive integer n -- the factor the frequency
// differentiation rule L{t^n g(t)} = (-1)^n G^(n)(s) keys on.
bool match_var_power_small_int(const SymExpr& expr, const std::string& var, int& n) {
    if (is_bare_var(expr, var)) {
        n = 1;
        return true;
    }
    double power = 0.0;
    if (expr.op == SymOp::Pow && expr.left && expr.right && is_bare_var(*expr.left, var) &&
        try_get_const_value(*expr.right, power) && is_small_nonneg_int(power, n)) {
        return n >= 1;
    }
    return false;
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


// The multiple of the canonical row that `numerator` represents, or nullopt if the
// row does not apply at all. It used to be a yes/no test against the canonical scale
// to within 1e-9, which refused two different things: a row at any other amplitude
// (1/((k^2+4)^1.5) is the same row as 2/((k^2+4)^1.5), halved), and the module's own
// printed output, since sym_to_string prints six decimals and 4.513517 differs from
// the exact scale by about 7e-8.
std::optional<double> hankel_scale_ratio(double numerator, int n, double a) {
    (void)n;
    const double expected = a;
    if (expected == 0.0) {
        return std::nullopt;
    }
    return numerator / expected;
}

SymExpr build_k2_plus_a2(const std::string& k, double a) {
    return build_s2_plus_a2(k, a);
}

bool match_one_over_sqrt_r2_plus_a2(const SymExpr& expr, const std::string& r_var, double& a,
                                   double& scale) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return false;
    }
    if (expr.right->op != SymOp::Sqrt || !expr.right->left) {
        return false;
    }
    return try_get_const_value(*expr.left, scale) && match_s2_plus_a2(*expr.right->left, r_var, a);
}

// c * exp(-a*var) / var, in either spelling of the negation. The old matcher required
// the exponent to be a Neg node, so exp(-(2*k))/k matched and exp(-2*k)/k -- where the
// minus binds to the coefficient and there is no Neg at all -- did not, though they
// are the same function. match_exp_neg_at already handles both.
bool match_scaled_exp_neg_over_var(const SymExpr& expr, const std::string& var, double& a,
                                   double& scale) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right || !is_bare_var(*expr.right, var)) {
        return false;
    }
    const SymExpr* decay = expr.left.get();
    scale = 1.0;
    if (decay->op == SymOp::Mul && decay->left && decay->right) {
        if (try_get_const_value(*decay->left, scale)) {
            decay = decay->right.get();
        } else if (try_get_const_value(*decay->right, scale)) {
            decay = decay->left.get();
        }
    }
    return match_exp_neg_at(*decay, var, a) && a > 0.0;
}

struct HankelKDomain {
    int power = 0;
    double rate = 0.0;
    double scale = 1.0;
};

std::optional<HankelKDomain> match_hankel_k_domain_rpow_exp(
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
    // Only the n = 0 row, a/(a^2+k^2)^(3/2), is invertible by matching a shape: for
    // n >= 1 the forward transform is a derivative of it with respect to a, and the
    // result is no longer a constant over a power of (a^2+k^2). The matcher used to
    // claim those too, and inverted them consistently with the wrong forward formula.
    const double exp_power = expr.right->right->value;
    if (exp_power != 1.5) {
        return std::nullopt;
    }
    const int n = 0;
    double numerator = 0.0;
    if (!try_get_const_value(*expr.left, numerator)) {
        return std::nullopt;
    }
    const auto ratio = hankel_scale_ratio(numerator, n, a);
    if (!ratio) {
        return std::nullopt;
    }
    return HankelKDomain{n, a, *ratio};
}

// H0[r^n exp(-a*r)](k) = (-1)^n d^n/da^n [ a / (a^2 + k^2)^(3/2) ].
//
// This used to be scale(n) * a / (k^2 + a^2)^((n+3)/2), which is right at n = 0 and
// wrong for every n above it: differentiating a/(a^2+k^2)^(3/2) with respect to a does
// not just raise the exponent and rescale, because a appears in the numerator too.
// H0[r*exp(-2r)] at k = 0.5 came back as 0.24988 where the defining Bessel integral
// gives 0.20813 -- the true value (2a^2 - k^2)/(a^2 + k^2)^(5/2). A wrong number, with
// no error, for the whole n >= 1 family.
//
// Differentiating symbolically rather than hand-deriving each row keeps every n right
// by construction. The differentiation variable is named after k so it cannot collide
// with the caller's spectral variable.
SymExpr hankel_forward_rpow_exp_neg(int n, double a, const std::string& k) {
    const std::string rate = k + "__hankel_rate";
    SymExpr transformed = sym_div(
        sym_var(rate),
        sym_pow(sym_add(sym_pow(sym_var(rate), sym_const(2.0)),
                        sym_pow(sym_var(k), sym_const(2.0))),
                sym_const(1.5)));
    for (int i = 0; i < n; ++i) {
        transformed = sym_simplify(sym_neg(sym_diff(std::move(transformed), rate)));
    }
    return sym_simplify(sym_substitute(transformed, rate, sym_const(a)));
}

SymExpr ihankel_inverse_rpow_exp_neg(int n, double a, const std::string& r) {
    SymExpr decay = sym_exp(sym_neg(sym_mul(sym_const(a), sym_var(r))));
    if (n == 0) {
        return decay;
    }
    return sym_mul(sym_pow(sym_var(r), sym_const(static_cast<double>(n))), std::move(decay));
}

// c / (a + t^n) with a > 0 and n a small positive integer, which is the whole
// rational family of the Mellin table in one shape. The matcher it replaces insisted
// on the numerator being exactly 1, the constant being exactly 1 and the power being
// exactly t, so 3/(1+t) (linearity), 1/(2+t) (the scaling rule) and 1/(1+t^2) (the
// n = 2 row, on the same table line as n = 1) were each refused.
//
//   M{c/(a + t^n)}(s) = (c/n) * a^(s/n - 1) * pi / sin(pi*s/n)
//
// which reduces to c*pi/sin(pi*s) at a = n = 1.
struct MellinRational {
    double coefficient = 1.0;
    double offset = 1.0;
    int power = 1;
};

std::optional<MellinRational> match_mellin_rational(const SymExpr& expr, const std::string& t_var) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return std::nullopt;
    }
    MellinRational matched;
    if (!try_get_const_value(*expr.left, matched.coefficient)) {
        return std::nullopt;
    }
    if (expr.right->op != SymOp::Add || !expr.right->left || !expr.right->right) {
        return std::nullopt;
    }
    auto match_power = [&](const SymExpr& term, int& power) {
        if (is_bare_var(term, t_var)) {
            power = 1;
            return true;
        }
        double exponent = 0.0;
        return term.op == SymOp::Pow && term.left && term.right && is_bare_var(*term.left, t_var) &&
               try_get_const_value(*term.right, exponent) && is_small_nonneg_int(exponent, power) &&
               power >= 1;
    };
    const SymExpr& left = *expr.right->left;
    const SymExpr& right = *expr.right->right;
    if (match_power(right, matched.power) && try_get_const_value(left, matched.offset)) {
        return matched.offset > 0.0 ? std::optional{matched} : std::nullopt;
    }
    if (match_power(left, matched.power) && try_get_const_value(right, matched.offset)) {
        return matched.offset > 0.0 ? std::optional{matched} : std::nullopt;
    }
    return std::nullopt;
}

// (1 + t)^(-m) for a small positive integer m. M{(1+t)^-m}(s) = Gamma(s)Gamma(m-s)/Gamma(m),
// which the reflection formula turns into an elementary rational multiple of
// pi/sin(pi*s) whenever m is an integer:
//
//   M{(1+t)^-m}(s) = pi/sin(pi*s) * product_{j=1}^{m-1} (j - s) / (m-1)!
// pi/sin(pi*s) * product_{j=1}^{m-1} (j - s) / (m-1)!  -- see match_one_plus_t_power.
SymExpr build_mellin_one_plus_t_power(int m, const std::string& s) {
    SymExpr reflection =
        sym_div(sym_const(std::numbers::pi),
                sym_sin(sym_mul(sym_const(std::numbers::pi), sym_var(s))));
    if (m == 1) {
        return reflection;
    }
    SymExpr product = sym_sub(sym_const(1.0), sym_var(s));
    for (int j = 2; j < m; ++j) {
        product = sym_mul(std::move(product),
                          sym_sub(sym_const(static_cast<double>(j)), sym_var(s)));
    }
    return sym_div(sym_mul(std::move(reflection), std::move(product)),
                   sym_const(factorial_int(m - 1)));
}

std::optional<int> match_one_plus_t_power(const SymExpr& expr, const std::string& t_var) {
    double exponent = 0.0;
    const SymExpr* base = nullptr;
    if (expr.op == SymOp::Div && expr.left && expr.right) {
        double numerator = 0.0;
        if (!try_get_const_value(*expr.left, numerator) || numerator != 1.0) {
            return std::nullopt;
        }
        if (expr.right->op != SymOp::Pow || !expr.right->left || !expr.right->right ||
            !try_get_const_value(*expr.right->right, exponent) || exponent <= 0.0) {
            return std::nullopt;
        }
        base = expr.right->left.get();
    } else if (expr.op == SymOp::Pow && expr.left && expr.right &&
               try_get_const_value(*expr.right, exponent) && exponent < 0.0) {
        exponent = -exponent;
        base = expr.left.get();
    } else {
        return std::nullopt;
    }
    int m = 0;
    if (!is_small_nonneg_int(exponent, m) || m < 1) {
        return std::nullopt;
    }
    if (!base || base->op != SymOp::Add || !base->left || !base->right) {
        return std::nullopt;
    }
    double one = 0.0;
    if ((try_get_const_value(*base->left, one) && one == 1.0 && is_bare_var(*base->right, t_var)) ||
        (try_get_const_value(*base->right, one) && one == 1.0 && is_bare_var(*base->left, t_var))) {
        return m;
    }
    return std::nullopt;
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

namespace {

/// One addend of a flattened sum, split into the numeric multiple and the thing it
/// multiplies: `3*x` is (3, x), `x` is (1, x), `-x` is (-1, x), `7` is (7, nothing).
///
/// The key is `sym_to_string` of the kernel, and it is a BUCKET key only -- two addends
/// that share it are then compared with `sym_equal_impl` before being added together.
/// The printer renders a constant with six decimals, so `sin(1.0000001*x)` and
/// `sin(1.0000002*x)` print identically; treating that text as equality would collapse
/// their difference to exactly zero. `test_symbolic_tables` asserts that expansion does
/// not do this, and caught the first version of this code doing it.
struct Addend {
    double coefficient = 1.0;
    std::optional<SymExpr> kernel; ///< absent for a pure constant
    std::string key;               ///< empty for a pure constant
};

Addend split_addend(const SymExpr& term, double sign);

void flatten_sum(const SymExpr& expr, double sign, std::vector<Addend>& out) {
    if (expr.op == SymOp::Add && expr.left && expr.right) {
        flatten_sum(*expr.left, sign, out);
        flatten_sum(*expr.right, sign, out);
        return;
    }
    if (expr.op == SymOp::Sub && expr.left && expr.right) {
        flatten_sum(*expr.left, sign, out);
        flatten_sum(*expr.right, -sign, out);
        return;
    }
    out.push_back(split_addend(expr, sign));
}

Addend split_addend(const SymExpr& term, double sign) {
    if (term.op == SymOp::Neg && term.left) {
        return split_addend(*term.left, -sign);
    }
    if (term.op == SymOp::Const) {
        return Addend{sign * term.value, std::nullopt, std::string()};
    }
    // Only a constant factor at the top of a product is peeled. `2*x` and `x*2` are the
    // shapes the parser and the differentiator produce; going deeper would mean
    // reassociating the product, which is expansion's job and not this one's.
    const SymExpr* body = &term;
    double factor = 1.0;
    if (term.op == SymOp::Mul && term.left && term.right) {
        if (term.left->op == SymOp::Const) {
            factor = term.left->value;
            body = term.right.get();
        } else if (term.right->op == SymOp::Const) {
            factor = term.right->value;
            body = term.left.get();
        }
    }
    return Addend{sign * factor, clone_expr(*body), sym_to_string(*body)};
}

SymExpr scaled_addend(const SymExpr& kernel, double coefficient) {
    if (coefficient == 1.0) {
        return clone_expr(kernel);
    }
    if (coefficient == -1.0) {
        return sym_neg(clone_expr(kernel));
    }
    return sym_mul(sym_const(coefficient), clone_expr(kernel));
}

/// `x + x` as `2*x`, and nothing else.
///
/// The Add and Sub cases of `sym_simplify` fold a constant into a constant and drop a
/// zero, and stop there, so a sum of like terms came back as written:
/// `sym_simplify("2*x + 3*x")` returned `((2.000000 * x) + (3.000000 * x))`, which is
/// the input with the spaces moved.
///
/// The narrow form is deliberate. Expansion already has a polynomial normal form, and
/// routing simplify through it would multiply products out -- `x*x` becoming
/// `(x ^ 2.000000)` -- in every one of simplify's callers, including the ODE solvers
/// that dispatch on the *op* of what it hands back. This flattens the sum, adds the
/// coefficients of terms that are the same term, and changes nothing else.
///
/// `std::nullopt` means nothing merged, and then the caller returns the expression it
/// already had rather than a rebuilt copy of it: a sum with no like terms in it comes
/// out exactly as it went in, which is what keeps this out of the way of the 180-odd
/// callers that are not asking for it.
std::optional<SymExpr> collect_sum_terms(const SymExpr& expr) {
    std::vector<Addend> addends;
    flatten_sum(expr, 1.0, addends);
    if (addends.size() < 2) {
        return std::nullopt;
    }

    std::vector<Addend> merged;
    std::unordered_map<std::string, std::vector<std::size_t>> buckets;
    bool collected = false;
    for (Addend& addend : addends) {
        if (!std::isfinite(addend.coefficient)) {
            // An infinity or a NaN among the coefficients is a value this has no
            // business adding up: 'inf*x - inf*x' is not 0 and saying so would be
            // inventing an answer.
            return std::nullopt;
        }
        std::vector<std::size_t>& bucket = buckets[addend.key];
        std::size_t target = merged.size();
        for (const std::size_t index : bucket) {
            const Addend& candidate = merged[index];
            const bool both_constant = !candidate.kernel && !addend.kernel;
            if (both_constant || (candidate.kernel && addend.kernel &&
                                  sym_equal_impl(*candidate.kernel, *addend.kernel))) {
                target = index;
                break;
            }
        }
        if (target == merged.size()) {
            bucket.push_back(merged.size());
            merged.push_back(std::move(addend));
            continue;
        }
        merged[target].coefficient += addend.coefficient;
        collected = true;
    }
    if (!collected) {
        return std::nullopt;
    }

    // First-appearance order, so a sum a user wrote comes back in the order they wrote
    // it. A canonical order would be defensible and would also rewrite every sum that
    // reaches here, which is a larger change than the one being made.
    std::optional<SymExpr> out;
    for (const Addend& addend : merged) {
        if (addend.coefficient == 0.0) {
            continue;
        }
        const bool negative = addend.coefficient < 0.0 && out.has_value();
        const double magnitude = negative ? -addend.coefficient : addend.coefficient;
        SymExpr piece = addend.kernel ? scaled_addend(*addend.kernel, magnitude)
                                      : sym_const(magnitude);
        if (!out) {
            out = std::move(piece);
        } else if (negative) {
            out = sym_sub(std::move(*out), std::move(piece));
        } else {
            out = sym_add(std::move(*out), std::move(piece));
        }
    }
    if (!out) {
        return sym_const(0.0);
    }
    return out;
}

} // namespace

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
        if (auto collected = collect_sum_terms(expr)) {
            return std::move(*collected);
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
        if (auto collected = collect_sum_terms(expr)) {
            return std::move(*collected);
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




// ---------------------------------------------------------------------------
// Canonical polynomial form.
//
// Expansion used to multiply out over the expression tree and never put like terms
// back together: (x+1)^3 came out as eight products rather than four terms, and
// (x+1)^8 as 256 rather than nine. That is unreadable, and it is also why nested
// powers exploded -- ((x+1)^8)^8 is 256^8 products distributed pairwise, which did
// not terminate.
//
// Expansion now happens on a canonical form. A polynomial is a map from monomial key
// to monomial; multiplying two of them is one pass over the cross product with like
// terms merged as they are produced, so an intermediate is never larger than the
// answer it represents. ((x+1)^8)^8 is degree 64 and no intermediate exceeds
// sixty-five terms.
//
// A monomial is a numeric coefficient times a product of ATOMS raised to powers. An
// atom is any factor that is not a number and that this machinery cannot look
// inside: a variable, or an opaque subexpression such as sin(x) or x^y. Anything
// that does not decompose is carried through as an atom rather than dropped.
//
// Atoms are identified by STRUCTURE, not by printed text. Printing is not injective:
// sym_to_string renders a constant with six decimals, so sin(1.0000001*x) and
// sin(1.0000002*x) both print as sin((1.000000 * x)). Keying atoms by their printed
// form would merge those two into one and expand their difference to exactly zero --
// a wrong answer with no error. The printed form is used only to bucket candidates;
// identity is settled by sym_equal, which compares values exactly.
// ---------------------------------------------------------------------------

constexpr int kMaxPolyDepth = 256;
constexpr std::size_t kMaxPolyTerms = 4096;

struct Monomial {
    double coefficient = 1.0;
    std::map<int, double> factors;  // atom id -> exponent
};

// key -> monomial. The map both orders terms and merges like ones.
using Polynomial = std::map<std::string, Monomial>;

struct PolyContext {
    std::vector<SymExpr> atoms;                       // id -> the expression
    std::map<std::string, std::vector<int>> buckets;  // printed form -> candidate ids
    bool overflowed = false;

    // Bucket by printed form for speed, then settle identity with sym_equal. Two
    // expressions that print alike but differ are given different ids.
    int intern(const SymExpr& expr) {
        std::string printed = sym_to_string(expr);
        auto& candidates = buckets[printed];
        for (const int id : candidates) {
            if (sym_equal(atoms[static_cast<std::size_t>(id)], expr)) {
                return id;
            }
        }
        const int id = static_cast<int>(atoms.size());
        atoms.push_back(clone_expr(expr));
        candidates.push_back(id);
        return id;
    }
};

// Stable text for an exponent, used only to build map keys: %.17g so that two
// exponents which differ at all get different keys.
std::string exponent_key(double value) {
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

std::string monomial_key(const Monomial& monomial) {
    std::string key;
    for (const auto& [atom, exponent] : monomial.factors) {
        if (exponent == 0.0) {
            continue;
        }
        key += std::to_string(atom);
        key += '^';
        key += exponent_key(exponent);
        key += '*';
    }
    return key;
}

void poly_add_term(Polynomial& poly, Monomial monomial) {
    std::erase_if(monomial.factors, [](const auto& entry) { return entry.second == 0.0; });
    if (monomial.coefficient == 0.0) {
        return;
    }
    const std::string key = monomial_key(monomial);
    const double contribution = monomial.coefficient;
    auto [it, inserted] = poly.try_emplace(key, std::move(monomial));
    if (!inserted) {
        it->second.coefficient += contribution;
        if (it->second.coefficient == 0.0) {
            poly.erase(it);
        }
    }
}

Polynomial poly_constant(double value) {
    Polynomial poly;
    Monomial monomial;
    monomial.coefficient = value;
    poly_add_term(poly, std::move(monomial));
    return poly;
}

Polynomial poly_atom(const SymExpr& expr, PolyContext& context) {
    Monomial monomial;
    monomial.factors[context.intern(expr)] = 1.0;
    Polynomial poly;
    poly_add_term(poly, std::move(monomial));
    return poly;
}

void poly_accumulate(Polynomial& into, const Polynomial& from, double scale) {
    for (const auto& [key, monomial] : from) {
        (void)key;
        Monomial scaled = monomial;
        scaled.coefficient *= scale;
        poly_add_term(into, std::move(scaled));
    }
}

Polynomial poly_multiply(const Polynomial& a, const Polynomial& b, PolyContext& context) {
    Polynomial product;
    for (const auto& [key_a, mono_a] : a) {
        (void)key_a;
        for (const auto& [key_b, mono_b] : b) {
            (void)key_b;
            Monomial combined;
            combined.coefficient = mono_a.coefficient * mono_b.coefficient;
            combined.factors = mono_a.factors;
            for (const auto& [atom, exponent] : mono_b.factors) {
                combined.factors[atom] += exponent;
            }
            poly_add_term(product, std::move(combined));
        }
        if (product.size() > kMaxPolyTerms) {
            context.overflowed = true;
            return product;
        }
    }
    return product;
}

bool poly_is_single_monomial(const Polynomial& poly, Monomial& out) {
    if (poly.size() != 1) {
        return false;
    }
    out = poly.begin()->second;
    return out.coefficient != 0.0;
}

Polynomial expand_to_poly(const SymExpr& expr, PolyContext& context, int depth);

Polynomial poly_power(const Polynomial& base, int exponent, PolyContext& context) {
    Polynomial result = poly_constant(1.0);
    for (int i = 0; i < exponent; ++i) {
        result = poly_multiply(result, base, context);
        if (context.overflowed) {
            return result;
        }
    }
    return result;
}

// Add/Sub/Neg chains are walked with an explicit stack rather than by recursion.
// A sum of a hundred terms parses into a hundred-deep left-leaning tree, and
// recursing it would either blow the depth guard -- declining an expansion that is
// perfectly ordinary -- or hold a hundred polynomial maps live at once.
bool flatten_sum(const SymExpr& expr, PolyContext& context, int depth, Polynomial& out) {
    std::vector<std::pair<const SymExpr*, double>> pending{{&expr, 1.0}};
    bool saw_sum = false;
    while (!pending.empty()) {
        const auto [node, sign] = pending.back();
        pending.pop_back();
        if (node->op == SymOp::Add && node->left && node->right) {
            saw_sum = true;
            pending.emplace_back(node->left.get(), sign);
            pending.emplace_back(node->right.get(), sign);
            continue;
        }
        if (node->op == SymOp::Sub && node->left && node->right) {
            saw_sum = true;
            pending.emplace_back(node->left.get(), sign);
            pending.emplace_back(node->right.get(), -sign);
            continue;
        }
        if (node->op == SymOp::Neg && node->left) {
            saw_sum = true;
            pending.emplace_back(node->left.get(), -sign);
            continue;
        }
        poly_accumulate(out, expand_to_poly(*node, context, depth + 1), sign);
        if (context.overflowed) {
            return saw_sum;
        }
    }
    return saw_sum;
}

Polynomial expand_to_poly(const SymExpr& expr, PolyContext& context, int depth) {
    if (depth > kMaxPolyDepth || context.overflowed) {
        context.overflowed = true;
        return poly_constant(0.0);
    }
    switch (expr.op) {
    case SymOp::Const:
        return poly_constant(expr.value);
    case SymOp::Var:
        return poly_atom(expr, context);
    case SymOp::Add:
    case SymOp::Sub:
    case SymOp::Neg: {
        Polynomial result;
        if (flatten_sum(expr, context, depth, result)) {
            return result;
        }
        break;
    }
    case SymOp::Mul:
        if (expr.left && expr.right) {
            const Polynomial lhs = expand_to_poly(*expr.left, context, depth + 1);
            const Polynomial rhs = expand_to_poly(*expr.right, context, depth + 1);
            return poly_multiply(lhs, rhs, context);
        }
        break;
    case SymOp::Div:
        if (expr.left && expr.right) {
            // Division only by a numeric factor. Cancelling an ATOM would rewrite
            // x/x as 1, which differs from x/x at x = 0, and no amount of algebra
            // here can rule that point out. Dividing by a number is unconditional.
            const Polynomial denominator = expand_to_poly(*expr.right, context, depth + 1);
            Monomial divisor;
            if (!poly_is_single_monomial(denominator, divisor) || !divisor.factors.empty()) {
                break;
            }
            Polynomial result;
            poly_accumulate(result, expand_to_poly(*expr.left, context, depth + 1),
                            1.0 / divisor.coefficient);
            return result;
        }
        break;
    case SymOp::Pow: {
        double power = 0.0;
        if (expr.left && expr.right && try_get_const_value(*expr.right, power)) {
            const bool integral = power == std::floor(power);
            if (integral && power >= 0.0 && power <= static_cast<double>(kMaxExpandExponent)) {
                const int whole = static_cast<int>(power);
                return poly_power(expand_to_poly(*expr.left, context, depth + 1), whole, context);
            }
            const Polynomial base = expand_to_poly(*expr.left, context, depth + 1);
            Monomial single;
            if (!poly_is_single_monomial(base, single)) {
                break;
            }
            // A non-integer exponent may only be pushed through a single atom with a
            // positive coefficient. (c*x)^p and c^p * x^p agree wherever either is
            // real, but (x^2)^0.5 is |x| and not x, and (x*y)^0.5 is not x^0.5*y^0.5
            // when both factors are negative.
            const bool distributable =
                integral || (single.coefficient > 0.0 && single.factors.size() <= 1 &&
                             (single.factors.empty() || single.factors.begin()->second == 1.0));
            if (!distributable) {
                break;
            }
            Monomial raised;
            raised.coefficient = std::pow(single.coefficient, power);
            for (const auto& [atom, exponent] : single.factors) {
                raised.factors[atom] = exponent * power;
            }
            if (!std::isfinite(raised.coefficient)) {
                break;
            }
            Polynomial result;
            poly_add_term(result, std::move(raised));
            return result;
        }
        break;
    }
    default:
        break;
    }
    return poly_atom(expr, context);
}

double monomial_degree(const Monomial& monomial) {
    double degree = 0.0;
    for (const auto& [atom, exponent] : monomial.factors) {
        (void)atom;
        degree += exponent;
    }
    return degree;
}

// |coefficient| * product(atom^exponent), with trivial factors collapsed and
// negative exponents gathered into a denominator so that x/y comes back as x/y
// rather than x * y^-1. The sign is returned separately so the caller can emit a
// subtraction instead of adding a negated term.
SymExpr build_monomial_expr(const Monomial& monomial, const std::vector<SymExpr>& atoms,
                            bool use_absolute_coefficient) {
    auto build_factor = [&](int id, double exponent) {
        SymExpr base = clone_expr(atoms[static_cast<std::size_t>(id)]);
        if (exponent == 1.0) {
            return base;
        }
        return sym_pow(std::move(base), sym_const(exponent));
    };

    SymExpr numerator;
    bool have_numerator = false;
    SymExpr denominator;
    bool have_denominator = false;
    for (const auto& [id, exponent] : monomial.factors) {
        if (exponent == 0.0) {
            continue;
        }
        if (exponent > 0.0) {
            SymExpr factor = build_factor(id, exponent);
            numerator = have_numerator ? sym_mul(std::move(numerator), std::move(factor))
                                       : std::move(factor);
            have_numerator = true;
        } else {
            SymExpr factor = build_factor(id, -exponent);
            denominator = have_denominator ? sym_mul(std::move(denominator), std::move(factor))
                                           : std::move(factor);
            have_denominator = true;
        }
    }

    const double coefficient =
        use_absolute_coefficient ? std::abs(monomial.coefficient) : monomial.coefficient;
    if (!have_numerator) {
        numerator = sym_const(coefficient);
    } else if (coefficient == -1.0) {
        numerator = sym_neg(std::move(numerator));
    } else if (coefficient != 1.0) {
        numerator = sym_mul(sym_const(coefficient), std::move(numerator));
    }
    if (!have_denominator) {
        return numerator;
    }
    return sym_div(std::move(numerator), std::move(denominator));
}

// Terms ordered by descending total degree, so the result reads the way a
// polynomial is written. Ties keep the map's own order, which is by atom id and
// exponent -- arbitrary but deterministic, which is what matters.
SymExpr poly_to_expr(const Polynomial& poly, const std::vector<SymExpr>& atoms) {
    std::vector<const Monomial*> ordered;
    ordered.reserve(poly.size());
    for (const auto& [key, monomial] : poly) {
        (void)key;
        if (monomial.coefficient != 0.0) {
            ordered.push_back(&monomial);
        }
    }
    if (ordered.empty()) {
        return sym_const(0.0);
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const Monomial* a, const Monomial* b) {
        return monomial_degree(*a) > monomial_degree(*b);
    });

    SymExpr result = build_monomial_expr(*ordered.front(), atoms, false);
    for (std::size_t i = 1; i < ordered.size(); ++i) {
        // A negative term is subtracted rather than added as a negation, so that
        // (x+y)*(x-y) reads as x^2 - y^2.
        const bool negative = ordered[i]->coefficient < 0.0;
        SymExpr term = build_monomial_expr(*ordered[i], atoms, negative);
        result = negative ? sym_sub(std::move(result), std::move(term))
                          : sym_add(std::move(result), std::move(term));
    }
    return result;
}

// Every coefficient has to stay finite for the regrouping to have preserved the
// value; a division by zero somewhere inside means handing back the original.
bool poly_is_finite(const Polynomial& poly) {
    for (const auto& [key, monomial] : poly) {
        (void)key;
        if (!std::isfinite(monomial.coefficient)) {
            return false;
        }
        for (const auto& [atom, exponent] : monomial.factors) {
            (void)atom;
            if (!std::isfinite(exponent)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

SymExpr sym_expand(SymExpr expr) {
    // Expand inside the opaque nodes first -- sin((x+1)*(x+2)) should have its
    // argument multiplied out even though the sine itself is not a polynomial --
    // then push the whole thing through the canonical form. The polynomial does the
    // distributing and the collecting in one step, so there is no separate
    // distribution pass here any more.
    if (expr.left) {
        expr.left = std::make_unique<SymExpr>(sym_expand(clone_expr(*expr.left)));
    }
    if (expr.right) {
        expr.right = std::make_unique<SymExpr>(sym_expand(clone_expr(*expr.right)));
    }

    PolyContext context;
    const Polynomial poly = expand_to_poly(expr, context, 0);
    if (context.overflowed || !poly_is_finite(poly)) {
        // Too many terms, too deep, or a coefficient that stopped being a number:
        // hand back what came in rather than a truncated or wrong answer.
        return sym_simplify(std::move(expr));
    }
    return sym_simplify(poly_to_expr(poly, context.atoms));
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

bool sym_equal(const SymExpr& a, const SymExpr& b) {
    return sym_equal_impl(a, b);
}

bool sym_is_unsupported(const SymExpr& result, const std::string& var) {
    return contains_unsupported_sentinel(result, var);
}

// Supported antiderivatives:
//   Const c                          -> c * var
//   Add / Sub / Neg                  -> integrate each child (linearity)
//   Mul / Div by a Const             -> constant pulled out (linearity)
//   Pow(u, Const n), n != -1         -> u^(n+1) / ((n+1) * a)
//   Pow(u, Const -1), Const c / u    -> log(u) / a
//   Sin / Cos / Tan / Exp / Sqrt / Log of u
// where u = a*var + b is any first-degree argument and a is its coefficient: every
// entry is the bare-argument row of the table divided by a, which is the linear
// case of the substitution rule. Small integer powers of a non-linear base are
// expanded first and integrated term by term.
//
// Unsupported forms (genuine chain rule, products of two var-dependent factors,
// ...) return sym_deriv(expr, var) as an explicit sentinel; so does any expression
// with an unsupported subterm, via decline_if_unsupported.
SymExpr sym_integrate(const SymExpr& expr, const std::string& var) {
    // Anything that does not mention the integration variable is a constant with
    // respect to it, whatever its shape. Only a literal and a bare foreign variable
    // used to be recognised, so k*x integrated and (1/k)*x did not -- and dy/dx = y/k,
    // an ordinary way to write a time constant, failed on that alone.
    if (!expression_uses_var(expr, var)) {
        return sym_mul(clone_expr(expr), sym_var(var));
    }
    switch (expr.op) {
    case SymOp::Const:
        return sym_mul(clone_expr(expr), sym_var(var));
    case SymOp::Var:
        if (expr.name == var) {
            return sym_div(sym_pow(sym_var(var), sym_const(2.0)), sym_const(2.0));
        }
        return sym_mul(clone_expr(expr), sym_var(var));
    case SymOp::Add:
        return decline_if_unsupported(
            sym_add(sym_integrate(*expr.left, var), sym_integrate(*expr.right, var)), expr, var);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(sym_integrate(*expr.left, var), sym_integrate(*expr.right, var)), expr, var);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(sym_integrate(*expr.left, var)), expr, var);
    case SymOp::Mul: {
        // try_get_const_value rather than an op test, so that a negative literal --
        // which the parser builds as Neg(Const), never Const(-c) -- is still a
        // constant factor. -3*x^2 used to fall through to the sentinel.
        double c = 0.0;
        if (try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_integrate(*expr.right, var)), expr, var);
        }
        if (try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_integrate(*expr.left, var)), expr, var);
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_integrate_unsupported(expr, var);
        }
        // f/c is the same linearity as c*f, and was the commoner spelling of the two
        // to decline: 0.5*x integrated and x/2 did not.
        double denom = 0.0;
        if (try_get_const_value(*expr.right, denom) && denom != 0.0) {
            return decline_if_unsupported(
                sym_div(sym_integrate(*expr.left, var), sym_const(denom)), expr, var);
        }
        double numerator = 0.0;
        if (try_get_const_value(*expr.left, numerator)) {
            // c/(a*var + b) -> (c/a) * log(a*var + b). The reciprocal of a linear
            // form: the n = -1 carve-out of the power rule, and the base case of
            // every partial-fraction decomposition.
            double a = 0.0;
            double b = 0.0;
            if (match_affine_in_var(*expr.right, var, a, b) && a != 0.0) {
                return sym_mul(sym_const(numerator / a), sym_log(clone_expr(*expr.right)));
            }
            // c/var^n -> c * var^(1-n) / (1-n), with n == 1 the log case above.
            double n = 0.0;
            if (expr.right->op == SymOp::Pow && expr.right->left && expr.right->right &&
                is_bare_var(*expr.right->left, var) && try_get_const_value(*expr.right->right, n)) {
                if (n == 1.0) {
                    return sym_mul(sym_const(numerator), sym_log(sym_var(var)));
                }
                return sym_div(
                    sym_mul(sym_const(numerator), sym_pow(sym_var(var), sym_const(1.0 - n))),
                    sym_const(1.0 - n));
            }
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Pow: {
        // try_get_const_value on the exponent, not an op test: x^(-2) parses as
        // Pow(Var, Neg(Const 2)), so the plain power rule at a negative exponent --
        // which the rule below has always computed correctly -- never ran.
        double n = 0.0;
        const bool const_exponent = expr.right && try_get_const_value(*expr.right, n);
        if (const_exponent && is_bare_var(*expr.left, var)) {
            if (n == -1.0) {
                return sym_log(sym_var(var));
            }
            return sym_div(sym_pow(sym_var(var), sym_const(n + 1.0)), sym_const(n + 1.0));
        }
        double a = 0.0;
        double b = 0.0;
        if (const_exponent && expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            if (n == -1.0) {
                return scale_antiderivative(sym_log(clone_expr(*expr.left)), a);
            }
            return sym_div(
                sym_pow(clone_expr(*expr.left), sym_const(n + 1.0)), sym_const(a * (n + 1.0)));
        }
        // A small integer power of something else: expand and integrate termwise.
        // (t+1)^2 is a polynomial the table covers completely once multiplied out.
        int small = 0;
        if (const_exponent && is_small_nonneg_int(n, small) && small >= 2) {
            SymExpr expanded = sym_expand(clone_expr(expr));
            // Recurse only when expansion actually removed the Pow, so this cannot
            // bounce between two spellings of the same expression.
            if (expanded.op != SymOp::Pow) {
                return decline_if_unsupported(sym_integrate(expanded, var), expr, var);
            }
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Sin: {
        double a = 0.0;
        double b = 0.0;
        if (expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            return scale_antiderivative(sym_neg(sym_cos(clone_expr(*expr.left))), a);
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Cos: {
        double a = 0.0;
        double b = 0.0;
        if (expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            return scale_antiderivative(sym_sin(clone_expr(*expr.left)), a);
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Tan: {
        // integral tan(u) du = -log(cos(u)).
        double a = 0.0;
        double b = 0.0;
        if (expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            return scale_antiderivative(sym_neg(sym_log(sym_cos(clone_expr(*expr.left)))), a);
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Exp: {
        double a = 0.0;
        double b = 0.0;
        if (expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            return scale_antiderivative(sym_exp(clone_expr(*expr.left)), a);
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Sqrt: {
        // The power rule at n = 1/2. The engine already integrated the Pow spelling
        // x^0.5 correctly; only this node type was missing from the switch.
        double a = 0.0;
        double b = 0.0;
        if (expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            return scale_antiderivative(
                sym_div(sym_pow(clone_expr(*expr.left), sym_const(1.5)), sym_const(1.5)), a);
        }
        return sym_integrate_unsupported(expr, var);
    }
    case SymOp::Log: {
        // integral log(u) du = u*log(u) - u.
        double a = 0.0;
        double b = 0.0;
        if (expr.left && match_affine_in_var(*expr.left, var, a, b) && a != 0.0) {
            // Built in two statements: as one expression the order in which the
            // arguments to sym_sub are constructed is unspecified, so std::move(u)
            // could empty u before clone_expr(u) ran -- which it did, giving the
            // antiderivative "( * log()) - x".
            SymExpr u = clone_expr(*expr.left);
            SymExpr product = sym_mul(clone_expr(u), sym_log(clone_expr(u)));
            SymExpr anti = sym_sub(std::move(product), std::move(u));
            return scale_antiderivative(std::move(anti), a);
        }
        return sym_integrate_unsupported(expr, var);
    }
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
    // y/g is the k(x)*y row with k = 1/g. Only the multiplied spelling was matched, so
    // dy/dx = y/2 -- exponential growth with a time constant, the commonest first-order
    // ODE there is -- declined while dy/dx = 0.5*y solved. Recursing rather than
    // requiring a bare y also covers (k*y)/g and (-y)/g, which is how -y/2 parses.
    if (expr.op == SymOp::Div && expr.left && expr.right &&
        !contains_var_name(*expr.right, dep_var)) {
        if (auto inner = sym_extract_y_multiplier(*expr.left, dep_var)) {
            return sym_div(std::move(*inner), clone_expr(*expr.right));
        }
    }
    if (expr.op == SymOp::Neg && expr.left) {
        if (auto inner = sym_extract_y_multiplier(*expr.left, dep_var)) {
            return sym_neg(std::move(*inner));
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

    // dy/dx = c*y^n separates to y^(1-n)/(1-n) = c*x + C. The exponent is read through
    // try_get_const_value because a negative literal is Neg(Const), so y^(-1) never
    // reached this rule though the header's own table row promises it for every n != 1;
    // and 1/y, which is the spelling a user types, is a Div rather than a Pow.
    {
        double coefficient = 1.0;
        double n = 0.0;
        bool matched = false;
        const SymExpr* power = &rhs;
        if (rhs.op == SymOp::Mul && rhs.left && rhs.right) {
            if (try_get_const_value(*rhs.left, coefficient)) {
                power = rhs.right.get();
            } else if (try_get_const_value(*rhs.right, coefficient)) {
                power = rhs.left.get();
            }
        }
        if (power->op == SymOp::Pow && power->left && power->right &&
            is_bare_var(*power->left, dep_var) && try_get_const_value(*power->right, n)) {
            matched = true;
        } else if (power->op == SymOp::Div && power->left && power->right) {
            double numerator = 0.0;
            double denominator_power = 0.0;
            if (try_get_const_value(*power->left, numerator)) {
                // A constant factor in the denominator is part of the coefficient:
                // 1/(2*y) is (1/2)*y^(-1).
                const SymExpr* denominator = power->right.get();
                double denominator_factor = 1.0;
                if (denominator->op == SymOp::Mul && denominator->left && denominator->right) {
                    double factor = 0.0;
                    if (try_get_const_value(*denominator->left, factor) && factor != 0.0) {
                        denominator_factor = factor;
                        denominator = denominator->right.get();
                    } else if (try_get_const_value(*denominator->right, factor) && factor != 0.0) {
                        denominator_factor = factor;
                        denominator = denominator->left.get();
                    }
                }
                if (is_bare_var(*denominator, dep_var)) {
                    coefficient *= numerator / denominator_factor;
                    n = -1.0;
                    matched = true;
                } else if (denominator->op == SymOp::Pow && denominator->left &&
                           denominator->right && is_bare_var(*denominator->left, dep_var) &&
                           try_get_const_value(*denominator->right, denominator_power)) {
                    coefficient *= numerator / denominator_factor;
                    n = -denominator_power;
                    matched = true;
                }
            }
        }
        const double one_minus_n = 1.0 - n;
        if (matched && n != 1.0 && one_minus_n != 0.0 && coefficient != 0.0) {
            SymExpr forcing = coefficient == 1.0
                                  ? sym_var(indep_var)
                                  : sym_mul(sym_const(coefficient), sym_var(indep_var));
            SymExpr shifted = sym_add(std::move(forcing), sym_var("C"));
            SymExpr inner = sym_mul(sym_const(one_minus_n), std::move(shifted));
            return sym_pow(std::move(inner), sym_const(1.0 / one_minus_n));
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
        return decline_if_unsupported(
            sym_add(sym_laplace(*expr.left, t, s), sym_laplace(*expr.right, t, s)), expr, t);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(sym_laplace(*expr.left, t, s), sym_laplace(*expr.right, t, s)), expr, t);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(sym_laplace(*expr.left, t, s)), expr, t);
    case SymOp::Mul: {
        // try_get_const_value rather than an op test: a negative literal is
        // Neg(Const), so -5*exp(2*t) used to miss linearity entirely while
        // 5*exp(2*t) transformed.
        double c = 0.0;
        if (try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_laplace(*expr.right, t, s)), expr, t);
        }
        if (try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_laplace(*expr.left, t, s)), expr, t);
        }
        // First shifting theorem: L{exp(a*t) g(t)} = G(s - a). Stated once here, it
        // supplies every s-shifted row of the table at once -- t*exp(a*t),
        // t^n*exp(a*t), exp(a*t)*sin(b*t), exp(a*t)*cos(b*t) -- instead of one
        // hand-written matcher each.
        double a = 0.0;
        const SymExpr* shifted = nullptr;
        if (match_exp_scaled_var(*expr.left, t, a)) {
            shifted = expr.right.get();
        } else if (match_exp_scaled_var(*expr.right, t, a)) {
            shifted = expr.left.get();
        }
        if (shifted != nullptr) {
            SymExpr g = sym_laplace(*shifted, t, s);
            if (!contains_unsupported_sentinel(g, t)) {
                return sym_simplify(sym_substitute(g, s, build_s_minus_a(s, a)));
            }
        }
        // Frequency differentiation: L{t^n g(t)} = (-1)^n d^n/ds^n G(s). The other
        // general rule, and the source of the t*sin(a*t) and t*cos(a*t) rows.
        int n = 0;
        const SymExpr* differentiated = nullptr;
        if (match_var_power_small_int(*expr.left, t, n)) {
            differentiated = expr.right.get();
        } else if (match_var_power_small_int(*expr.right, t, n)) {
            differentiated = expr.left.get();
        }
        if (differentiated != nullptr) {
            SymExpr g = sym_laplace(*differentiated, t, s);
            if (!contains_unsupported_sentinel(g, t)) {
                for (int i = 0; i < n; ++i) {
                    g = sym_simplify(sym_neg(sym_diff(std::move(g), s)));
                }
                return g;
            }
        }
        return sym_laplace_unsupported(expr, t);
    }
    case SymOp::Div: {
        // f/c is the same linearity as c*f. sym_laplace had no Div case at all, so
        // t/2 declined while 0.5*t transformed.
        double denom = 0.0;
        if (expr.right && try_get_const_value(*expr.right, denom) && denom != 0.0) {
            return decline_if_unsupported(
                sym_div(sym_laplace(*expr.left, t, s), sym_const(denom)), expr, t);
        }
        return sym_laplace_unsupported(expr, t);
    }
    case SymOp::Pow: {
        double power = 0.0;
        const bool const_exponent = expr.right && try_get_const_value(*expr.right, power);
        int n = 0;
        if (const_exponent && is_bare_var(*expr.left, t) && is_small_nonneg_int(power, n)) {
            return sym_div(
                sym_const(factorial_int(n)),
                sym_pow(sym_var(s), sym_const(static_cast<double>(n + 1))));
        }
        // A small integer power of something else -- (t+1)^2 and friends -- is a
        // polynomial the table covers completely once multiplied out.
        if (const_exponent && is_small_nonneg_int(power, n) && n >= 2) {
            SymExpr expanded = sym_expand(clone_expr(expr));
            if (expanded.op != SymOp::Pow) {
                return decline_if_unsupported(sym_laplace(expanded, t, s), expr, t);
            }
        }
        return sym_laplace_unsupported(expr, t);
    }
    case SymOp::Exp: {
        double a = 0.0;
        if (match_scaled_var(*expr.left, t, a)) {
            return sym_div(sym_const(1.0), build_s_minus_a(s, a));
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

// Supported inverse Laplace rules. Every entry is keyed on the shape of the
// denominator, with the numerator taken as an arbitrary first-degree p*s + q:
//
//   (p*s + q) / (s - a)^n          -> exp(a*t) * [p*t^(n-2)/(n-2)! + (q+p*a)*t^(n-1)/(n-1)!]
//   (p*s + q) / ((s - a)^2 + b^2)  -> exp(a*t) * [p*cos(b*t) + ((q+p*a)/b)*sin(b*t)]
//   (p*s + q) / (s^2 - a^2)        -> (p/2 + q/(2a))*exp(a*t) + (p/2 - q/(2a))*exp(-a*t)
//
// a = 0 recovers the unshifted rows (1/s^n, sine, cosine), so the shifting theorem
// costs no separate matcher. The numerator being general is what the previous table
// lacked: it required each numerator to be *exactly* the constant the canonical row
// carries, so 2/(s^2+4) inverted and 1/(s^2+4) -- the same row, differently scaled --
// did not.
//
// Linearity: Add, Sub, Neg, Mul by a Const, and a Const factor in the denominator.
// Unsupported forms return sym_deriv(expr, s) as an explicit sentinel; so does any
// expression with an unsupported subterm, via decline_if_unsupported.
SymExpr sym_ilaplace(const SymExpr& expr, const std::string& s, const std::string& t) {
    switch (expr.op) {
    case SymOp::Const:
        if (expr.value == 0.0) {
            return sym_const(0.0);
        }
        // A non-zero constant is c*delta(t), a distribution rather than a function.
        return sym_ilaplace_unsupported(expr, s);
    case SymOp::Add:
        return decline_if_unsupported(
            sym_add(sym_ilaplace(*expr.left, s, t), sym_ilaplace(*expr.right, s, t)), expr, s);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(sym_ilaplace(*expr.left, s, t), sym_ilaplace(*expr.right, s, t)), expr, s);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(sym_ilaplace(*expr.left, s, t)), expr, s);
    case SymOp::Mul: {
        double c = 0.0;
        if (try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_ilaplace(*expr.right, s, t)), expr, s);
        }
        if (try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_ilaplace(*expr.left, s, t)), expr, s);
        }
        return sym_ilaplace_unsupported(expr, s);
    }
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_ilaplace_unsupported(expr, s);
        }
        // A constant factor in the denominator is the same linearity the numerator
        // already enjoyed: 0.5/s inverted while 1/(2*s) declined.
        if (expr.right->op == SymOp::Mul && expr.right->left && expr.right->right) {
            double factor = 0.0;
            const SymExpr* rest = nullptr;
            if (try_get_const_value(*expr.right->left, factor)) {
                rest = expr.right->right.get();
            } else if (try_get_const_value(*expr.right->right, factor)) {
                rest = expr.right->left.get();
            }
            if (rest != nullptr && factor != 0.0) {
                SymExpr inner =
                    sym_ilaplace(sym_div(clone_expr(*expr.left), clone_expr(*rest)), s, t);
                if (!contains_unsupported_sentinel(inner, s)) {
                    return sym_div(std::move(inner), sym_const(factor));
                }
            }
        }

        // Numerator as p*s + q; a pure constant is the p = 0 case.
        double p = 0.0;
        double q = 0.0;
        if (!match_affine_in_var(*expr.left, s, p, q)) {
            p = 0.0;
            if (!try_get_const_value(*expr.left, q)) {
                return sym_ilaplace_unsupported(expr, s);
            }
        }

        double a = 0.0;
        double b = 0.0;
        int n = 0;
        if (match_shifted_quadratic(*expr.right, s, a, b)) {
            // p*s + q = p*(s - a) + (q + p*a): the first part is the cosine row, the
            // second the sine row, both shifted by exp(a*t).
            const double sine_scale = (q + p * a) / b;
            if (p == 0.0 && sine_scale == 0.0) {
                return sym_const(0.0);
            }
            SymExpr result;
            if (p != 0.0) {
                result = attach_scale(p, sym_cos(sym_mul(sym_const(b), sym_var(t))));
            }
            if (sine_scale != 0.0) {
                SymExpr sine =
                    attach_scale(sine_scale, sym_sin(sym_mul(sym_const(b), sym_var(t))));
                result = p != 0.0 ? sym_add(std::move(result), std::move(sine)) : std::move(sine);
            }
            return attach_exp(std::move(result), a, t);
        }
        if (match_shifted_power(*expr.right, s, a, n)) {
            const double tail = q + p * a;
            // n == 1 with a surviving s in the numerator is p*delta(t) plus a
            // function; the delta is a distribution, so the whole thing is declined
            // rather than silently dropped.
            if (n == 1 && p != 0.0) {
                return sym_ilaplace_unsupported(expr, s);
            }
            const bool has_leading = n >= 2 && p != 0.0;
            if (!has_leading && tail == 0.0) {
                return sym_const(0.0);
            }
            SymExpr result;
            if (has_leading) {
                result = build_monomial(p / factorial_int(n - 2), n - 2, t);
            }
            if (tail != 0.0) {
                SymExpr term = build_monomial(tail / factorial_int(n - 1), n - 1, t);
                result = has_leading ? sym_add(std::move(result), std::move(term))
                                     : std::move(term);
            }
            return attach_exp(std::move(result), a, t);
        }
        if (match_difference_of_squares(*expr.right, s, a)) {
            // p*cosh(a*t) + (q/a)*sinh(a*t), written with exponentials because SymOp
            // carries no hyperbolic nodes.
            const double rising = p / 2.0 + q / (2.0 * a);
            const double falling = p / 2.0 - q / (2.0 * a);
            if (rising == 0.0 && falling == 0.0) {
                return sym_const(0.0);
            }
            SymExpr result;
            if (rising != 0.0) {
                result = attach_exp(sym_const(rising), a, t);
            }
            if (falling != 0.0) {
                SymExpr term = attach_exp(sym_const(falling), -a, t);
                result = rising != 0.0 ? sym_add(std::move(result), std::move(term))
                                       : std::move(term);
            }
            return result;
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
//   1 / (1 + t)          -> pi / sin(pi * s)
// The two exponential rows are gone: M{t^n e^{-a t}} is Gamma(s+n)/a^(s+n), and they
// answered n!/a^(s+n), which is that only at s = 1. SymOp has no Gamma to state them
// with, so they decline until the §10 core does.
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
        return decline_if_unsupported(
            sym_add(sym_mellin(*expr.left, t, s), sym_mellin(*expr.right, t, s)), expr, t);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(sym_mellin(*expr.left, t, s), sym_mellin(*expr.right, t, s)), expr, t);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(sym_mellin(*expr.left, t, s)), expr, t);
    case SymOp::Mul: {
        double c = 0.0;
        if (try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_mellin(*expr.right, t, s)), expr, t);
        }
        if (try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_mellin(*expr.left, t, s)), expr, t);
        }
        // M{t^n e^{-a t}}(s) is Gamma(s + n) / a^(s + n), and this row used to answer
        // n! / a^(s + n) -- the Gamma dropped, which is right only where Gamma(s+n) is
        // n!, that is at s = 1. Under the convention this table declares two lines above
        // its own rational and log rows, that is a wrong answer rather than a missing
        // one. There is no Gamma in SymOp, so the row cannot be stated correctly here;
        // it declines until the §10 core has one.
        // The shifting rule M{t^a f(t)}(s) = M{f}(s + a), which is the Mellin
        // analogue of the Laplace first shifting theorem and, like it, supplies a
        // whole column of the table from one statement.
        int shift = 0;
        const SymExpr* shifted = nullptr;
        if (match_var_power_small_int(*expr.left, t, shift)) {
            shifted = expr.right.get();
        } else if (match_var_power_small_int(*expr.right, t, shift)) {
            shifted = expr.left.get();
        }
        if (shifted != nullptr) {
            SymExpr transformed = sym_mellin(*shifted, t, s);
            if (!contains_unsupported_sentinel(transformed, t)) {
                return sym_simplify(sym_substitute(
                    transformed, s,
                    sym_add(sym_var(s), sym_const(static_cast<double>(shift)))));
            }
        }
        return sym_mellin_unsupported(expr, t);
    }
    case SymOp::Div: {
        double denominator = 0.0;
        if (expr.right && try_get_const_value(*expr.right, denominator) && denominator != 0.0) {
            return decline_if_unsupported(
                sym_div(sym_mellin(*expr.left, t, s), sym_const(denominator)), expr, t);
        }
        if (const auto rational = match_mellin_rational(expr, t)) {
            const double n = static_cast<double>(rational->power);
            SymExpr reflection = sym_div(
                sym_const(std::numbers::pi),
                sym_sin(sym_div(sym_mul(sym_const(std::numbers::pi), sym_var(s)), sym_const(n))));
            SymExpr scaled = sym_mul(sym_const(rational->coefficient / n), std::move(reflection));
            if (rational->offset == 1.0) {
                return sym_simplify(std::move(scaled));
            }
            return sym_simplify(sym_mul(
                std::move(scaled),
                sym_pow(sym_const(rational->offset),
                        sym_sub(sym_div(sym_var(s), sym_const(n)), sym_const(1.0)))));
        }
        if (const auto m = match_one_plus_t_power(expr, t)) {
            return sym_simplify(build_mellin_one_plus_t_power(*m, s));
        }
        // t^k / D is the shifting rule as well -- the power sits in the numerator of a
        // quotient rather than in a product, which is how t/(1+t) is written.
        int shift = 0;
        if (expr.left && match_var_power_small_int(*expr.left, t, shift)) {
            SymExpr reciprocal =
                sym_mellin(sym_div(sym_const(1.0), clone_expr(*expr.right)), t, s);
            if (!contains_unsupported_sentinel(reciprocal, t)) {
                return sym_simplify(sym_substitute(
                    reciprocal, s,
                    sym_add(sym_var(s), sym_const(static_cast<double>(shift)))));
            }
        }
        return sym_mellin_unsupported(expr, t);
    }
    case SymOp::Log:
        // M{log(1 + t)}(s) = pi / (s * sin(pi*s)), valid for -1 < Re(s) < 0. A
        // canonical table entry, and elementary.
        if (expr.left && expr.left->op == SymOp::Add && expr.left->left && expr.left->right) {
            double one = 0.0;
            const bool one_plus_t =
                (try_get_const_value(*expr.left->left, one) && one == 1.0 &&
                 is_bare_var(*expr.left->right, t)) ||
                (try_get_const_value(*expr.left->right, one) && one == 1.0 &&
                 is_bare_var(*expr.left->left, t));
            if (one_plus_t) {
                return sym_div(
                    sym_const(std::numbers::pi),
                    sym_mul(sym_var(s),
                            sym_sin(sym_mul(sym_const(std::numbers::pi), sym_var(s)))));
            }
        }
        return sym_mellin_unsupported(expr, t);
    case SymOp::Pow:
        if (const auto m = match_one_plus_t_power(expr, t)) {
            return sym_simplify(build_mellin_one_plus_t_power(*m, s));
        }
        if (is_bare_var(*expr.left, t) && expr.right->op == SymOp::Const) {
            return sym_div(
                sym_const(1.0),
                sym_add(sym_var(s), clone_expr(*expr.right)));
        }
        return sym_mellin_unsupported(expr, t);
    case SymOp::Exp:
        // M{e^{-a t}}(s) is Gamma(s) / a^s, and this row answered 1 / a^s -- the same
        // dropped Gamma as the t^n e^{-a t} row above, right only at s = 1 where
        // Gamma(1) = 1. It declines rather than answering.
        return sym_mellin_unsupported(expr, t);
    default:
        return sym_mellin_unsupported(expr, t);
    }
}

// Supported inverse Mellin rules (paired with forward table):
//   c / s                  -> c
//   1 / (s + a)            -> t^a
//   pi / sin(pi * s)       -> 1 / (1 + t)
// The n!/a^{n+s} row is gone with its forward partner: it inverted a formula the
// forward table should never have produced.
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
        // pi/sin(pi*s) -> 1/(1+t). Both pi's are compared to a tolerance rather than
        // for equality: sym_to_string prints six decimals, so a spectrum copied back
        // out of the REPL carries 3.141593 and an exact comparison could not read the
        // module's own output. The tolerance is matched to the printer.
        auto is_pi = [](double value) {
            return std::abs(value - std::numbers::pi) <= 1e-5 * std::numbers::pi;
        };
        double sin_coefficient = 0.0;
        if (expr.right->op == SymOp::Sin && expr.right->left && is_pi(numerator)) {
            const SymExpr& sin_arg = *expr.right->left;
            if (sin_arg.op == SymOp::Mul && sin_arg.left && sin_arg.right &&
                try_get_const_value(*sin_arg.left, sin_coefficient) && is_pi(sin_coefficient) &&
                is_bare_var(*sin_arg.right, s)) {
                return sym_div(sym_const(1.0), sym_add(sym_const(1.0), sym_var(t)));
            }
        }
        double a = 0.0;
        if (match_add_var_plus_const(*expr.right, s, a) && numerator == 1.0) {
            return sym_pow(sym_var(t), sym_const(a));
        }
        // 1/a^s -> e^{-a t} and n!/a^{s+n} -> t^n e^{-a t} were the inverses of the two
        // forward rows removed above. Those rows were wrong -- the true transform
        // carries a Gamma(s+n) the table dropped -- so inverting them turned a formula
        // the module should never produce back into a function, and would answer a
        // spectrum nothing here generates. Both are gone with their partners.
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
        return decline_if_unsupported(
            sym_add(sym_hankel(*expr.left, r, k), sym_hankel(*expr.right, r, k)),
            expr, r);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(sym_hankel(*expr.left, r, k), sym_hankel(*expr.right, r, k)),
            expr, r);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(sym_hankel(*expr.left, r, k)), expr, r);
    case SymOp::Mul: {
        double c = 0.0;
        if (try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_hankel(*expr.right, r, k)), expr, r);
        }
        if (try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_hankel(*expr.left, r, k)), expr, r);
        }
        if (const auto matched = match_tpow_exp_neg(expr, r)) {
            return hankel_forward_rpow_exp_neg(matched->first, matched->second, k);
        }
        return sym_hankel_unsupported(expr, r);
    }
    case SymOp::Exp: {
        double a = 0.0;
        if (match_exp_neg_at(expr, r, a)) {
            return hankel_forward_rpow_exp_neg(0, a, k);
        }
        // The Gaussian: H0[exp(-a*r^2)] = exp(-k^2/(4a)) / (2a). The single most-cited
        // order-0 pair, and self-reciprocal at a = 1/2.
        if (const auto coefficient = match_quadratic_coefficient(*expr.left, r)) {
            if (*coefficient < 0.0) {
                const double rate = -*coefficient;
                return sym_div(
                    sym_exp(sym_neg(sym_div(sym_pow(sym_var(k), sym_const(2.0)),
                                            sym_const(4.0 * rate)))),
                    sym_const(2.0 * rate));
            }
        }
        return sym_hankel_unsupported(expr, r);
    }
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_hankel_unsupported(expr, r);
        }
        double denominator = 0.0;
        if (try_get_const_value(*expr.right, denominator) && denominator != 0.0) {
            return decline_if_unsupported(
                sym_div(sym_hankel(*expr.left, r, k), sym_const(denominator)), expr, r);
        }
        double a = 0.0;
        double scale = 1.0;
        if (match_one_over_sqrt_r2_plus_a2(expr, r, a, scale)) {
            return sym_simplify(attach_scale(
                scale,
                sym_div(sym_exp(sym_neg(sym_mul(sym_const(a), sym_var(k)))), sym_var(k))));
        }
        // The Lipschitz integral: H0[c*exp(-a*r)/r] = c/sqrt(a^2 + k^2). The
        // screened-Coulomb pair, in every Hankel table.
        if (match_scaled_exp_neg_over_var(expr, r, a, scale)) {
            return sym_simplify(
                attach_scale(scale, sym_div(sym_const(1.0), sym_sqrt(build_k2_plus_a2(k, a)))));
        }
        double numerator = 0.0;
        if (try_get_const_value(*expr.left, numerator)) {
            // H0[1/r] = 1/k, the self-reciprocal Coulomb kernel.
            if (is_bare_var(*expr.right, r)) {
                return sym_simplify(
                    attach_scale(numerator, sym_div(sym_const(1.0), sym_var(k))));
            }
            // H0[(r^2 + a^2)^(-3/2)] = exp(-a*k)/a, which is the pair above
            // differentiated with respect to a.
            double exponent = 0.0;
            if (expr.right->op == SymOp::Pow && expr.right->left && expr.right->right &&
                try_get_const_value(*expr.right->right, exponent) && exponent == 1.5 &&
                match_s2_plus_a2(*expr.right->left, r, a) && a > 0.0) {
                return sym_simplify(attach_scale(
                    numerator / a, sym_exp(sym_neg(sym_mul(sym_const(a), sym_var(k))))));
            }
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
        return decline_if_unsupported(
            sym_add(sym_ihankel(*expr.left, k, r), sym_ihankel(*expr.right, k, r)),
            expr, k);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(sym_ihankel(*expr.left, k, r), sym_ihankel(*expr.right, k, r)),
            expr, k);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(sym_ihankel(*expr.left, k, r)), expr, k);
    case SymOp::Mul: {
        double c = 0.0;
        if (try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_ihankel(*expr.right, k, r)), expr, k);
        }
        if (try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), sym_ihankel(*expr.left, k, r)), expr, k);
        }
        return sym_ihankel_unsupported(expr, k);
    }
    case SymOp::Exp: {
        // The order-0 transform is its own inverse, so every forward row is an inverse
        // row with r and k swapped. exp(-a*k) had no inverse entry though
        // sym_hankel("exp(-a*r)") has always produced its partner.
        double a = 0.0;
        if (match_exp_neg_at(expr, k, a)) {
            return hankel_forward_rpow_exp_neg(0, a, r);
        }
        if (const auto coefficient = match_quadratic_coefficient(*expr.left, k)) {
            if (*coefficient < 0.0) {
                const double rate = -*coefficient;
                return sym_div(
                    sym_exp(sym_neg(sym_div(sym_pow(sym_var(r), sym_const(2.0)),
                                            sym_const(4.0 * rate)))),
                    sym_const(2.0 * rate));
            }
        }
        return sym_ihankel_unsupported(expr, k);
    }
    case SymOp::Div: {
        if (!expr.left || !expr.right) {
            return sym_ihankel_unsupported(expr, k);
        }
        double denominator = 0.0;
        if (try_get_const_value(*expr.right, denominator) && denominator != 0.0) {
            return decline_if_unsupported(
                sym_div(sym_ihankel(*expr.left, k, r), sym_const(denominator)), expr, k);
        }
        double a = 0.0;
        double scale = 1.0;
        if (match_scaled_exp_neg_over_var(expr, k, a, scale)) {
            return sym_simplify(attach_scale(
                scale, sym_div(sym_const(1.0), sym_sqrt(build_k2_plus_a2(r, a)))));
        }
        // The partner of the Lipschitz row, in the other direction: the transform is
        // self-inverse, so c/sqrt(a^2 + k^2) comes back as c*exp(-a*r)/r. Without it the
        // pair could be transformed forward and not back.
        if (match_one_over_sqrt_r2_plus_a2(expr, k, a, scale) && a > 0.0) {
            return sym_simplify(attach_scale(
                scale,
                sym_div(sym_exp(sym_neg(sym_mul(sym_const(a), sym_var(r)))), sym_var(r))));
        }
        double numerator = 0.0;
        if (try_get_const_value(*expr.left, numerator) && is_bare_var(*expr.right, k)) {
            return sym_simplify(attach_scale(numerator, sym_div(sym_const(1.0), sym_var(r))));
        }
        if (const auto matched = match_hankel_k_domain_rpow_exp(expr, k)) {
            return sym_simplify(attach_scale(
                matched->scale,
                ihankel_inverse_rpow_exp_neg(matched->power, matched->rate, r)));
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

namespace {

void gather_free_variables(const SymExpr& expr, std::vector<std::string>& out) {
    if (expr.op == SymOp::Var) {
        if (std::find(out.begin(), out.end(), expr.name) == out.end()) {
            out.push_back(expr.name);
        }
        return;
    }
    if (expr.left) {
        gather_free_variables(*expr.left, out);
    }
    if (expr.right) {
        gather_free_variables(*expr.right, out);
    }
}

} // namespace

std::vector<std::string> sym_free_variables(const SymExpr& expr) {
    std::vector<std::string> names;
    gather_free_variables(expr, names);
    std::sort(names.begin(), names.end());
    return names;
}

std::string sym_to_string(const SymExpr& expr) {
    switch (expr.op) {
    case SymOp::Const:
        return format_scalar(expr.value);
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

// A numeric two-sided probe. Returns NaN when no limit could be established, which
// callers must check.
//
// The previous implementation initialised its running estimate to 0.0 and returned it
// unconditionally, so a function undefined on one side of the point -- sqrt or log at a
// domain edge -- fell through every iteration of the refinement loop and handed back
// that initialiser as if it were a computed limit: sym_limit(sqrt(x) + 5, "x", 0)
// returned 0.000000 where the answer is 5. It also drove the step to 1e-15, where
// (1 - cos(x))/x^2 evaluates to (1 - 1)/1e-30 = 0, and returned that too.
double sym_limit(const SymExpr& expr, const std::string& var, double point) {
    const auto eval_at = [&](double x) {
        return sym_eval(expr, {{var, x}});
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double point_scale = std::max(1.0, std::abs(point));

    const double direct = eval_at(point);
    if (std::isfinite(direct)) {
        // A finite value at the point, approached from at least one side, is the limit.
        // Every operation the evaluator has is continuous wherever it is defined, so
        // there is no removable discontinuity here to be caught out by -- and requiring
        // BOTH sides, which is what this used to do, disqualifies every function with a
        // one-sided domain and sent it into the loop that then fabricated a zero.
        const double far = 1e-4 * point_scale;
        const double near = 1e-8 * point_scale;
        for (const double side : {-1.0, 1.0}) {
            const double at_far = eval_at(point + side * far);
            const double at_near = eval_at(point + side * near);
            if (!std::isfinite(at_far) || !std::isfinite(at_near)) {
                continue;
            }
            const double gap_far = std::abs(at_far - direct);
            const double gap_near = std::abs(at_near - direct);
            if (gap_near <= gap_far && gap_near < 1e-3 * std::max(1.0, std::abs(direct))) {
                return direct;
            }
        }
    }

    // Each side on its own, then compared.
    //
    // These used to be averaged before either had settled, and the average of two
    // divergences is not a limit: 1/x at 0 has left = -1/h and right = +1/h, whose mean
    // is exactly 0 at every h, so the samples "converged" instantly and the answer came
    // back 0.000000. 1/x^3, 1/sin(x) and tan(x) at pi/2 were the same. Estimating the
    // two sequences separately and requiring them to agree is the definition of a
    // two-sided limit, and needs no threshold on how far apart they are allowed to be
    // along the way -- which is what the averaging was implicitly guessing at.
    //
    // The step floor and keeping the best-converged sample rather than the last are
    // both deliberate: past a certain h the samples stop improving and decay into
    // rounding noise, and the decay eventually settles on a constant whose successive
    // differences are exactly zero and therefore look like perfect convergence.
    // (1 - cos(x))/x^2 is the case that matters -- its samples reach 0.4999999970 and
    // then collapse to 0 at h = 1e-8.
    struct OneSided {
        double value = std::numeric_limits<double>::quiet_NaN();
        bool converged = false;
    };
    const auto approach = [&](double side) {
        OneSided result;
        double best_delta = std::numeric_limits<double>::infinity();
        double previous = std::numeric_limits<double>::quiet_NaN();
        double h = 1e-2 * point_scale;
        for (int step = 0; step < 10; ++step) {
            const double sample = eval_at(point + side * h);
            if (std::isfinite(sample)) {
                if (std::isfinite(previous)) {
                    const double delta =
                        std::abs(sample - previous) / std::max(1.0, std::abs(sample));
                    if (std::isfinite(best_delta) && delta > 4.0 * best_delta) {
                        break;
                    }
                    if (delta < best_delta) {
                        best_delta = delta;
                        result.value = sample;
                    }
                    if (delta < 1e-12) {
                        result.value = sample;
                        result.converged = true;
                        return result;
                    }
                }
                previous = sample;
            }
            h *= 0.1;
        }
        result.converged = best_delta <= 1e-6;
        return result;
    };

    const OneSided from_left = approach(-1.0);
    const OneSided from_right = approach(1.0);
    if (from_left.converged && from_right.converged) {
        const double scale = std::max(1.0, std::abs(from_left.value));
        if (std::abs(from_left.value - from_right.value) > 1e-6 * scale) {
            // Both sides settle, on different values: abs(x)/x at 0 approaches -1 and
            // +1, and has no limit.
            return nan;
        }
        return 0.5 * (from_left.value + from_right.value);
    }
    // A one-sided domain is not a failure: sqrt(x) + 5 at 0 has no left side at all,
    // and requiring both is what used to send it into the loop that fabricated a zero.
    if (from_left.converged) {
        return from_left.value;
    }
    if (from_right.converged) {
        return from_right.value;
    }
    // Neither side settled: a divergent or non-existent limit, not a number. log(x) at
    // 0 marched off towards -infinity and the old code returned whichever value it
    // happened to stop on.
    return nan;
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

        return sym_simplify(std::move(coef));
    }
    if (leaf.op == SymOp::Neg && leaf.left) {
        auto inner = extract_linear_term(*leaf.left, vars, matched_var);
        if (!inner) {
            return std::nullopt;
        }
        return sym_neg(std::move(*inner));
    }

    // f/g with g free of the unknowns is linear in the same variables as f, with
    // every coefficient divided by g. Without this, x/2 was not recognised as a term
    // at all -- and, because flatten_linear_sum used to discard what it could not
    // parse, the system came back singular rather than wrong.
    if (leaf.op == SymOp::Div && leaf.left && leaf.right) {
        bool denominator_uses_unknown = false;
        for (const auto& solve_var : vars) {
            if (contains_var_name(*leaf.right, solve_var)) {
                denominator_uses_unknown = true;
                break;
            }
        }
        if (!denominator_uses_unknown) {
            auto inner = extract_linear_term(*leaf.left, vars, matched_var);
            if (inner) {
                return sym_simplify(sym_div(std::move(*inner), clone_expr(*leaf.right)));
            }
        }
        return std::nullopt;
    }

    // Anything left that does not mention an unknown is a constant of this system --
    // sin(y) when solving for x, exp(a), a bare symbol. Returning nullopt for it,
    // which is what this did, made flatten_linear_sum drop the term: "x + sin(y) - 1"
    // solved to x = 1 instead of x = 1 - sin(y), and "x - exp(a)" to x = -0. Both were
    // wrong answers with no error attached.
    for (const auto& solve_var : vars) {
        if (contains_var_name(leaf, solve_var)) {
            return std::nullopt;
        }
    }
    return clone_expr(leaf);
}

// Returns false when some term of the sum is not linear in the unknowns. It used to
// return void and simply skip such a term, so a quadratic or a product of two
// unknowns silently became a linear system missing a piece, and the caller reported
// a confident solution to an equation it had not been given.
bool flatten_linear_sum(
    const SymExpr& expr,
    double sign,
    const std::vector<std::string>& vars,
    std::map<std::string, SymExpr>& var_coeffs,
    SymExpr& constant) {
    switch (expr.op) {
    case SymOp::Add:
        if (expr.left && !flatten_linear_sum(*expr.left, sign, vars, var_coeffs, constant)) {
            return false;
        }
        return !expr.right || flatten_linear_sum(*expr.right, sign, vars, var_coeffs, constant);
    case SymOp::Sub:
        if (expr.left && !flatten_linear_sum(*expr.left, sign, vars, var_coeffs, constant)) {
            return false;
        }
        return !expr.right || flatten_linear_sum(*expr.right, -sign, vars, var_coeffs, constant);
    case SymOp::Neg:
        return !expr.left || flatten_linear_sum(*expr.left, -sign, vars, var_coeffs, constant);
    case SymOp::Div: {
        // A numeric denominator divides every coefficient of the numerator, so it can
        // ride on `sign` and the sum inside it still splits: (x + 1)/2 and x/2 + y/3
        // both work out here rather than having to be a single leaf term.
        double denominator = 0.0;
        if (expr.left && expr.right && try_eval_const(*expr.right, denominator) &&
            denominator != 0.0) {
            return flatten_linear_sum(*expr.left, sign / denominator, vars, var_coeffs, constant);
        }
        break;
    }
    default:
        break;
    }
    {
        std::string matched_var;
        auto parsed = extract_linear_term(expr, vars, matched_var);
        if (!parsed) {
            return false;
        }
        SymExpr scaled = scale_expr(std::move(*parsed), sign);
        if (matched_var.empty()) {
            constant = sym_simplify(sym_add(std::move(constant), std::move(scaled)));
        } else {
            var_coeffs[matched_var] =
                sym_simplify(sym_add(std::move(var_coeffs[matched_var]), std::move(scaled)));
        }
        return true;
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
    // Expand before flattening so that a bracketed coefficient -- 2*(x + 1) - 8, the
    // ordinary way to write such an equation -- becomes a sum of terms this can read.
    SymExpr normalised = sym_simplify(sym_expand(clone_expr(equation)));
    if (!flatten_linear_sum(normalised, 1.0, vars, row.var_coeffs, row.constant)) {
        return std::nullopt;
    }
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
    // var/c is the same scaling as (1/c)*var, and exp(-t/2) is an ordinary way to
    // write a decay rate.
    if (expr.op == SymOp::Div && expr.left && expr.right && is_named_var(*expr.left, var)) {
        double denom = 0.0;
        if (try_get_const_value(*expr.right, denom) && denom != 0.0) {
            return 1.0 / denom;
        }
    }
    return std::nullopt;
}

// The signed coefficient c in c * var^2, for every spelling the parser produces:
// var^2 itself, a constant multiple either way round, a quotient by a constant, and
// any of those negated. The Gaussian matchers keyed on one hard-coded shape --
// Neg(Mul(Const, Pow(var, 2))) -- so exp(-(t^2)) (no Mul at all), exp(-2*t^2) (the
// minus binds to the coefficient, leaving no Neg) and exp(-t^2/2) (a Div, not a Mul)
// all missed the table, though each is an ordinary way to write a Gaussian.
std::optional<double> match_quadratic_coefficient(const SymExpr& expr, const std::string& var) {
    if (expr.op == SymOp::Pow && expr.left && expr.right && is_named_var(*expr.left, var) &&
        expr.right->op == SymOp::Const && expr.right->value == 2.0) {
        return 1.0;
    }
    if (expr.op == SymOp::Neg && expr.left) {
        if (const auto inner = match_quadratic_coefficient(*expr.left, var)) {
            return -*inner;
        }
        return std::nullopt;
    }
    if (expr.op == SymOp::Sub && expr.left && expr.right && is_const_zero(*expr.left)) {
        if (const auto inner = match_quadratic_coefficient(*expr.right, var)) {
            return -*inner;
        }
        return std::nullopt;
    }
    double factor = 0.0;
    if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (try_get_const_value(*expr.left, factor)) {
            if (const auto inner = match_quadratic_coefficient(*expr.right, var)) {
                return factor * *inner;
            }
        }
        if (try_get_const_value(*expr.right, factor)) {
            if (const auto inner = match_quadratic_coefficient(*expr.left, var)) {
                return factor * *inner;
            }
        }
        return std::nullopt;
    }
    if (expr.op == SymOp::Div && expr.left && expr.right &&
        try_get_const_value(*expr.right, factor) && factor != 0.0) {
        if (const auto inner = match_quadratic_coefficient(*expr.left, var)) {
            return *inner / factor;
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
    const auto coefficient = match_quadratic_coefficient(*expr.left, t_var);
    if (!coefficient || *coefficient >= 0.0) {
        return std::nullopt;
    }
    return -*coefficient;
}

// c / (a^2 + var^2), a > 0. Returns a and sets `scale` to c / (2a), the multiple of
// the canonical row this is. The old matcher demanded c == 2a exactly, so only the
// pre-scaled spelling was accepted and 1/(1+w^2) -- the unit Lorentzian, the same row
// at half the amplitude -- was refused.
std::optional<double> match_rational_decay_form(
    const SymExpr& expr, const std::string& omega_var, double& scale) {
    if (expr.op != SymOp::Div || !expr.left || !expr.right) {
        return std::nullopt;
    }
    double numerator = 0.0;
    if (!try_get_const_value(*expr.left, numerator)) {
        return std::nullopt;
    }
    double shift = 0.0;
    double a = 0.0;
    if (!match_shifted_quadratic(*expr.right, omega_var, shift, a) || shift != 0.0 || a <= 0.0) {
        return std::nullopt;
    }
    scale = numerator / (2.0 * a);
    return a;
}

// scale * exp(-var^2/(4a)) with scale == sqrt(pi/a): the transform of exp(-a*t^2).
// `a` is read off the exponent, which is exact, rather than derived from the scale,
// which is not: sym_to_string prints six decimals, so a spectrum copied back out of
// the REPL carries a scale good to about 1e-6. The old matcher required agreement to
// 1e-9 and so could not read the module's own printed output -- the Gaussian pair
// failed to round-trip. The scale is still checked, to a tolerance matched to the
// printer rather than to the arithmetic.
std::optional<double> match_gaussian_spectrum_form(
    const SymExpr& expr, const std::string& omega_var, double& scale_out) {
    if (expr.op != SymOp::Mul || !expr.left || !expr.right) {
        return std::nullopt;
    }
    double scale = 0.0;
    const SymExpr* gaussian = nullptr;
    if (try_get_const_value(*expr.left, scale) && expr.right->op == SymOp::Exp) {
        gaussian = expr.right.get();
    } else if (try_get_const_value(*expr.right, scale) && expr.left->op == SymOp::Exp) {
        gaussian = expr.left.get();
    } else {
        return std::nullopt;
    }
    if (scale <= 0.0 || !gaussian->left) {
        return std::nullopt;
    }
    const auto coefficient = match_quadratic_coefficient(*gaussian->left, omega_var);
    if (!coefficient || *coefficient >= 0.0) {
        return std::nullopt;
    }
    // exponent coefficient == -1/(4a)
    const double a = -1.0 / (4.0 * *coefficient);
    if (a <= 0.0) {
        return std::nullopt;
    }
    const double canonical = std::sqrt(std::numbers::pi / a);
    if (std::abs(scale - canonical) > 1e-5 * canonical) {
        return std::nullopt;
    }
    scale_out = scale / canonical;
    return a;
}

// The base a of a geometric sequence a^n.
//
// This used to also accept a constant multiple and return sym_mul(coefficient, base)
// as the base -- which moved the pole. Z{3*2^n} came back as z/(z - 6) where the
// answer is 3*z/(z - 2): a wrong result, silently, for every scaled geometric
// sequence. A constant factor is linearity and belongs in transform_linearity, not
// in the base.
std::optional<SymExpr> match_geometric_sequence(const SymExpr& expr, const std::string& n_var) {
    if (expr.op == SymOp::Pow && expr.left && expr.right && is_named_var(*expr.right, n_var) &&
        !expression_uses_var(*expr.left, n_var)) {
        return clone_expr(*expr.left);
    }
    // exp(a*n) is the same row written the other way round, with base exp(a).
    if (expr.op == SymOp::Exp && expr.left) {
        const SymExpr& inner = *expr.left;
        if (is_named_var(inner, n_var)) {
            return sym_const(std::numbers::e);
        }
        if (inner.op == SymOp::Mul && inner.left && inner.right) {
            if (is_named_var(*inner.right, n_var) && !expression_uses_var(*inner.left, n_var)) {
                return sym_exp(clone_expr(*inner.left));
            }
            if (is_named_var(*inner.left, n_var) && !expression_uses_var(*inner.right, n_var)) {
                return sym_exp(clone_expr(*inner.right));
            }
        }
    }
    return std::nullopt;
}

// w in cos(w*n) or sin(w*n), where w does not itself involve n.
std::optional<SymExpr> match_sampled_frequency(
    const SymExpr& expr, const std::string& n_var, SymOp wanted) {
    if (expr.op != wanted || !expr.left) {
        return std::nullopt;
    }
    const SymExpr& inner = *expr.left;
    if (is_named_var(inner, n_var)) {
        return sym_const(1.0);
    }
    if (inner.op == SymOp::Mul && inner.left && inner.right) {
        if (is_named_var(*inner.right, n_var) && !expression_uses_var(*inner.left, n_var)) {
            return clone_expr(*inner.left);
        }
        if (is_named_var(*inner.left, n_var) && !expression_uses_var(*inner.right, n_var)) {
            return clone_expr(*inner.right);
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
    if (!den.left || !den.right || !is_named_var(*den.left, z_var)) {
        return std::nullopt;
    }
    if (den.op == SymOp::Sub) {
        return clone_expr(*den.right);
    }
    // z/(z + a) is the pole at -a, which is where the alternating sequence (-1)^n
    // lives -- one of the commonest single-pole cases there is, and previously
    // unmatched because only the Sub spelling was accepted.
    if (den.op == SymOp::Add) {
        double a = 0.0;
        if (try_get_const_value(*den.right, a)) {
            return sym_const(-a);
        }
        return sym_neg(clone_expr(*den.right));
    }
    return std::nullopt;
}

} // namespace

// Canonical pairs (MVP):
//   exp(-a|t|) ~ exp(-a*t)  <->  2a/(a^2 + omega^2)
//   exp(-a*t^2)             <->  sqrt(pi/a)*exp(-omega^2/(4a))
//   a^n                     <->  z/(z-a)
namespace {

// Linearity, shared by the Fourier pair and the Z-transform. All three were pure
// table lookup with no recursion at all -- exp(-2*t^2) transformed and 3*exp(-2*t^2)
// did not, though F[c*f] = c*F[f] is the defining property of a transform. Applied
// after the table rows, so a row whose own top node is a Div or a Mul still matches
// first.
SymExpr transform_linearity(
    const SymExpr& expr, const std::string& from_var,
    const std::function<SymExpr(const SymExpr&)>& recurse) {
    switch (expr.op) {
    case SymOp::Add:
        return decline_if_unsupported(
            sym_add(recurse(*expr.left), recurse(*expr.right)), expr, from_var);
    case SymOp::Sub:
        return decline_if_unsupported(
            sym_sub(recurse(*expr.left), recurse(*expr.right)), expr, from_var);
    case SymOp::Neg:
        return decline_if_unsupported(sym_neg(recurse(*expr.left)), expr, from_var);
    case SymOp::Mul: {
        double c = 0.0;
        if (expr.left && try_get_const_value(*expr.left, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), recurse(*expr.right)), expr, from_var);
        }
        if (expr.right && try_get_const_value(*expr.right, c)) {
            return decline_if_unsupported(
                sym_mul(sym_const(c), recurse(*expr.left)), expr, from_var);
        }
        return sym_transform_unsupported(expr, from_var);
    }
    case SymOp::Div: {
        double denom = 0.0;
        if (expr.right && try_get_const_value(*expr.right, denom) && denom != 0.0) {
            return decline_if_unsupported(
                sym_div(recurse(*expr.left), sym_const(denom)), expr, from_var);
        }
        return sym_transform_unsupported(expr, from_var);
    }
    default:
        return sym_transform_unsupported(expr, from_var);
    }
}

} // namespace

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
    // The Lorentzian, forward. F[2a/(a^2 + t^2)] = 2*pi*exp(-a|omega|) is the dual of
    // the two-sided exponential above; only the inverse direction of this pair was
    // implemented, so the module could transform one half of it and not the other.
    // The library writes the two-sided decay without an absolute value, as the
    // inverse direction already does.
    double lorentzian_scale = 0.0;
    if (const auto a = match_rational_decay_form(expr, t_var, lorentzian_scale)) {
        SymExpr decay = sym_exp(sym_neg(sym_mul(sym_const(*a), sym_var(omega_var))));
        return sym_simplify(
            sym_mul(sym_const(lorentzian_scale * 2.0 * std::numbers::pi), std::move(decay)));
    }
    return transform_linearity(expr, t_var, [&](const SymExpr& child) {
        return sym_fourier(child, t_var, omega_var);
    });
}

SymExpr sym_ifourier(const SymExpr& expr, const std::string& omega_var, const std::string& t_var) {
    double decay_scale = 0.0;
    if (const auto a = match_rational_decay_form(expr, omega_var, decay_scale)) {
        SymExpr decay = sym_exp(sym_neg(sym_mul(sym_const(*a), sym_var(t_var))));
        return sym_simplify(attach_scale(decay_scale, std::move(decay)));
    }
    double gaussian_scale = 0.0;
    if (const auto a = match_gaussian_spectrum_form(expr, omega_var, gaussian_scale)) {
        SymExpr gaussian =
            sym_exp(sym_neg(sym_mul(sym_const(*a), sym_pow(sym_var(t_var), sym_const(2.0)))));
        return sym_simplify(attach_scale(gaussian_scale, std::move(gaussian)));
    }
    return transform_linearity(expr, omega_var, [&](const SymExpr& child) {
        return sym_ifourier(child, omega_var, t_var);
    });
}

// Supported Z-transform rules:
//   c                     -> c * z/(z - 1)
//   a^n, exp(a*n)         -> z/(z - a)
//   n^k * f(n)            -> (-z d/dz)^k F(z)
//   cos(w*n)              -> z*(z - cos w) / (z^2 - 2*z*cos w + 1)
//   sin(w*n)              -> z*sin w       / (z^2 - 2*z*cos w + 1)
// Linearity: Add, Sub, Neg, Mul/Div by a Const.
SymExpr sym_ztransform(const SymExpr& expr, const std::string& n_var, const std::string& z_var) {
    if (auto base = match_geometric_sequence(expr, n_var)) {
        return sym_simplify(sym_div(sym_var(z_var), sym_sub(sym_var(z_var), std::move(*base))));
    }
    if (expr.op == SymOp::Const) {
        return sym_simplify(sym_mul(
            clone_expr(expr), sym_div(sym_var(z_var), sym_sub(sym_var(z_var), sym_const(1.0)))));
    }
    // The sampled sinusoids, which share the denominator z^2 - 2*z*cos(w) + 1.
    for (const SymOp wanted : {SymOp::Cos, SymOp::Sin}) {
        auto w = match_sampled_frequency(expr, n_var, wanted);
        if (!w) {
            continue;
        }
        SymExpr denominator = sym_add(
            sym_sub(sym_pow(sym_var(z_var), sym_const(2.0)),
                    sym_mul(sym_const(2.0), sym_mul(sym_var(z_var), sym_cos(clone_expr(*w))))),
            sym_const(1.0));
        SymExpr numerator =
            wanted == SymOp::Cos
                ? sym_mul(sym_var(z_var), sym_sub(sym_var(z_var), sym_cos(clone_expr(*w))))
                : sym_mul(sym_var(z_var), sym_sin(clone_expr(*w)));
        return sym_simplify(sym_div(std::move(numerator), std::move(denominator)));
    }
    // Z{n^k f(n)} = (-z d/dz)^k F(z): the discrete counterpart of the Laplace
    // frequency-differentiation rule, and the whole n^k family in one statement.
    // Z{n} = z/(z-1)^2 is its k = 1, f = 1 case.
    int k = 0;
    SymExpr inner;
    bool have_inner = false;
    if (match_var_power_small_int(expr, n_var, k)) {
        inner = sym_const(1.0);
        have_inner = true;
    } else if (expr.op == SymOp::Mul && expr.left && expr.right) {
        if (match_var_power_small_int(*expr.left, n_var, k)) {
            inner = clone_expr(*expr.right);
            have_inner = true;
        } else if (match_var_power_small_int(*expr.right, n_var, k)) {
            inner = clone_expr(*expr.left);
            have_inner = true;
        }
    }
    if (have_inner) {
        SymExpr transformed = sym_ztransform(inner, n_var, z_var);
        if (!contains_unsupported_sentinel(transformed, n_var)) {
            for (int i = 0; i < k; ++i) {
                transformed = sym_simplify(
                    sym_neg(sym_mul(sym_var(z_var), sym_diff(std::move(transformed), z_var))));
            }
            return transformed;
        }
    }
    return transform_linearity(expr, n_var, [&](const SymExpr& child) {
        return sym_ztransform(child, n_var, z_var);
    });
}

SymExpr sym_iztransform(const SymExpr& expr, const std::string& z_var, const std::string& n_var) {
    if (auto base = match_z_over_z_minus_a(expr, z_var)) {
        double value = 0.0;
        // z/(z - 1) is the unit sequence; 1^n is the same thing spelled worse.
        if (try_get_const_value(*base, value) && value == 1.0) {
            return sym_const(1.0);
        }
        return sym_simplify(sym_pow(std::move(*base), sym_var(n_var)));
    }
    // z/(z - a)^2 -> n * a^(n-1): the inverse of the multiplication-by-n rule, and
    // the partner of the Z{n} row above. Without it the table was asymmetric -- every
    // repeated pole a partial-fraction expansion produces was a dead end.
    if (expr.op == SymOp::Div && expr.left && is_named_var(*expr.left, z_var) && expr.right &&
        expr.right->op == SymOp::Pow && expr.right->left && expr.right->right &&
        expr.right->right->op == SymOp::Const && expr.right->right->value == 2.0) {
        const SymExpr& denominator = *expr.right->left;
        if (denominator.op == SymOp::Sub && denominator.left && denominator.right &&
            is_named_var(*denominator.left, z_var)) {
            double a = 0.0;
            if (try_get_const_value(*denominator.right, a) && a != 0.0) {
                if (a == 1.0) {
                    return sym_var(n_var);
                }
                return sym_simplify(sym_mul(
                    sym_var(n_var),
                    sym_pow(sym_const(a), sym_sub(sym_var(n_var), sym_const(1.0)))));
            }
        }
    }
    return transform_linearity(expr, z_var, [&](const SymExpr& child) {
        return sym_iztransform(child, z_var, n_var);
    });
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
