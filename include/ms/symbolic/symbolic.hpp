#pragma once

#include <cstddef>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ms {

enum class SymOp {
    Add,
    Mul,
    Var,
    Const,
    Deriv,
    Sin,
    Cos,
    Exp,
    Log,
    Pow,
    Sub,
    Div,
    Neg,
    Tan,
    Sqrt
};

struct SymExpr {
    SymOp op = SymOp::Const;
    double value = 0.0;
    std::string name;
    std::unique_ptr<SymExpr> left;
    std::unique_ptr<SymExpr> right;
};

SymExpr sym_const(double value);
SymExpr sym_var(const std::string& name);
SymExpr sym_add(SymExpr a, SymExpr b);
SymExpr sym_mul(SymExpr a, SymExpr b);
SymExpr sym_sub(SymExpr a, SymExpr b);
SymExpr sym_div(SymExpr a, SymExpr b);
SymExpr sym_neg(SymExpr a);
SymExpr sym_sin(SymExpr arg);
SymExpr sym_cos(SymExpr arg);
SymExpr sym_tan(SymExpr arg);
SymExpr sym_exp(SymExpr arg);
SymExpr sym_log(SymExpr arg);
SymExpr sym_sqrt(SymExpr arg);
SymExpr sym_pow(SymExpr base, SymExpr exponent);
SymExpr sym_deriv(SymExpr expr, const std::string& var);
SymExpr sym_diff(SymExpr expr, const std::string& var);
SymExpr sym_simplify(SymExpr expr);
SymExpr sym_expand(SymExpr expr);
SymExpr sym_collect(const SymExpr& expr, const std::string& var);
SymExpr sym_integrate(const SymExpr& expr, const std::string& var);
// Forward/inverse Laplace transforms. Unsupported forms return sym_deriv(expr, t_or_s)
// as an explicit sentinel (same convention as sym_integrate).
SymExpr sym_laplace(const SymExpr& expr, const std::string& t, const std::string& s);
SymExpr sym_ilaplace(const SymExpr& expr, const std::string& s, const std::string& t);
SymExpr sym_substitute(const SymExpr& expr, const std::string& var, const SymExpr& replacement);
double sym_eval(const SymExpr& expr, const std::map<std::string, double>& env);
std::string sym_to_string(const SymExpr& expr);

struct SymParseError {
    std::string message;
    size_t position = 0;
};

std::expected<SymExpr, SymParseError> sym_parse(const std::string& text);

double sym_limit(const SymExpr& expr, const std::string& var, double point);
SymExpr sym_series(const SymExpr& expr, const std::string& var, double point, int order);

struct SymSolveError {
    std::string message;
};

std::expected<std::map<std::string, SymExpr>, SymSolveError> sym_solve_linear(
    const std::vector<SymExpr>& equations, const std::vector<std::string>& vars);

} // namespace ms
