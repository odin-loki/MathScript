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
// Forward/inverse transforms. Unsupported forms return sym_deriv(expr, var)
// as an explicit sentinel (same convention as sym_integrate).
SymExpr sym_laplace(const SymExpr& expr, const std::string& t, const std::string& s);
SymExpr sym_ilaplace(const SymExpr& expr, const std::string& s, const std::string& t);
SymExpr sym_mellin(const SymExpr& expr, const std::string& t, const std::string& s);
SymExpr sym_imellin(const SymExpr& expr, const std::string& s, const std::string& t);
SymExpr sym_hankel(const SymExpr& expr, const std::string& r, const std::string& k);
SymExpr sym_ihankel(const SymExpr& expr, const std::string& k, const std::string& r);
SymExpr sym_fourier(const SymExpr& expr, const std::string& t_var, const std::string& omega_var);
SymExpr sym_ifourier(const SymExpr& expr, const std::string& omega_var, const std::string& t_var);
SymExpr sym_ztransform(const SymExpr& expr, const std::string& n_var, const std::string& z_var);
SymExpr sym_iztransform(const SymExpr& expr, const std::string& z_var, const std::string& n_var);
// First-order ODE dy/d(indep)=rhs. Table-driven separable MVP; unsupported -> sym_deriv(rhs, indep).
SymExpr sym_dsolve(const SymExpr& rhs, const std::string& indep_var, const std::string& dep_var);
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
