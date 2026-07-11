#pragma once

#include <cstddef>
#include <expected>
#include <map>
#include <memory>
#include <string>

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
SymExpr sym_integrate(const SymExpr& expr, const std::string& var);
SymExpr sym_substitute(const SymExpr& expr, const std::string& var, const SymExpr& replacement);
double sym_eval(const SymExpr& expr, const std::map<std::string, double>& env);
std::string sym_to_string(const SymExpr& expr);

struct SymParseError {
    std::string message;
    size_t position = 0;
};

std::expected<SymExpr, SymParseError> sym_parse(const std::string& text);

} // namespace ms
