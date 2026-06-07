#pragma once

#include <map>
#include <memory>
#include <string>

namespace ms {

enum class SymOp { Add, Mul, Var, Const, Deriv, Sin, Cos, Exp, Log, Pow };

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
SymExpr sym_sin(SymExpr arg);
SymExpr sym_cos(SymExpr arg);
SymExpr sym_exp(SymExpr arg);
SymExpr sym_log(SymExpr arg);
SymExpr sym_pow(SymExpr base, SymExpr exponent);
SymExpr sym_deriv(SymExpr expr, const std::string& var);
SymExpr sym_diff(SymExpr expr, const std::string& var);
SymExpr sym_simplify(SymExpr expr);
double sym_eval(const SymExpr& expr, const std::map<std::string, double>& env);
std::string sym_to_string(const SymExpr& expr);

} // namespace ms
