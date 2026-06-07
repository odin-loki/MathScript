// MathScript Symbolic Expression Library
// CAS (Computer Algebra System) core

#pragma once

#include "ms/error/error_types.hpp"

namespace ms::sym {

// Symbol class for variable and constant representation
class Symbol {
public:
    Symbol(const std::string& name);
    Symbol(double value);
    
    std::string name() const;
    double value() const;
    
    Symbol operator+(const Symbol& o) const;
    Symbol operator-(const Symbol& o) const;
    Symbol operator*(const Symbol& o) const;
    Symbol operator/(const Symbol& o) const;
    Symbol operator^(const Symbol& o) const;
    
private:
    std::string name_;
    double value_;
};

// Expression class for arbitrary expressions
class Expression {
public:
    Expression() = default;
    Expression(const std::string& expr);
    Expression(double value);
    
    std::string to_string() const;
    std::string to_latex() const;
    
    Result<double> eval(const std::map<std::string, double>& subs) const;
    Result<Expression> diff(const Symbol& var) const;
    Result<Expression> integrate(const Symbol& var) const;
    
private:
    std::string expr_;
    bool has_expression_ = true;
};

// Simplify expression
Result<Expression> simplify(const Expression& expr);

// Expand expression
Result<Expression> expand(const Expression& expr);

// Collect terms
Result<Expression> collect(const Expression& expr, const Symbol& var) const;

} // namespace ms::sym