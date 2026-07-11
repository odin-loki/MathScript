// MathScript Symbolic Expression Header
// CAS (Computer Algebra System) core

#pragma once

#include <map>
#include <string>

namespace ms {

/// Lightweight string-based symbolic expression with parse-and-eval support.
/// Grammar: +, -, *, / (left-associative precedence), unary -, parentheses,
/// numeric literals, variables (identifiers), and sin/cos/exp/log/sqrt/tanh calls.
/// Axiom binds data columns to variables x0, x1, x2, ...
class Sym {
public:
    Sym() = default;
    Sym(const char* expr);
    Sym(double value);

    Sym& operator+=(const Sym& other);
    Sym& operator-=(const Sym& other);
    Sym& operator*=(const Sym& other);
    Sym& operator/=(const Sym& other);

    Sym operator+(const Sym& other) const;
    Sym operator-(const Sym& other) const;
    Sym operator*(const Sym& other) const;
    Sym operator/(const Sym& other) const;

    /// Evaluate with variable bindings. Unbound variables default to 0.0.
    /// Malformed expressions evaluate to 0.0 (no throw).
    double eval(const std::map<std::string, double>& env = {}) const;
    std::string to_string() const;

private:
    std::string expr_;
    bool has_value_ = true;
    double value_ = 0.0;

    static std::string wrap_operand(const Sym& sym);
};

} // namespace ms
