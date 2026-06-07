// MathScript Symbolic Expression Implementation

#include "ms/core/sym.hpp"

namespace ms {

Sym::Sym(const char* expr) {
    expr_ = expr;
    has_value_ = false;
}

Sym::Sym(double value) {
    value_ = value;
    expr_ = "";
    has_value_ = true;
}

Sym& Sym::operator+=(const Sym& other) {
    expr_ += "+" + other.expr_;
    return *this;
}

Sym& Sym::operator-=(const Sym& other) {
    expr_ += "-" + other.expr_;
    return *this;
}

Sym& Sym::operator*=(const Sym& other) {
    expr_ += "*" + other.expr_;
    return *this;
}

Sym& Sym::operator/=(const Sym& other) {
    expr_ += "/" + other.expr_;
    return *this;
}

Sym Sym::operator+(const Sym& other) const {
    Sym result = *this;
    result += other;
    return result;
}

Sym Sym::operator-(const Sym& other) const {
    Sym result = *this;
    result -= other;
    return result;
}

Sym Sym::operator*(const Sym& other) const {
    Sym result = *this;
    result *= other;
    return result;
}

Sym Sym::operator/(const Sym& other) const {
    Sym result = *this;
    result /= other;
    return result;
}

double Sym::eval() const {
    if (has_value_) return value_;
    // Simplified - would parse and evaluate expression
    return 0.0;
}

std::string Sym::to_string() const {
    return expr_;
}

} // namespace ms