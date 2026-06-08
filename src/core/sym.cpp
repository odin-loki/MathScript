// MathScript Symbolic Expression Implementation

#include <cstdlib>
#include <limits>

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
    if (has_value_) {
        return value_;
    }
    if (expr_.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    char* end = nullptr;
    const double parsed = std::strtod(expr_.c_str(), &end);
    if (end == expr_.c_str() + expr_.size()) {
        return parsed;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

std::string Sym::to_string() const {
    return expr_;
}

} // namespace ms