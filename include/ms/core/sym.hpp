// MathScript Symbolic Expression Header
// CAS (Computer Algebra System) core

#pragma once

#include <string>

namespace ms {

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
    
    double eval() const;
    std::string to_string() const;
    
private:
    std::string expr_;
    bool has_value_ = true;
    double value_ = 0.0;
};

} // namespace ms