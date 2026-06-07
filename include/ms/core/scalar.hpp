// MathScript Scalar Type
// Scalar with physical units

#pragma once

#include <string>

namespace ms {

class Scalar {
public:
    Scalar(double value = 0.0, std::string unit = "", std::string prefix = "");
    
    double value() const { return value_; }
    std::string unit() const { return unit_; }
    std::string full_unit() const { return prefix_ + unit_; }
    
    Scalar operator+(const Scalar& other) const;
    Scalar operator-(const Scalar& other) const;
    Scalar operator*(const Scalar& other) const;
    Scalar operator/(const Scalar& other) const;
    
private:
    double value_;
    std::string unit_;
    std::string prefix_;
};

} // namespace ms