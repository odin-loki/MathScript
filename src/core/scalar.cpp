#include "ms/core/scalar.hpp"

namespace ms {

Scalar::Scalar(double value, std::string unit, std::string prefix)
    : value_(value), unit_(std::move(unit)), prefix_(std::move(prefix)) {}

Scalar Scalar::operator+(const Scalar& other) const {
    return Scalar(value_ + other.value_, unit_);
}

Scalar Scalar::operator-(const Scalar& other) const {
    return Scalar(value_ - other.value_, unit_);
}

Scalar Scalar::operator*(const Scalar& other) const {
    return Scalar(value_ * other.value_, unit_ + other.unit_);
}

Scalar Scalar::operator/(const Scalar& other) const {
    return Scalar(value_ / other.value_);
}

} // namespace ms
