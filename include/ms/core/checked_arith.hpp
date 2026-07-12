// MathScript Checked Arithmetic
// Overflow-safe, saturating, and wrapping numeric utilities

#pragma once

#include "ms/error/error_types.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace ms {

namespace detail {

template<typename T>
inline constexpr bool is_std_arithmetic_v =
    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template<typename T>
Result<T> overflow(std::string_view op) {
    return std::unexpected(Error{OverflowError{op}});
}

template<typename T>
Result<T> domain_error(std::string_view fn, std::string_view reason) {
    return std::unexpected(Error{DomainError{std::string(fn), std::string(reason)}});
}

template<typename T>
  requires(std::is_integral_v<T>)
bool add_would_overflow(T a, T b) {
    if constexpr (sizeof(T) < 8) {
        if constexpr (std::is_signed_v<T>) {
            const int64_t sum =
                static_cast<int64_t>(a) + static_cast<int64_t>(b);
            return sum < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
                   sum > static_cast<int64_t>(std::numeric_limits<T>::max());
        } else {
            const uint64_t sum =
                static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
            return sum > static_cast<uint64_t>(std::numeric_limits<T>::max());
        }
    } else if constexpr (std::is_signed_v<T>) {
        return (b > 0 && a > std::numeric_limits<T>::max() - b) ||
               (b < 0 && a < std::numeric_limits<T>::min() - b);
    } else {
        return a > std::numeric_limits<T>::max() - b;
    }
}

template<typename T>
  requires(std::is_integral_v<T>)
bool sub_would_overflow(T a, T b) {
    if constexpr (sizeof(T) < 8) {
        if constexpr (std::is_signed_v<T>) {
            const int64_t diff =
                static_cast<int64_t>(a) - static_cast<int64_t>(b);
            return diff < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
                   diff > static_cast<int64_t>(std::numeric_limits<T>::max());
        } else {
            return a < b;
        }
    } else if constexpr (std::is_signed_v<T>) {
        return (b < 0 && a > std::numeric_limits<T>::max() + b) ||
               (b > 0 && a < std::numeric_limits<T>::min() + b);
    } else {
        return a < b;
    }
}

template<typename T>
  requires(std::is_integral_v<T>)
bool mul_would_overflow(T a, T b) {
    if (a == 0 || b == 0) {
        return false;
    }

    if constexpr (sizeof(T) < 8) {
        if constexpr (std::is_signed_v<T>) {
            const int64_t product =
                static_cast<int64_t>(a) * static_cast<int64_t>(b);
            return product < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
                   product > static_cast<int64_t>(std::numeric_limits<T>::max());
        } else {
            const uint64_t product =
                static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
            return product > static_cast<uint64_t>(std::numeric_limits<T>::max());
        }
    } else if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min()) {
            return b != 0 && b != 1;
        }
        if (b == std::numeric_limits<T>::min()) {
            return a != 0 && a != 1;
        }

        if (a > 0) {
            if (b > 0) {
                return a > std::numeric_limits<T>::max() / b;
            }
            return b < std::numeric_limits<T>::min() / a;
        }

        if (b > 0) {
            return a < std::numeric_limits<T>::min() / b;
        }
        return b < std::numeric_limits<T>::max() / a;
    } else {
        return a > std::numeric_limits<T>::max() / b;
    }
}

template<typename T>
  requires(std::is_integral_v<T>)
T saturating_add_value(T a, T b) {
    if (!add_would_overflow(a, b)) {
        return static_cast<T>(a + b);
    }
    if constexpr (std::is_unsigned_v<T>) {
        return std::numeric_limits<T>::max();
    } else {
        return (b > 0) ? std::numeric_limits<T>::max()
                       : std::numeric_limits<T>::min();
    }
}

template<typename T>
  requires(std::is_integral_v<T>)
T saturating_sub_value(T a, T b) {
    if (!sub_would_overflow(a, b)) {
        return static_cast<T>(a - b);
    }
    if constexpr (std::is_unsigned_v<T>) {
        return std::numeric_limits<T>::min();
    } else {
        return (b > 0) ? std::numeric_limits<T>::min()
                       : std::numeric_limits<T>::max();
    }
}

template<typename T>
  requires(std::is_integral_v<T>)
T saturating_mul_value(T a, T b) {
    if (!mul_would_overflow(a, b)) {
        return static_cast<T>(a * b);
    }
    if constexpr (std::is_unsigned_v<T>) {
        return std::numeric_limits<T>::max();
    } else {
        return ((a > 0) == (b > 0)) ? std::numeric_limits<T>::max()
                                    : std::numeric_limits<T>::min();
    }
}

template<typename T>
  requires(std::is_floating_point_v<T>)
bool float_invalid(T value) {
    return std::isnan(value) || std::isinf(value);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Checked arithmetic
// ---------------------------------------------------------------------------

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_add(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        if (detail::add_would_overflow(a, b)) {
            return detail::overflow<T>("add");
        }
        return static_cast<T>(a + b);
    } else {
        const T result = a + b;
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("add");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_sub(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        if (detail::sub_would_overflow(a, b)) {
            return detail::overflow<T>("sub");
        }
        return static_cast<T>(a - b);
    } else {
        const T result = a - b;
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("sub");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_mul(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        if (detail::mul_would_overflow(a, b)) {
            return detail::overflow<T>("mul");
        }
        return static_cast<T>(a * b);
    } else {
        const T result = a * b;
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("mul");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_div(T a, T b) {
    if (b == T{0}) {
        return detail::domain_error<T>("div", "division by zero");
    }

    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
            if (a == std::numeric_limits<T>::min() && b == T{-1}) {
                return detail::overflow<T>("div");
            }
        }
        return static_cast<T>(a / b);
    } else {
        const T result = a / b;
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("div");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_mod(T a, T b) {
    if (b == T{0}) {
        return detail::domain_error<T>("mod", "modulo by zero");
    }

    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
            if (a == std::numeric_limits<T>::min() && b == T{-1}) {
                return detail::overflow<T>("mod");
            }
        }
        return static_cast<T>(a % b);
    } else {
        const T result = std::fmod(a, b);
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("mod");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_neg(T a) {
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
            if (a == std::numeric_limits<T>::min()) {
                return detail::overflow<T>("neg");
            }
        }
        return static_cast<T>(-a);
    } else {
        const T result = -a;
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("neg");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_abs(T a) {
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
            if (a == std::numeric_limits<T>::min()) {
                return detail::overflow<T>("abs");
            }
            return a < T{0} ? static_cast<T>(-a) : a;
        } else {
            return a;
        }
    } else {
        const T result = std::abs(a);
        if (detail::float_invalid(result)) {
            return detail::overflow<T>("abs");
        }
        return result;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
Result<T> checked_pow(T base, unsigned exp) {
    if (exp == 0) {
        return T{1};
    }

    T result = base;
    for (unsigned i = 1; i < exp; ++i) {
        auto next = checked_mul(result, base);
        if (!next) {
            return next;
        }
        result = *next;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Saturating arithmetic
// ---------------------------------------------------------------------------

template<typename T>
  requires detail::is_std_arithmetic_v<T>
T saturating_add(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        return detail::saturating_add_value(a, b);
    } else {
        const auto result = checked_add(a, b);
        if (result) {
            return *result;
        }
        return (b >= T{0}) ? std::numeric_limits<T>::max()
                           : std::numeric_limits<T>::lowest();
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
T saturating_sub(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        return detail::saturating_sub_value(a, b);
    } else {
        const auto result = checked_sub(a, b);
        if (result) {
            return *result;
        }
        return (b >= T{0}) ? std::numeric_limits<T>::lowest()
                           : std::numeric_limits<T>::max();
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
T saturating_mul(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        return detail::saturating_mul_value(a, b);
    } else {
        const auto result = checked_mul(a, b);
        if (result) {
            return *result;
        }
        return ((a >= T{0}) == (b >= T{0})) ? std::numeric_limits<T>::max()
                                             : std::numeric_limits<T>::lowest();
    }
}

// ---------------------------------------------------------------------------
// Wrapping arithmetic
// ---------------------------------------------------------------------------

template<typename T>
  requires detail::is_std_arithmetic_v<T>
T wrapping_add(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_unsigned_v<T>) {
            return static_cast<T>(a + b);
        } else {
            using U = std::make_unsigned_t<T>;
            return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
        }
    } else {
        return a + b;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
T wrapping_sub(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_unsigned_v<T>) {
            return static_cast<T>(a - b);
        } else {
            using U = std::make_unsigned_t<T>;
            return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
        }
    } else {
        return a - b;
    }
}

template<typename T>
  requires detail::is_std_arithmetic_v<T>
T wrapping_mul(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_unsigned_v<T>) {
            return static_cast<T>(a * b);
        } else {
            using U = std::make_unsigned_t<T>;
            return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
        }
    } else {
        return a * b;
    }
}

// ---------------------------------------------------------------------------
// Float introspection helpers
// ---------------------------------------------------------------------------

template<typename T>
  requires std::is_floating_point_v<T>
bool is_nan(T x) {
    return std::isnan(x);
}

template<typename T>
  requires std::is_floating_point_v<T>
bool is_inf(T x) {
    return std::isinf(x);
}

template<typename T>
  requires std::is_floating_point_v<T>
bool is_finite(T x) {
    return std::isfinite(x);
}

template<typename T>
  requires std::is_floating_point_v<T>
bool is_normal(T x) {
    return std::isnormal(x);
}

template<typename T>
  requires std::is_floating_point_v<T>
bool signbit(T x) {
    return std::signbit(x);
}

template<typename T>
  requires std::is_floating_point_v<T>
T ulp(T x) {
    if (std::isnan(x) || std::isinf(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == T{0}) {
        return std::nextafter(T{0}, std::numeric_limits<T>::infinity()) - T{0};
    }
    return std::nextafter(x, std::numeric_limits<T>::infinity()) - x;
}

template<typename T>
  requires std::is_floating_point_v<T>
T nextafter(T x, T toward) {
    if (std::isnan(x) || std::isinf(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isnan(toward)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return std::nextafter(x, toward);
}

template<typename T>
  requires std::is_floating_point_v<T>
constexpr T eps() {
    return std::numeric_limits<T>::epsilon();
}

template<typename T>
  requires std::is_floating_point_v<T>
constexpr T huge() {
    return std::numeric_limits<T>::max();
}

template<typename T>
  requires std::is_floating_point_v<T>
constexpr T tiny() {
    return std::numeric_limits<T>::min();
}

// ---------------------------------------------------------------------------
// Cast helpers
// ---------------------------------------------------------------------------

template<typename T, typename U>
  requires detail::is_std_arithmetic_v<T> && detail::is_std_arithmetic_v<U>
Result<T> narrow(U x) {
    if constexpr (std::is_floating_point_v<U>) {
        if (std::isnan(x)) {
            return detail::overflow<T>("narrow");
        }
        if (std::isinf(x)) {
            return detail::overflow<T>("narrow");
        }
    }

    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_integral_v<U>) {
            if (x < static_cast<U>(std::numeric_limits<T>::min()) ||
                x > static_cast<U>(std::numeric_limits<T>::max())) {
                return detail::overflow<T>("narrow");
            }
        } else {
            if (x < static_cast<U>(std::numeric_limits<T>::min()) ||
                x > static_cast<U>(std::numeric_limits<T>::max())) {
                return detail::overflow<T>("narrow");
            }
        }
    } else {
        if constexpr (std::is_integral_v<U>) {
            if (static_cast<long double>(x) > static_cast<long double>(std::numeric_limits<T>::max()) ||
                static_cast<long double>(x) < static_cast<long double>(std::numeric_limits<T>::lowest())) {
                return detail::overflow<T>("narrow");
            }
        }
    }

    return static_cast<T>(x);
}

template<typename T, typename U>
  requires detail::is_std_arithmetic_v<T> && detail::is_std_arithmetic_v<U>
T widen(U x) {
    return static_cast<T>(x);
}

} // namespace ms
