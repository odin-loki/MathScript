// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/sym2/bridge.hpp"

#include <cmath>

namespace ms::sym2 {

namespace {

using bignum::BigInt;
using bignum::Rational;

/// The largest denominator a double is allowed to be recovered as. A literal anyone
/// types -- 0.5, 2.25, 0.001 -- is a fraction over a power of ten with a handful of
/// digits. Past this the double is almost certainly the result of a computation rather
/// than a written-down number, and pinning an exact fraction to it would be asserting a
/// precision it does not have.
constexpr double kMaxRecoveredDenominator = 1e9;

/// Recover the exact value a double stands for, when it stands for one.
///
/// The old core stored every number as a double, so the exactness this core exists to
/// provide has to be reconstructed on the way in. A double that is an integer, or a
/// short decimal, is converted exactly -- 0.25 becomes 1/4, not 0.25 -- and anything
/// else stays a Real. Multiplying by powers of ten and testing for an integer result is
/// what distinguishes the two: 0.25 * 100 is exactly 25, while 1/3 * 10^k never lands
/// on an integer.
ExprRef exact_or_real(double value) {
    if (!std::isfinite(value)) {
        if (std::isnan(value)) {
            return constant("nan");
        }
        return constant(value > 0.0 ? "inf" : "-inf");
    }
    if (value == std::floor(value) && std::abs(value) < 9.0e15) {
        return integer(static_cast<long long>(value));
    }
    double scale = 1.0;
    for (int digits = 1; digits <= 9; ++digits) {
        scale *= 10.0;
        const double scaled = value * scale;
        if (scaled == std::floor(scaled) && std::abs(scaled) < 9.0e15 &&
            scale <= kMaxRecoveredDenominator) {
            // Round-trip check: the fraction must be the same double it came from, or
            // it is a different number wearing the same printed form.
            const double back = static_cast<double>(static_cast<long long>(scaled)) / scale;
            if (back == value) {
                return rational(BigInt(static_cast<long long>(scaled)),
                                BigInt(static_cast<long long>(scale)));
            }
        }
    }
    return real(value);
}

SymExpr legacy_const(double value) { return sym_const(value); }

} // namespace

ExprRef from_legacy(const SymExpr& expr) {
    const auto left = [&expr]() -> ExprRef {
        return expr.left ? from_legacy(*expr.left) : undefined();
    };
    const auto right = [&expr]() -> ExprRef {
        return expr.right ? from_legacy(*expr.right) : undefined();
    };
    switch (expr.op) {
    case SymOp::Const:
        return exact_or_real(expr.value);
    case SymOp::Var:
        return symbol(expr.name);
    case SymOp::Add:
        return add({left(), right()});
    case SymOp::Sub:
        return sub(left(), right());
    case SymOp::Mul:
        return mul({left(), right()});
    case SymOp::Div:
        return div(left(), right());
    case SymOp::Neg:
        return neg(left());
    case SymOp::Pow:
        return pow(left(), right());
    case SymOp::Sin:
        return function("sin", {left()});
    case SymOp::Cos:
        return function("cos", {left()});
    case SymOp::Tan:
        return function("tan", {left()});
    case SymOp::Exp:
        return function("exp", {left()});
    case SymOp::Log:
        return function("log", {left()});
    case SymOp::Sqrt:
        return function("sqrt", {left()});
    case SymOp::Deriv:
        // The old API's "no closed form" sentinel. It always meant an unevaluated
        // derivative; here it is one.
        return derivative(left(), {symbol(expr.name)});
    }
    return undefined();
}

namespace {

SymExpr to_legacy_impl(const ExprRef& e, bool& lossy);

SymExpr legacy_product(const std::vector<ExprRef>& args, bool& lossy) {
    SymExpr result = to_legacy_impl(args[0], lossy);
    for (std::size_t i = 1; i < args.size(); ++i) {
        result = sym_mul(std::move(result), to_legacy_impl(args[i], lossy));
    }
    return result;
}

SymExpr to_legacy_impl(const ExprRef& e, bool& lossy) {
    if (!e) {
        lossy = true;
        return legacy_const(std::numeric_limits<double>::quiet_NaN());
    }
    switch (e->head) {
    case Head::Integer: {
        double value = 0.0;
        as_double(e, value);
        // A BigInt past 2^53 does not survive a double. Saying so is the point of the
        // flag: the old core cannot hold the value, and pretending otherwise is how a
        // wrong answer gets out.
        const bool exact_round_trip = std::abs(value) < 9.0e15;
        lossy = lossy || !exact_round_trip;
        return legacy_const(value);
    }
    case Head::Rational: {
        double value = 0.0;
        as_double(e, value);
        // 1/3 is not representable. This is the conversion §10.1 says makes
        // simplification impossible in the old core, and it happens here rather than
        // silently everywhere.
        lossy = true;
        return legacy_const(value);
    }
    case Head::Real: {
        double value = 0.0;
        as_double(e, value);
        return legacy_const(value);
    }
    case Head::Symbol:
        return sym_var(*std::get_if<std::string>(&e->atom));
    case Head::Constant: {
        const std::string& name = *std::get_if<std::string>(&e->atom);
        if (name == "pi") return legacy_const(3.14159265358979323846);
        if (name == "e") return legacy_const(2.71828182845904523536);
        if (name == "inf") return legacy_const(std::numeric_limits<double>::infinity());
        if (name == "-inf") return legacy_const(-std::numeric_limits<double>::infinity());
        lossy = true;
        return legacy_const(std::numeric_limits<double>::quiet_NaN());
    }
    case Head::Add: {
        SymExpr result = to_legacy_impl(e->args[0], lossy);
        for (std::size_t i = 1; i < e->args.size(); ++i) {
            result = sym_add(std::move(result), to_legacy_impl(e->args[i], lossy));
        }
        return result;
    }
    case Head::Mul:
        return legacy_product(e->args, lossy);
    case Head::Pow:
        return sym_pow(to_legacy_impl(e->args[0], lossy), to_legacy_impl(e->args[1], lossy));
    case Head::Function: {
        const std::string& name = *std::get_if<std::string>(&e->atom);
        if (e->args.size() == 1) {
            if (name == "sin") return sym_sin(to_legacy_impl(e->args[0], lossy));
            if (name == "cos") return sym_cos(to_legacy_impl(e->args[0], lossy));
            if (name == "tan") return sym_tan(to_legacy_impl(e->args[0], lossy));
            if (name == "exp") return sym_exp(to_legacy_impl(e->args[0], lossy));
            if (name == "log") return sym_log(to_legacy_impl(e->args[0], lossy));
            if (name == "sqrt") return sym_sqrt(to_legacy_impl(e->args[0], lossy));
        }
        // The old core has six function operators and no way to add a seventh, which
        // is §10.1's "single-argument functions only". An opaque variable keeps the
        // expression structurally intact and is honestly marked as a downgrade: the
        // old engine cannot differentiate or transform it.
        lossy = true;
        return sym_var(to_string(e));
    }
    case Head::Derivative:
        // Round-trips to the sentinel it came from.
        return sym_deriv(to_legacy_impl(e->args[0], lossy),
                         e->args.size() > 1 ? to_string(e->args[1]) : std::string{});
    case Head::Integral:
    case Head::Limit:
        lossy = true;
        return sym_var(to_string(e));
    }
    lossy = true;
    return legacy_const(std::numeric_limits<double>::quiet_NaN());
}

} // namespace

SymExpr to_legacy(const ExprRef& expr, bool* lossy) {
    bool degraded = false;
    SymExpr result = to_legacy_impl(expr, degraded);
    if (lossy != nullptr) {
        *lossy = degraded;
    }
    return result;
}

Result<ExprRef> parse(const std::string& text) {
    auto parsed = sym_parse(text);
    if (!parsed) {
        return std::unexpected(DomainError{"sym2::parse", parsed.error().message});
    }
    return from_legacy(*parsed);
}

} // namespace ms::sym2
