// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include "ms/interp/repl_engine.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>

namespace ms::interp {

// Declared here so MatrixCallCtx's argument helpers can use them without this
// header pulling in repl_engine_internal.hpp, which every handler already
// includes for its own body. Definitions live in repl_engine_internal.cpp.
namespace detail {
bool parse_number(const std::string& text, double& value);
Result<double> eval_scalar_expr(const SessionState& state, const std::string& expr_text);
} // namespace detail

constexpr std::size_t kMaxReplMatrixElems = 262144;
inline const char* kReplMatrixTooLarge = "matrix too large (max 262144 elements)";

inline bool repl_elems_allowed(std::size_t rows, std::size_t cols) {
    return rows == 0 || cols == 0 ||
           (rows <= kMaxReplMatrixElems && cols <= kMaxReplMatrixElems &&
            rows <= kMaxReplMatrixElems / cols);
}

inline bool repl_dims_allowed(double m_d, double n_d, std::size_t& rows, std::size_t& cols) {
    if (!(m_d >= 1.0) || !(n_d >= 1.0) || m_d != std::floor(m_d) || n_d != std::floor(n_d)) {
        return false;
    }
    if (m_d > static_cast<double>(kMaxReplMatrixElems) ||
        n_d > static_cast<double>(kMaxReplMatrixElems)) {
        return false;
    }
    rows = static_cast<std::size_t>(m_d);
    cols = static_cast<std::size_t>(n_d);
    return repl_elems_allowed(rows, cols);
}

/// How large an integer argument may be before the REPL presumes a mistake rather than
/// a request.
///
/// `repl_engine_internal.cpp` has used this number for a matrix index and a matrix count
/// since the accessors were added, with the reasoning that any index this large is out
/// of range for every matrix the REPL can hold and that stopping here keeps the
/// conversion below well defined. A special-function ORDER is the third kind of argument
/// it applies to, for the second half of that reason and not the first.
constexpr double kMaxReplIntegerArgument = 1e7;

/// An exact non-negative integral double, written the way a count is written.
/// `format_scalar` would render 100000000 as "100000000.000000", which is not how the
/// user typed it and not how a count reads.
inline std::string describe_count(double value) {
    if (value < 18446744073709551616.0) {  // 2^64, exact as a double
        return std::to_string(static_cast<unsigned long long>(value));
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

/// An integer argument that is not an extent -- a special-function order, a
/// combinatorial `n`, a branch index. Nothing is allocated per unit of it and there is
/// no product to charge it against, so `ExtentBudget` is the wrong shape for it.
///
/// The conversion is the part that was undefined. Every one of these read
///
///     const int n = static_cast<int>(args[0]);
///     if (n < 0 || ...) { /* reject */ }
///
/// and `static_cast<int>` of a double outside `int`'s range is undefined behaviour
/// rather than a wrap: by the time the guard reads `n` there is no value there to test.
/// On x86-64 the conversion happens to yield INT_MIN, so `n < 0` rejected the input by
/// accident, in a way that reads exactly like a guard. The range is decided on the
/// double here, before any conversion.
///
/// The bound is a WORK bound rather than an accuracy one. `legendre_p(n, x)` is a
/// recurrence of one step per unit inside a REPL that has no way to interrupt one:
/// `legendre_p(1750000000, 0.5)` returns, and takes longer than the twenty seconds a
/// probe will wait for it. These recurrences still carry several correct digits well
/// past 1e7; what they do not do is finish. At 1e7 a three-term recurrence is about a
/// tenth of a second, and no order anyone writes down is within four orders of
/// magnitude of it.
///
/// Truncation is refused rather than performed. `combo_nchoosek(5.5, 2)` used to answer
/// as though 5 had been written.
inline Result<int> checked_int_argument(const std::string& fn, const char* what,
                                        double value) {
    if (!std::isfinite(value) || value != std::floor(value)) {
        return std::unexpected(
            DomainError{fn, std::string("expected an integer ") + what});
    }
    if (std::abs(value) > kMaxReplIntegerArgument) {
        return std::unexpected(DomainError{
            fn, std::string(what) + " " + describe_count(std::abs(value)) +
                    " is too large; an integer argument is bounded at " +
                    describe_count(kMaxReplIntegerArgument)});
    }
    return static_cast<int>(value);
}


/// Reads one size-like REPL argument -- a count, an order, a grid extent -- and charges
/// it against a budget of elements.
///
/// Three things go wrong with such an argument before any work starts, and the idiom
/// this replaces caught none of them:
///
///     const int n_i = static_cast<int>(n_d);
///     if (n_i < 0 || n_d != n_i) { ... }
///
///   - **The cast IS the check.** `static_cast<int>` of a double outside `int`'s range
///     is undefined behaviour, not a wrap, so by the time the guard reads `n_i` the
///     program has already left the standard behind. The range has to be decided on the
///     double.
///   - **A count that fits is not a count that is affordable.** `fem_poisson1d(1e8)` is
///     a perfectly good `int` and asks for 800 MB.
///   - **A cap on each extent alone is not a cap on the allocation.** `impad(A, 1e6)`
///     passes any per-dimension bound and then asks for four trillion elements, because
///     what gets allocated is the PRODUCT. So extents are multiplied as they are read.
///
/// The budget is `kMaxReplMatrixElems`, which is not a new number: it is the limit the
/// REPL already enforces on any matrix it stores. All this moves is *when* -- from after
/// the allocation, where the answer is a diagnostic about a matrix that has already been
/// built, to before it, where it is a diagnostic about the argument that asked for it.
///
/// It has to be before, because the library is built with `-fno-exceptions`: a
/// `std::bad_alloc` out of `std::vector` reaches `std::terminate` and the process is
/// gone without printing anything. There is no catching it afterwards.
class ExtentBudget {
public:
    explicit ExtentBudget(const char* fn) : fn_(fn) {}

    /// `what` names the argument the way the handler's own signature does, so that the
    /// message points at something the user can see in what they typed.
    Result<std::size_t> take(const char* what, double value) {
        if (!std::isfinite(value) || value != std::floor(value) || value < 0.0) {
            return std::unexpected(
                DomainError{fn_, std::string("expected non-negative integer ") + what});
        }
        if (value > static_cast<double>(remaining_)) {
            return std::unexpected(DomainError{
                fn_, std::string(what) + " " + describe_count(value) +
                         " is too large; the result is limited to " +
                         std::to_string(kMaxReplMatrixElems) + " elements"});
        }
        const auto taken = static_cast<std::size_t>(value);
        // A zero extent buys an empty result and costs nothing, so it must not divide.
        remaining_ /= (taken == 0 ? 1 : taken);
        return taken;
    }

private:
    const char* fn_;
    std::size_t remaining_ = kMaxReplMatrixElems;
};

struct MatrixCallCtx {
    Interpreter& interp;
    explicit MatrixCallCtx(Interpreter& i) : interp(i) {}
    Result<Matrix<double>> resolve_operand(const std::string& text) {
        return interp.eval_matrix_operand(text);
    }

    // The argument helpers below used to be stamped into every generated handler
    // as a local lambda, whether or not it used them: parse_uint64_arg was
    // defined in 480 translation units and called in one. They are identical in
    // every copy, so they belong here once.

    /// A numeric literal, or a scalar expression evaluated in the session.
    Result<double> parse_scalar_arg(const std::string& arg_text, const char* fn) {
        double value = 0.0;
        if (detail::parse_number(arg_text, value)) {
            return value;
        }
        auto expr = detail::eval_scalar_expr(state(), arg_text);
        if (!expr) {
            return std::unexpected(DomainError{fn, "expected numeric scalar argument"});
        }
        return *expr;
    }

    /// A dimension: a positive integer, exactly representable as the double it
    /// arrived as.
    static Result<std::size_t> parse_positive_size_arg(double value, const char* fn,
                                                       const char* label) {
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(i);
    }

    /// A non-negative whole number.
    static Result<std::uint64_t> parse_uint64_arg(double value, const char* fn,
                                                  const char* label) {
        if (value < 0.0 || value != std::floor(value)) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::uint64_t>(value);
    }
    SessionState& state() { return interp.state_; }
    const SessionState& state() const { return interp.state_; }
    auto& session_objects() { return interp.session_objects_; }
    const auto& session_objects() const { return interp.session_objects_; }
};

using MatrixCallHandler = Result<Matrix<double>> (*)(Interpreter&, const MatrixCallAssign&);

void register_matrix_call(std::string_view name, MatrixCallHandler handler);
void ensure_matrix_calls_registered();
bool is_matrix_call_callee(const std::string& callee);
Result<Matrix<double>> dispatch_matrix_call(Interpreter& interp, const MatrixCallAssign& assign);

} // namespace ms::interp
