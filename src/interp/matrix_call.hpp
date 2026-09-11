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


/// 2^64, exactly representable as a double and one past the last value a `uint64_t`
/// holds. A double is convertible to `uint64_t` exactly when it is `>= 0` and `<` this.
constexpr double kTwoPow64 = 18446744073709551616.0;

/// The largest double that is also a `uint64_t`. Doubles are spaced 2^11 apart up here,
/// so it is 2^64 - 2048 and not 2^64 - 1: `kTwoPow64 - 1.0` rounds straight back to
/// `kTwoPow64` and would clamp to a value the destination cannot hold.
constexpr double kMaxU64AsDouble = 18446744073709549568.0;
static_assert(kMaxU64AsDouble < kTwoPow64);

/// A non-negative integer argument the command takes as a `uint64_t`.
///
/// `static_cast<uint64_t>` of a double outside `[0, 2^64)` is undefined behaviour in
/// exactly the way `static_cast<int>` is, and the guard every one of these sites used --
///
///     if (arg < 0.0 || std::floor(arg) != arg) { /* reject */ }
///     ... static_cast<uint64_t>(arg) ...
///
/// tests neither end of that range. It rejects negatives and fractions and then converts
/// anything else, including 1e300. `numthy_sum_divisors(18446744073709551615)` answered
/// `0`: the literal is not representable as a double and rounds UP to exactly 2^64, one
/// past the last value the destination holds, so the conversion had nothing to return
/// and the REPL printed whatever it produced as though it were sigma.
///
/// `max_value` is the largest value this particular command can answer for. It is usually
/// a WORK bound rather than a representability one -- `numthy_prime_nth` can name the
/// nth prime long after it stops being able to find it -- so it is the caller's to pass,
/// and it is clamped to the representable range here so no caller can widen it by
/// mistake.
inline Result<std::uint64_t> checked_u64_argument(const std::string& fn, const char* what,
                                                  double value, double max_value) {
    if (!std::isfinite(value) || value != std::floor(value)) {
        return std::unexpected(
            DomainError{fn, std::string("expected an integer ") + what});
    }
    if (value < 0.0) {
        return std::unexpected(
            DomainError{fn, std::string("expected non-negative integer ") + what});
    }
    const double limit = max_value < kMaxU64AsDouble ? max_value : kMaxU64AsDouble;
    if (value > limit) {
        return std::unexpected(DomainError{
            fn, std::string(what) + " " + describe_count(value) +
                    " is too large; this command is bounded at " + describe_count(limit)});
    }
    return static_cast<std::uint64_t>(value);
}

/// The largest `unsigned`, exact as a double. Also the ceiling for anything stored as a
/// 32-bit half of a wider value.
constexpr double kMaxU32AsDouble = 4294967295.0;

/// A random seed, which is an `unsigned` and so has its own range.
///
/// The same cast-before-check shape as everywhere else, and here it does not merely
/// admit nonsense, it quietly makes two different seeds the same seed:
///
///     finance_mc_european_call(100,100,1,0.05,0.2,1000,42)          10.799620
///     finance_mc_european_call(100,100,1,0.05,0.2,1000,4294967296)  10.757478
///     finance_mc_european_call(100,100,1,0.05,0.2,1000,1e300)       10.757478
///
/// The last two agree because neither conversion had a value to produce; a user varying
/// the seed to see the Monte Carlo error would have been reading one sample twice.
inline Result<unsigned> checked_seed_argument(const std::string& fn, const char* what,
                                              double value) {
    if (!std::isfinite(value) || value != std::floor(value) || value < 0.0) {
        return std::unexpected(
            DomainError{fn, std::string("expected non-negative integer ") + what});
    }
    if (value > kMaxU32AsDouble) {
        return std::unexpected(DomainError{
            fn, std::string(what) + " " + describe_count(value) +
                    " is too large; a seed is bounded at " + describe_count(kMaxU32AsDouble)});
    }
    return static_cast<unsigned>(value);
}

/// How long the REPL is willing to disappear for.
///
/// A command runs to completion. There is no interrupt, no progress bar, and no way to
/// take the prompt back, so a bound on a super-linear argument is really a bound on
/// TIME, and it has two halves that should not be confused with each other. This number
/// is the POLICY half: it is one number for the whole REPL, and it is a judgement about
/// what a user will sit through, not a fact about any command.
///
/// It is the same judgement `kMaxReplIntegerArgument` already encodes. That comment
/// justifies 1e7 by a duration -- "a three-term recurrence is about a tenth of a
/// second" -- which means 1e7 was never a bound on the argument's MAGNITUDE. It was a
/// bound on the work a linear command does per unit of it, read out in the argument's
/// own units because for a linear command the two coincide. For a quadratic command
/// they do not, and the linear reading is catastrophic:
/// `finance_binomial_call(S,K,T,r,sigma,1e7)` is a perfectly ordinary integer that asks
/// for 5e13 node visits, about twelve days.
constexpr double kMaxReplCommandWorkNanos = 2.5e8;  // a quarter of a second

/// The same policy for a command where a large argument is a REQUEST rather than a slip.
///
/// The distinction is in what the argument means, not in how patient anyone is feeling.
/// A binomial tree converges like 1/steps and is done by about a thousand: nobody types
/// `steps = 1000000` on purpose, so bounding it costs no one anything. A Monte Carlo
/// converges like 1/sqrt(n_paths), so a million paths is not a slip -- it is three digits
/// of accuracy, and it is the entire reason the command exists. Holding it to a quarter
/// of a second would take the command away rather than protect it, so the bound here is
/// only doing the one job that remains: keeping the answer finite.
///
/// It is deliberately NOT used for the time-stepping solvers, whose `steps` is just as
/// deliberate. They allocate one grid per step, so for them a longer time budget is also
/// a bigger allocation, and the thing being bounded is a 16 GB `std::bad_alloc` rather
/// than a wait.
constexpr double kMaxReplSimulationWorkNanos = 4e9;  // four seconds

/// An argument whose cost is `value^power` units of work, at `nanos_per_unit` each.
///
/// `nanos_per_unit` is the MEASUREMENT half, and it is the caller's to supply because it
/// is a fact about that command and nothing else. The numbers the callers pass differ by
/// a factor of seventy -- a binomial-tree node is a multiply-add, a Gauss-Bonnet grid
/// point is a numerical quadrature of a curvature tensor -- which is exactly why a single
/// shared cap on "quadratic arguments" would be wrong in both directions at once: tight
/// enough to cost the tree three decimal places, and still seventeen seconds for the
/// quadrature. Each number, and the run it was measured from, is recorded in
/// docs/PLAN_STATUS.md.
///
/// The derived cap is a machine-dependent number, as the 1e7 above always was. It is
/// chosen so the answer is wrong in the safe direction on a slower machine: a command
/// that takes a second instead of a quarter is still a command that came back.
inline Result<int> checked_superlinear_argument(const std::string& fn, const char* what,
                                                double value, int power,
                                                double nanos_per_unit,
                                                double budget_nanos = kMaxReplCommandWorkNanos) {
    if (!std::isfinite(value) || value != std::floor(value)) {
        return std::unexpected(
            DomainError{fn, std::string("expected an integer ") + what});
    }
    const double units = budget_nanos / nanos_per_unit;
    const auto cap = static_cast<double>(
        static_cast<long long>(std::pow(units, 1.0 / static_cast<double>(power))));
    if (std::abs(value) > cap) {
        return std::unexpected(DomainError{
            fn, std::string(what) + " " + describe_count(std::abs(value)) +
                    " is too large; this command does work proportional to " + what +
                    "^" + std::to_string(power) + ", so it is bounded at " +
                    describe_count(cap) + " rather than at " +
                    describe_count(kMaxReplIntegerArgument)});
    }
    return static_cast<int>(value);
}

/// The step count of a fixed-step ODE solver, bounded by the trajectory it keeps.
///
/// These solvers return every step, and the REPL prints that as a `steps + 1` by (at
/// least) two matrix. So a step count past half the matrix cap is work done for an answer
/// that could never be shown: `ode_backward_euler("-50*y", 0, 1, 1, 200000)` integrated
/// for 0.2 s and was then refused for being 400002 elements, and the linear cap admits
/// 1e7 of them -- two minutes to reach the same refusal.
///
/// The range is decided on the double. `static_cast<int>` of a value outside int's range
/// is undefined behaviour rather than a wrap, so a guard written on the result of the cast
/// is reading a value the standard no longer accounts for.
inline Result<int> checked_ode_trajectory_steps(const std::string& fn, double steps) {
    if (!std::isfinite(steps) || steps != std::floor(steps) || steps < 0.0) {
        return std::unexpected(DomainError{fn, "expected non-negative integer steps"});
    }
    if (!repl_elems_allowed(static_cast<std::size_t>(
                                steps > 4.0e9 ? 4.0e9 : steps) + 1, 2)) {
        return std::unexpected(DomainError{
            fn, "steps " + describe_count(steps) +
                    " is too large; the trajectory is one row per step and is limited to " +
                    std::to_string(kMaxReplMatrixElems) + " elements"});
    }
    return static_cast<int>(steps);
}

/// The dimension of an internal matrix that MORE THAN ONE argument sizes.
///
/// This replaces a guard that bounded one parameter's magnitude with Mathieu's rule --
/// `ceil(sqrt(|q|))` -- baked in. That was the wrong shape for the three spheroidal
/// commands it was also applied to, and wrong by a SQUARE ROOT, because their dimension
/// is
///
///     spheroidal:  (n - m) / 2 + 22 + ceil(|c|)
///     Mathieu:     max(24, n + 16 + ceil(sqrt(|q|)))
///
/// so the square-root reading admitted `spheroidal_lambda(0, 0, 1e10)` -- a ten-billion
/// entry tridiagonal -- and all three of `spheroidal_lambda`, `spheroidal_s1` and
/// `spheroidal_s2` were measured ABORTING the process on it, which is the failure the
/// guard had been added to prevent. Neither family bounded the ORDER either, and that is
/// a dimension too: `spheroidal_lambda(1e7, 0, 0)` was still running at 25 s.
///
/// So the dimension is computed by the caller, from that command's own rule, and only
/// the bound lives here. `how` spells the rule out in the message, because a user who is
/// told a number without being told which argument produced it cannot act on it.
inline Result<double> checked_internal_matrix_dimension(const std::string& fn,
                                                        double dimension,
                                                        const std::string& how) {
    if (!std::isfinite(dimension)) {
        return std::unexpected(DomainError{fn, "expected finite arguments"});
    }
    if (dimension > static_cast<double>(kMaxReplMatrixElems)) {
        return std::unexpected(DomainError{
            fn, "the arguments size an internal matrix at about " +
                    describe_count(std::ceil(dimension)) + " (" + how +
                    "), and this command is limited to " +
                    std::to_string(kMaxReplMatrixElems) + " elements"});
    }
    return dimension;
}

/// The dimension of Mathieu's characteristic matrix, spelled as `special.cpp` spells it:
/// `max(24, n + 16 + ceil(sqrt(|q|)))`. The order is part of it, and used not to be
/// bounded at all -- `mathieu_a(1e7, 1)` built a ten-million-entry tridiagonal and took
/// 1.9 s to do it.
inline Result<double> checked_mathieu_dimension(const std::string& fn, double n, double q) {
    if (!std::isfinite(n) || !std::isfinite(q)) {
        return std::unexpected(DomainError{fn, "expected finite n and q"});
    }
    const double dim = std::max(24.0, std::abs(n) + 16.0 + std::ceil(std::sqrt(std::abs(q))));
    return checked_internal_matrix_dimension(fn, dim, "n + 16 + ceil(sqrt(|q|))");
}

/// The same for the spheroidal family, whose rule is `(n - m)/2 + 22 + ceil(|c|)` -- linear
/// in `c` where Mathieu's is a square root of `q`, which is the whole reason these are two
/// functions and not one.
inline Result<double> checked_spheroidal_dimension(const std::string& fn, double n, double m,
                                                   double c) {
    if (!std::isfinite(n) || !std::isfinite(m) || !std::isfinite(c)) {
        return std::unexpected(DomainError{fn, "expected finite n, m and c"});
    }
    const double index = std::floor((n - m) / 2.0);
    const double dim = (index < 0.0 ? 0.0 : index) + 22.0 + std::ceil(std::abs(c));
    return checked_internal_matrix_dimension(fn, dim, "(n - m)/2 + 22 + ceil(|c|)");
}

/// How many simplices a complex over `n_points` can hold at `max_dim`, as a double.
///
/// The enumerations in `topo` are written as nested loops with a distance test in each
/// one, so the cost and the result are the same number: every tuple that passes becomes a
/// row. When every distance is inside the radius -- `zeros(n, n)` is exactly that -- the
/// count is the full sum of binomials, and it is the honest worst case rather than a
/// pessimistic one.
///
/// Summed in a loop that stops as soon as it is past the cap, so a large `n` cannot make
/// the product itself overflow on the way to being rejected.
inline double simplex_count_upper_bound(std::size_t n_points, int max_dim, double give_up) {
    const double n = static_cast<double>(n_points);
    // Every `topo` enumeration here stops at tetrahedra; a larger max_dim adds nothing.
    const int top = max_dim < 0 ? 0 : (max_dim > 3 ? 3 : max_dim);
    double total = 0.0;
    double choose = 1.0;
    for (int j = 1; j <= top + 1; ++j) {
        // C(n, j) = C(n, j-1) * (n - j + 1) / j
        choose *= (n - static_cast<double>(j) + 1.0) / static_cast<double>(j);
        if (choose < 0.0) {
            choose = 0.0;
        }
        total += choose;
        if (total > give_up) {
            return total;
        }
    }
    return total;
}

/// Bounds a simplicial enumeration by what its answer could hold.
///
/// `topo_vietoris_rips(zeros(200, 200), 1, 2)` spent twenty seconds building 1.3 million
/// simplices and was then refused by the matrix cap, which is the shape the bignum
/// commands had: work for an answer that could never be shown. The rows are simplices and
/// each is three wide, so the count that fits is `kMaxReplMatrixElems / 3`, and moving the
/// same verdict to the front of the command costs the twenty seconds nothing.
///
/// Bounding the RESULT also bounds the work, which is why there is no separate time
/// budget here: 87381 simplices were measured at 1.4 s for max_dim 2 and 2.8 s for
/// max_dim 3, both inside `kMaxReplSimulationWorkNanos`.
inline Result<int> checked_simplex_dimension(const std::string& fn, std::size_t n_points,
                                             int max_dim) {
    if (max_dim < 0) {
        return std::unexpected(
            DomainError{fn, "expected non-negative integer max_dim"});
    }
    const double allowed = static_cast<double>(kMaxReplMatrixElems) / 3.0;
    const double count = simplex_count_upper_bound(n_points, max_dim, allowed);
    if (count > allowed) {
        return std::unexpected(DomainError{
            fn, "max_dim " + std::to_string(max_dim) + " over " +
                    describe_count(static_cast<double>(n_points)) +
                    " points can enumerate " + describe_count(count) +
                    " simplices, and the result is one row of three per simplex, which is "
                    "limited to " + describe_count(allowed) + " rows"});
    }
    return max_dim;
}

/// A budget for a command whose cost is a PRODUCT rather than a power of one argument: a
/// time-stepping solver that does `steps` sweeps of a grid, a Monte Carlo that walks
/// `n_paths` paths of `n_steps` each.
///
/// No single factor in such a command looks wrong, which is exactly the problem.
/// `checked_int_argument` bounds each of them at 1e7 on its own and says nothing about
/// the two together, so the product it admits is 1e14. This is `ExtentBudget`'s shape --
/// multiply the factors as they are read, charge them against one budget -- applied to
/// work instead of to elements.
///
/// For the solvers it is both at once, because they keep the whole trajectory: one grid
/// per step, of which the REPL reads only `.back()`. So
/// `pde_heat_1d(ones(200,1), 0.1, 0.1, 0.001, 10000000)` asks for a 16 GB history to
/// return 200 numbers, and under `-fno-exceptions` that is not a slow command but a dead
/// process -- the `std::bad_alloc` out of `std::vector` reaches `std::terminate` and the
/// REPL is gone without printing anything. Seven commands were measured aborting exactly
/// so, at `steps` = 1e7, which the linear cap admits: `pde_heat_1d`, `pde_heat_1d_cn`,
/// `pde_advection_1d`, `pde_advection_1d_lax_wendroff`, `pde_reaction_diffusion_1d`,
/// `pde_heat_2d` and `pde_wave_2d`. `pde_heat_2d_cn_adi` was still running at 35 s.
///
/// None of them was reachable by the existing oversized-argument sweep, and the reason
/// is worth stating plainly: that sweep probes 3000000000 and 1e18, which the linear cap
/// REJECTS. A sweep made of values the guard turns away cannot find a command that dies
/// on a value the guard lets through.
class WorkBudget {
public:
    /// `nanos_per_unit` is the measured cost of one unit of the product -- one cell of
    /// one sweep, one step of one path. See `checked_superlinear_argument` above for why
    /// this is the caller's number and not a shared one.
    /// `fn` is taken by value so callers can pass the dispatcher's own `fn` /
    /// `assign.callee` rather than retyping the command name as a literal. Ten of these
    /// call sites sit in fall-through branches of a multi-command `if`, where the name
    /// nearest above them in the file is a DIFFERENT command -- one literal was wrong
    /// that way on the first pass.
    WorkBudget(std::string fn, double nanos_per_unit,
               double budget_nanos = kMaxReplCommandWorkNanos)
        : fn_(std::move(fn)), remaining_(budget_nanos / nanos_per_unit) {}

    /// A factor that is not an argument: the extent of a matrix the command will sweep.
    /// Charging it before reading the arguments makes the bound on them shrink as the
    /// data grows, which is the relationship that actually holds -- a hundred steps of a
    /// large grid costs what ten thousand steps of a small one does.
    void charge(std::size_t units) {
        remaining_ /= (units == 0 ? 1.0 : static_cast<double>(units));
    }

    /// An argument that enters the product TWICE, because what it names is square: a
    /// `value` by `value` filter kernel visits `value^2` cells at every pixel.
    ///
    /// It cannot be spelled as two `take` calls. The first would compare `value` against
    /// a budget that has not yet been charged for the second, so a kernel whose SQUARE is
    /// past the budget passes the only check that looks at it.
    Result<int> take_square(const char* what, double value) {
        if (!std::isfinite(value) || value != std::floor(value) || value < 0.0) {
            return std::unexpected(
                DomainError{fn_, std::string("expected non-negative integer ") + what});
        }
        if (value * value > remaining_) {
            return std::unexpected(DomainError{
                fn_, std::string(what) + " " + describe_count(value) +
                         " is too large; this command does work proportional to " + what +
                         "^2 times the size of what it is given, and for this input it is "
                         "bounded at " + describe_count(std::floor(std::sqrt(remaining_)))});
        }
        remaining_ /= (value == 0.0 ? 1.0 : value * value);
        return static_cast<int>(value);
    }

    Result<int> take(const char* what, double value) {
        if (!std::isfinite(value) || value != std::floor(value)) {
            return std::unexpected(
                DomainError{fn_, std::string("expected an integer ") + what});
        }
        if (value < 0.0) {
            return std::unexpected(
                DomainError{fn_, std::string("expected non-negative integer ") + what});
        }
        if (value > remaining_) {
            return std::unexpected(DomainError{
                fn_, std::string(what) + " " + describe_count(value) +
                         " is too large; this command does work proportional to " + what +
                         " times the size of what it is given, and for this input it is "
                         "bounded at " + describe_count(std::floor(remaining_))});
        }
        remaining_ /= (value == 0.0 ? 1.0 : value);
        return static_cast<int>(value);
    }

private:
    std::string fn_;
    double remaining_;
};

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

    /// The order of a dense matrix the command assembles from the extents already taken,
    /// charged as a SECOND factor because what gets allocated is its square.
    ///
    /// `kMaxReplMatrixElems` bounds the RESULT, and the result of a FEM solve is a vector
    /// of `order` values. The stiffness matrix it solves through is `order` by `order`,
    /// and that is what is actually allocated -- densely, in `assemble_stiffness_1d` and
    /// its 2-D and 3-D siblings. So `fem_poisson1d(262144)` sits exactly ON the element
    /// budget and asks for 550 GB, which under `-fno-exceptions` is a dead process rather
    /// than a diagnostic; it was measured aborting at that value with the extent guard
    /// already in place.
    ///
    /// Which is the same sentence as "a cap on each extent is not a cap on the
    /// allocation", one level further out: **a cap on the result is not a cap on the
    /// working set.**
    Result<std::size_t> charge_dense_order(const char* what, std::size_t order) {
        if (order > remaining_) {
            return std::unexpected(DomainError{
                fn_, std::string(what) + " gives a dense " + std::to_string(order) +
                         " by " + std::to_string(order) + " matrix, and this command "
                         "is limited to " + std::to_string(kMaxReplMatrixElems) +
                         " elements"});
        }
        remaining_ /= (order == 0 ? 1 : order);
        return order;
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
    ///
    /// The range is decided on the double. `static_cast<int>(value)` first -- which is
    /// how this read -- is undefined for anything outside `int`, so `i < 1` was testing a
    /// value that the standard does not say exists. It is the same defect the free
    /// `checked_int_argument` above was written for, in the helper 485 generated handlers
    /// call.
    static Result<std::size_t> parse_positive_size_arg(double value, const char* fn,
                                                       const char* label) {
        if (!std::isfinite(value) || value < 1.0 || value != std::floor(value) ||
            value > kMaxReplIntegerArgument) {
            return std::unexpected(DomainError{fn, label});
        }
        return static_cast<std::size_t>(value);
    }

    /// A non-negative whole number.
    ///
    /// The guard rejected negatives and fractions and then converted whatever was left,
    /// so 1e300 reached `static_cast<std::uint64_t>` -- undefined, and in practice a
    /// fabricated answer rather than a diagnostic.
    static Result<std::uint64_t> parse_uint64_arg(double value, const char* fn,
                                                  const char* label) {
        if (!std::isfinite(value) || value < 0.0 || value != std::floor(value) ||
            !(value < kTwoPow64)) {
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
