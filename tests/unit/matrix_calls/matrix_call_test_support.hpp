// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Shared assertions for the generated matrix-call dispatch tests.
//
// This header is hand-written and the per-domain translation units next to it
// are not: `scripts/gen_matrix_call_tests.py` emits one TEST per handler per
// property, and every one of them bottoms out in a function here. Keeping the
// assertions in one place means a change to what "correctly rejected" means is a
// change to three functions rather than to four thousand generated lines.
#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "matrix_call.hpp"
#include "ms/interp/repl_engine.hpp"

namespace ms::interp::testing {

// The error dispatch produces when nothing claims the call. Every handler
// pre-initialises its result to exactly this and overwrites it only after its
// guard has passed, so seeing it back means the guard rejected the call.
inline constexpr const char* kUnsupportedFn = "assign";
inline constexpr const char* kUnsupportedReason = "unsupported matrix call";

// Argument names chosen to be inert: not a defined variable, not parseable as a
// matrix literal, and not parseable as a nested matrix call. Anything that could
// be parsed as a call would send eval_matrix_operand back through the dispatcher
// and the test would be measuring recursion rather than the handler.
inline std::vector<std::string> undefined_args(std::size_t count) {
    std::vector<std::string> args;
    args.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        args.push_back("__ms_no_such_operand_" + std::to_string(i));
    }
    return args;
}

/// Dispatch with a real matrix literal everywhere except one argument.
///
/// A probe that makes every argument undefined only ever reaches the first
/// `resolve_operand`; the second and later propagations sit behind it, untested. A
/// literal is used rather than a session variable so the probe depends on nothing
/// but the parser.
inline Result<Matrix<double>> dispatch_with_bad_arg(const std::string& callee,
                                                    std::size_t arity,
                                                    std::size_t bad_index) {
    Interpreter interp;
    MatrixCallAssign assign;
    assign.target = "__ms_dispatch_target";
    assign.callee = callee;
    assign.args.reserve(arity);
    for (std::size_t i = 0; i < arity; ++i) {
        assign.args.push_back(i == bad_index ? "__ms_no_such_operand"
                                             : "[1, 2; 3, 4]");
    }
    return dispatch_matrix_call(interp, assign);
}

inline Result<Matrix<double>> dispatch(const std::string& callee, std::size_t arity) {
    Interpreter interp;
    MatrixCallAssign assign;
    assign.target = "__ms_dispatch_target";
    assign.callee = callee;
    assign.args = undefined_args(arity);
    return dispatch_matrix_call(interp, assign);
}

inline bool is_unsupported(const Result<Matrix<double>>& r) {
    if (r.has_value()) {
        return false;
    }
    const auto* domain = std::get_if<DomainError>(&r.error());
    return domain != nullptr && domain->function == kUnsupportedFn &&
           domain->reason == kUnsupportedReason;
}

inline bool is_domain_error(const Result<Matrix<double>>& r, const char* fn,
                            const char* reason) {
    if (r.has_value()) {
        return false;
    }
    const auto* domain = std::get_if<DomainError>(&r.error());
    return domain != nullptr && domain->function == fn && domain->reason == reason;
}

inline std::string describe(const Result<Matrix<double>>& r) {
    if (r.has_value()) {
        return "a matrix";
    }
    if (const auto* domain = std::get_if<DomainError>(&r.error())) {
        return "DomainError{" + domain->function + ", " + domain->reason + "}";
    }
    return "a non-domain error";
}

/// The callee reaches a handler at all.
///
/// A handler that compiles, links, and is never registered is invisible: the
/// dispatcher reports "unsupported matrix call" and the failure looks like a
/// typo in the user's script rather than a missing registration.
inline void expect_registered(const char* callee) {
    EXPECT_TRUE(is_matrix_call_callee(callee))
        << callee << " is not registered, so no script can call it";
}

/// An argument count the guard does not accept is rejected, not acted on.
///
/// The guard is the only thing standing between a wrong-arity call and an
/// out-of-range `assign.args[i]`, so this is a memory-safety property as much as
/// a diagnostics one.
/// `fn` and `reason` come from the handler's own source via the manifest, not from
/// a convention. 481 handlers leave the pre-initialised sentinel in place; four
/// reject with a message naming the signature they expected, which is a better
/// diagnostic and would have failed a test that assumed the sentinel for all of
/// them. Asserting the real contract catches a handler that changes its rejection
/// silently, which asserting the convention could not.
inline void expect_arity_rejected(const char* callee, std::size_t arity,
                                  const char* fn, const char* reason) {
    const auto r = dispatch(callee, arity);
    EXPECT_TRUE(is_domain_error(r, fn, reason))
        << callee << " with " << arity << " argument(s) returned " << describe(r)
        << "; expected DomainError{" << fn << ", " << reason << "}";
}

/// An accepted argument count with operands that do not exist fails cleanly.
///
/// Two properties in one dispatch. The call must not crash and must not invent a
/// result, and the error must not be "unsupported matrix call" -- if it were, the
/// guard would have rejected an arity the manifest says it accepts, and the
/// arity data behind every other generated test would be wrong.
inline void expect_undefined_operand_propagates(const char* callee, std::size_t arity) {
    const auto r = dispatch(callee, arity);
    ASSERT_FALSE(r.has_value())
        << callee << " returned a matrix for undefined operands";
    EXPECT_FALSE(is_unsupported(r))
        << callee << " rejected " << arity << " argument(s), which the manifest records "
        << "as an accepted arity";
}

/// An undefined operand at any position fails cleanly.
///
/// The contract is the same as for position 0, but it has to hold for every
/// argument the handler resolves: a caller who mistypes the third name should get
/// an error naming the problem, not a crash and not a result computed from the two
/// arguments that did resolve.
inline void expect_bad_operand_at(const char* callee, std::size_t arity,
                                  std::size_t bad_index) {
    const auto r = dispatch_with_bad_arg(callee, arity, bad_index);
    ASSERT_FALSE(r.has_value())
        << callee << " returned a matrix with argument " << bad_index << " undefined";
    EXPECT_FALSE(is_unsupported(r))
        << callee << " rejected " << arity << " argument(s), which the manifest "
        << "records as an accepted arity";
}

} // namespace ms::interp::testing
