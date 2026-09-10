// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Setting an environment variable from a test, on both toolchains.
//
// `setenv` and `unsetenv` are POSIX and MSVC has neither, so a test that reaches
// for them compiles everywhere the developer looked and fails on Windows CI forty
// minutes later. That is what happened to the ISA-gating tests.
//
// Included by relative path rather than through an include directory: these are
// three functions, and adding a search path to every test target to carry them
// would be a larger change than the problem.
#pragma once

#include <cstdlib>
#include <string>

namespace ms::testing {

inline void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    ::_putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}

inline void unset_env(const char* name) {
#if defined(_WIN32)
    // Windows has no unsetenv. Assigning an empty value removes the variable
    // outright, so getenv afterwards returns nullptr -- the same observable state
    // POSIX unsetenv leaves behind.
    ::_putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

/// Sets a variable for a scope and restores whatever was there before.
///
/// The restore matters more than it looks: these tests set an ISA ceiling, and an
/// expectation that fails partway through would otherwise leak that ceiling into
/// every test that runs after it in the same binary.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        if (const char* const prev = std::getenv(name)) {
            had_previous_ = true;
            previous_ = prev;
        }
        if (value == nullptr) {
            unset_env(name);
        } else {
            set_env(name, value);
        }
    }
    ~ScopedEnv() {
        if (had_previous_) {
            set_env(name_, previous_.c_str());
        } else {
            unset_env(name_);
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};

} // namespace ms::testing
