// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/error/error_types.hpp"
#include <sstream>
#include <type_traits>

namespace ms {

std::string format_error(const Error& error) {
    return std::visit([](const auto& e) -> std::string {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, DimensionMismatch>) {
            std::ostringstream out;
            out << "dimension mismatch: got " << e.got_rows << "x" << e.got_cols;
            if (e.expected_rows != 0 || e.expected_cols != 0) {
                out << ", expected " << e.expected_rows << "x" << e.expected_cols;
            }
            return out.str();
        } else if constexpr (std::is_same_v<T, SingularMatrix>) {
            return "singular matrix";
        } else if constexpr (std::is_same_v<T, DomainError>) {
            return std::string(e.function) + ": " + e.reason;
        } else if constexpr (std::is_same_v<T, ConvergenceFail>) {
            return "convergence failed";
        } else if constexpr (std::is_same_v<T, ParseError>) {
            // The position is the reason this type exists rather than SymbolicError,
            // and dropping it here made every caller carrying one report as if it had
            // none. Zero for both is how a caller says it does not know where.
            if (e.line == 0 && e.col == 0) {
                return e.msg;
            }
            std::ostringstream out;
            out << "line " << e.line << ", column " << e.col << ": " << e.msg;
            return out.str();
        } else if constexpr (std::is_same_v<T, ValueOutOfRange>) {
            std::ostringstream out;
            out << e.param << " value " << e.value
                << " out of range [" << e.lo << ", " << e.hi << "]";
            return out.str();
        } else {
            return "mathscript error";
        }
    }, error);
}

} // namespace ms
