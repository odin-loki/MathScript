#pragma once

#include "ms/interp/repl_engine.hpp"
#include <string_view>

namespace ms::interp {

struct MatrixCallCtx {
    Interpreter& interp;
    explicit MatrixCallCtx(Interpreter& i) : interp(i) {}
    Result<Matrix<double>> resolve_operand(const std::string& text) {
        return interp.eval_matrix_operand(text);
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
