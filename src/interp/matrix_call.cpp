#include "matrix_call.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace ms::interp {
namespace {

std::unordered_map<std::string, MatrixCallHandler>& matrix_call_map() {
    static std::unordered_map<std::string, MatrixCallHandler> map;
    return map;
}

} // namespace

void register_matrix_call(std::string_view name, MatrixCallHandler handler) {
    matrix_call_map().emplace(std::string(name), handler);
}

bool is_matrix_call_callee(const std::string& callee) {
    ensure_matrix_calls_registered();
    return matrix_call_map().contains(callee);
}

Result<Matrix<double>> dispatch_matrix_call(Interpreter& interp, const MatrixCallAssign& assign) {
    ensure_matrix_calls_registered();
    const auto it = matrix_call_map().find(assign.callee);
    if (it == matrix_call_map().end()) {
        return std::unexpected(DomainError{"assign", "unsupported matrix call"});
    }
    return it->second(interp, assign);
}

} // namespace ms::interp
