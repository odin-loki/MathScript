#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_cellmemory_long_term_state(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "cellmemory_long_term_state" && assign.args.size() == 1) {
        std::string handle = trim_copy(assign.args[0]);
        if (!is_identifier(handle)) {
            return std::unexpected(DomainError{
                "cellmemory_long_term_state", "expected CellMemory handle identifier"});
        }
        const auto it = ctx.session_objects().find(handle);
        if (it == ctx.session_objects().end()) {
            return std::unexpected(DomainError{
                "cellmemory_long_term_state", "session object not found: " + handle});
        }
        if (!std::holds_alternative<cellai::CellMemory>(it->second)) {
            return std::unexpected(DomainError{
                "cellmemory_long_term_state",
                std::string("session object '") + handle + "' is not a CellMemory"});
        }
        result = std::get<cellai::CellMemory>(it->second).long_term_state();
    }

    return result;
}

void ms_register_matrix_call_cellmemory_long_term_state() {
    register_matrix_call("cellmemory_long_term_state", &handle_cellmemory_long_term_state);
}

} // namespace ms::interp
