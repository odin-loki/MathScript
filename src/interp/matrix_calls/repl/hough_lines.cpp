// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {

Result<Matrix<double>> handle_hough_lines(Interpreter& interp, const MatrixCallAssign& assign) {
    using namespace detail;
    MatrixCallCtx ctx(interp);

    Result<Matrix<double>> result =
        std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "hough_lines" &&
               (assign.args.size() == 1 || assign.args.size() == 2 ||
                assign.args.size() == 5)) {
        auto matrix = ctx.resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        double edge_threshold = 0.5;
        int n_theta = 180;
        int n_rho = 200;
        int vote_threshold = 50;
        if (assign.args.size() >= 2) {
            if (!parse_number(assign.args[1], edge_threshold)) {
                return std::unexpected(DomainError{
                    "hough_lines",
                    "expected hough_lines(M[, edge]) or hough_lines(M, edge, n_theta, n_rho, vote)"});
            }
        }
        if (assign.args.size() == 5) {
            double n_theta_d = 0.0;
            double n_rho_d = 0.0;
            double vote_d = 0.0;
            if (!parse_number(assign.args[2], n_theta_d) || !parse_number(assign.args[3], n_rho_d) ||
                !parse_number(assign.args[4], vote_d)) {
                return std::unexpected(DomainError{
                    "hough_lines",
                    "expected hough_lines(M[, edge]) or hough_lines(M, edge, n_theta, n_rho, vote)"});
            }
            n_theta = static_cast<int>(n_theta_d);
            n_rho = static_cast<int>(n_rho_d);
            vote_threshold = static_cast<int>(vote_d);
            if (n_theta_d != n_theta || n_rho_d != n_rho || vote_d != vote_threshold) {
                return std::unexpected(
                    DomainError{"hough_lines", "expected integer n_theta, n_rho, vote"});
            }
        }
        result = eval_hough_lines(*matrix, edge_threshold, n_theta, n_rho, vote_threshold);
    }

    return result;
}

void ms_register_matrix_call_hough_lines() {
    register_matrix_call("hough_lines", &handle_hough_lines);
}

} // namespace ms::interp
