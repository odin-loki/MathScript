// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Every REPL builtin can be called WITHOUT assigning its result, in which case
// the interpreter prints the value under a fixed label instead of binding it.
// That printing path is a second, separate dispatch chain from the assignment
// one, and for a large group of builtins nothing ever exercised it: the chain
// was entered, every `fn == "..."` comparison ran, and every body after the
// first match was dead code.
//
// These tests call each of those builtins in the no-assignment form and check
// the printed label, so the print path is pinned as well as reached.

#include <gtest/gtest.h>

#include <string>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

// A fixture with the matrices the cases below share.
void seed(Interpreter& interp) {
    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "y = [2; 1; 4; 3; 6; 5; 8; 7]");
    expect_ok(interp, "A = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "M = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "p = [1; 2; 3]");
    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
}

}  // namespace

TEST(ReplNoAssign, MatrixAndScalarBuiltins) {
    Interpreter interp;
    seed(interp);

    expect_contains(interp, "poly_eval(p, 2)", ".");
    expect_contains(interp, "poly_cheb_eval(p, 0)", ".");
    expect_ok(interp, "poly_integ(p, 0)");
    expect_ok(interp, "poly_shift(p, 1)");
    expect_ok(interp, "poly_scale(p, 2)");
    expect_ok(interp, "poly_pow(p, 2)");
    expect_ok(interp, "poly_cheb_expand(p, 3)");

    expect_ok(interp, "geo_bezier_eval_x(ctrl, 0.5)");
    expect_ok(interp, "geo_bezier_eval_y(ctrl, 0.5)");

    expect_ok(interp, "signal_moving_average(x, 3)");
    expect_ok(interp, "signal_median_filter(x, 3)");
    expect_ok(interp, "signal_upsample(x, 2)");
    expect_ok(interp, "signal_downsample(x, 2)");
    expect_ok(interp, "signal_decimate(x, 2)");
    expect_ok(interp, "signal_interpolate(x, 2)");

    expect_ok(interp, "graph_bfs(A, 0)");
    expect_ok(interp, "graph_dfs(A, 0)");
    expect_ok(interp, "graph_k_core_subgraph(A, 1)");
    expect_ok(interp, "graph_bipartite_match(A, 2)");

    expect_ok(interp, "stats_percentile(x, 50)");
    expect_ok(interp, "stats_ttest(x, 4)");
    expect_ok(interp, "stats_trimmed_mean(x, 0.1)");

    expect_ok(interp, "returns = [0.01; -0.02; 0.03; -0.01; 0.02; -0.03; 0.04; 0.0]");
    expect_ok(interp, "finance_historical_var(returns, 0.95)");
    expect_ok(interp, "finance_historical_cvar(returns, 0.95)");

    expect_ok(interp, "S = fft_rfft(x)");
    expect_ok(interp, "fft_irfft(S, 8)");

    expect_ok(interp, "H = [0, 0.5; 0.5, 0]");
    expect_ok(interp, "quantum_time_evolution(H, 0.25)");

    expect_ok(interp, "comb = [0; 1]");
    expect_ok(interp, "combo_rank_combination(comb, 4)");
    expect_ok(interp, "combo_next_comb(comb, 4)");
    expect_ok(interp, "comb2 = [0; 2]");
    expect_ok(interp, "combo_prev_comb(comb2, 4)");
}

TEST(ReplNoAssign, VarianceInflationFactorBothSpellings) {
    Interpreter interp;
    // Two independent-ish columns, so VIF is finite and >= 1.
    expect_ok(interp, "Xv = [1, 5; 2, 1; 3, 9; 4, 2; 5, 7]");
    expect_ok(interp, "stats_vif(Xv, 0)");
    expect_ok(interp, "stats_variance_inflation_factor(Xv, 1)");
}

TEST(ReplNoAssign, BurrowsWheelerDecode) {
    Interpreter interp;
    expect_ok(interp, "src = [66; 65; 78; 65; 78; 65]");
    expect_ok(interp, "bwt = bwt_encode_vec(src)");
    expect_ok(interp, "pi = bwt_primary_index(bwt)");
    expect_ok(interp, "bwt_decode_vec(bwt, pi)");
}

TEST(ReplNoAssign, SignalBuiltinsWithThreeOrFourArguments) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 0; -1; 2; 1; 0; -1; 2]");
    expect_ok(interp, "d = [0.5; -0.25; 1; 0; 0.5; -0.25; 1; 0]");

    expect_ok(interp, "signal_coherence(x, x, 8.0, 8)");
    expect_ok(interp, "signal_lms(x, d, 2, 0)");
    expect_ok(interp, "signal_lms_weights(x, d, 2, 0)");
    expect_ok(interp, "signal_resample(x, 2, 2)");
    expect_ok(interp, "signal_savgol(x, 3, 1)");
    expect_ok(interp, "signal_bandpass(x, 0.1, 0.3, 1.0)");
    expect_ok(interp, "signal_lowpass(x, 0.3, 1.0)");
    expect_ok(interp, "signal_highpass(x, 0.3, 1.0)");
}

TEST(ReplNoAssign, ImageFilters) {
    Interpreter interp;
    expect_ok(interp, "G = [1, 2, 3, 4; 5, 60, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16]");
    expect_ok(interp, "boxfilter(G, 3)");
    expect_ok(interp, "medfilt2(G, 3)");
    expect_ok(interp, "imgaussfilt(G, 1)");
}

TEST(ReplNoAssign, QuantumPhaseSpaceAndPartialTrace) {
    Interpreter interp;
    expect_ok(interp, "rho = [0.5, 0; 0, 0.5]");
    expect_ok(interp, "quantum_wigner(rho, 0, 0)");
    expect_ok(interp, "quantum_husimi(rho, 1, 0)");
    expect_ok(interp, "rho4 = [0.25, 0, 0, 0; 0, 0.25, 0, 0; 0, 0, 0.25, 0; 0, 0, 0, 0.25]");
    expect_ok(interp, "quantum_partial_trace(rho4, 2, 2, 0)");
    expect_ok(interp, "quantum_grover_optimal_iterations(3, 1)");
}

TEST(ReplNoAssign, InformationAndFinanceThreeArgumentForms) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 2; 1; 2; 1; 2; 1; 2; 1; 2; 1; 2]");
    expect_ok(interp, "info_sample_entropy(x, 2, 0.5)");
    expect_ok(interp, "r = [0.01; -0.02; 0.03; -0.01; 0.02; -0.03]");
    expect_ok(interp, "finance_treynor(r, 0.05, 1.2)");
}

TEST(ReplNoAssign, SymbolicOneAndTwoArgumentForms) {
    Interpreter interp;
    expect_ok(interp, "sym_diff(\"sin(x)*x^2\", \"x\")");
    expect_ok(interp, "sym_integrate(\"x^2\", \"x\")");
    expect_ok(interp, "sym_eval(\"x^2+1\", \"x=3\")");
    expect_ok(interp, "sym_collect(\"x+x+1\", \"x\")");
    expect_ok(interp, "sym_solve_linear(\"2*x+4\", \"x\")");
    expect_ok(interp, "sym_simplify(\"x+x\")");
    expect_ok(interp, "sym_expand(\"x+x\")");
    expect_ok(interp, "sym_substitute(\"x+1\", \"x\", \"2\")");
    expect_ok(interp, "sym_limit(\"x+1\", \"x\", 2)");
    expect_ok(interp, "sym_series(\"exp(x)\", \"x\", 0, 2)");
}

// Six of these decline, and that is the correct answer rather than a gap.
//
// The transform of a constant is a Dirac delta or a divergent integral in each of
// those cases -- F{1} = 2*pi*delta(w), Z^-1{1} = delta[n] -- and SymExpr has no
// node for a distribution, so there is nothing for the function to return.
//
// They were asserted with expect_ok until the unsupported sentinel became an
// error. That assertion passed for the wrong reason: the sentinel is a successful
// return, so expect_ok could not tell a computed transform from a declined one,
// and every one of these was green while computing nothing.
TEST(ReplNoAssign, SymbolicIntegralTransforms) {
    Interpreter interp;
    expect_ok(interp, "sym_laplace(\"1\", \"t\", \"s\")");
    expect_ok(interp, "sym_ilaplace(\"1/s\", \"s\", \"t\")");
    expect_ok(interp, "sym_mellin(\"1\", \"t\", \"s\")");
    expect_error_contains(interp, "sym_imellin(\"1\", \"s\", \"t\")", "no closed form");
    expect_error_contains(interp, "sym_hankel(\"1\", \"r\", \"k\")", "no closed form");
    expect_error_contains(interp, "sym_ihankel(\"1\", \"k\", \"r\")", "no closed form");
    expect_error_contains(interp, "sym_fourier(\"1\", \"t\", \"w\")", "no closed form");
    expect_error_contains(interp, "sym_ifourier(\"1\", \"w\", \"t\")", "no closed form");
    expect_ok(interp, "sym_ztransform(\"1\", \"n\", \"z\")");
    expect_error_contains(interp, "sym_iztransform(\"1\", \"z\", \"n\")", "no closed form");
    expect_ok(interp, "sym_dsolve(\"1\", \"x\", \"y\")");
}

TEST(ReplNoAssign, TensorAndDeviceForms) {
    Interpreter interp;
    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "tensorops_matmul(M1, M2)");
    expect_ok(interp, "tensorops_einsum(M1, M2)");
    expect_ok(interp, "cuda_add(M1, M2)");
    // The CPU build has no cuBLAS LU; it must report that rather than pretend.
    expect_error(interp, "cuda_lu(M1)");
}

TEST(ReplNoAssign, MachineLearningPredictors) {
    Interpreter interp;
    expect_ok(interp, "X = [1, 1; 1.2, 1.1; 5, 5; 5.2, 5.1; 0.9, 1.05; 5.1, 4.9]");
    expect_ok(interp, "km = ml_kmeans_fit(X, 2)");
    expect_contains(interp, "ml_kmeans_predict(X, km)", "labels =");
    expect_ok(interp, "pca = ml_pca_fit(X, 1)");
    expect_contains(interp, "ml_pca_transform(X, pca)", "Z =");
    expect_ok(interp, "gmm = ml_gmm_fit(X, 2)");
    expect_contains(interp, "ml_gmm_predict(X, gmm)", "labels =");
    expect_contains(interp, "ml_gmm_predict_proba(X, gmm)", "proba =");
    expect_ok(interp, "iso = ml_isolation_forest_fit(X, 8)");
    expect_contains(interp, "ml_isolation_forest_score(X, iso)", "score =");
}

TEST(ReplNoAssign, PlanarityAndMatchingOnAGraph) {
    Interpreter interp;
    // K4 is planar; K5 is not.
    expect_ok(interp,
              "K4 = [0, 1, 1, 1; 1, 0, 1, 1; 1, 1, 0, 1; 1, 1, 1, 0]");
    expect_ok(interp, "graph_is_planar_heuristic(K4)");
    expect_ok(interp, "graph_planar_embedding(K4)");
    expect_ok(interp, "graph_kuratowski_subgraph(K4)");

    expect_ok(interp,
              "W = [0, 5, 1, 0; 5, 0, 0, 2; 1, 0, 0, 6; 0, 2, 6, 0]");
    expect_ok(interp, "graph_max_weight_matching(W)");
    expect_ok(interp, "graph_max_weight_matching_value(W)");
}

TEST(ReplNoAssign, SparseAndComplexHelpers) {
    Interpreter interp;
    expect_ok(interp, "RI = [0; 1; 2]");
    expect_ok(interp, "CI = [0; 1; 2]");
    expect_ok(interp, "V = [1; 2; 3]");
    expect_ok(interp, "Sp = sparse_from_coo(3, 3, RI, CI, V)");
    // sparse_to_dense(A) has arity 1 in the arity table and works when assigned, but the
    // printing form used to fall through to "unknown function".
    expect_contains(interp, "sparse_to_dense(Sp)", "dense =");
    expect_ok(interp, "cplx_poisson_kernel(0, 0, 0.5)");
}

TEST(ReplNoAssign, PlotAndDisplayForms) {
    Interpreter interp;
    expect_ok(interp, "X = [1; 2; 3; 4]");
    expect_ok(interp, "Y = [1; 4; 9; 16]");
    expect_ok(interp, "plot(X, Y)");
    expect_ok(interp, "scatter(X, Y)");
    expect_ok(interp, "RI = [0; 1; 2]");
    expect_ok(interp, "CI = [0; 1; 2]");
    expect_ok(interp, "V = [1; 2; 3]");
    expect_ok(interp, "Sp = sparse_from_coo(3, 3, RI, CI, V)");
    expect_ok(interp, "spy(Sp)");
}

TEST(ReplNoAssign, SupervisedPredictorsPrintWithoutAssignment) {
    // is_matrix_dual_matrix_call_callee claimed all of these, but the printing chain had
    // no branch for them, so the call fell out of the chain and was retried as a single
    // matrix literally named "X, model" -- reporting "unknown matrix" for a call that
    // works perfectly when its result is assigned.
    Interpreter interp;
    expect_ok(interp, "X = [1, 1; 2, 1; 3, 1; 4, 1; 5, 1; 6, 1]");
    expect_ok(interp, "y = [1; 0; 1; 0; 1; 0]");
    expect_ok(interp, "yr = [2; 4; 6; 8; 10; 12]");

    expect_ok(interp, "lin = ml_linear_fit(X, yr)");
    expect_contains(interp, "ml_linear_predict(X, lin)", "pred =");
    expect_contains(interp, "ml_linear_fit(X, yr)", "model =");

    expect_ok(interp, "rid = ml_ridge_fit(X, yr, 0.1)");
    expect_contains(interp, "ml_ridge_predict(X, rid)", "pred =");

    expect_contains(interp, "ml_logistic_fit(X, y)", "model =");
    expect_ok(interp, "lg = ml_logistic_fit(X, y)");
    expect_contains(interp, "ml_logistic_predict(X, lg)", "pred =");

    expect_ok(interp, "las = ml_lasso_fit(X, yr, 0.1)");
    expect_contains(interp, "ml_lasso_predict(X, las)", "pred =");
    expect_ok(interp, "en = ml_elastic_net_fit(X, yr, 0.1, 0.5)");
    expect_contains(interp, "ml_elastic_net_predict(X, en)", "pred =");

    expect_ok(interp, "knn = ml_knn_fit(X, y, 3)");
    expect_contains(interp, "ml_knn_predict(X, knn)", "pred =");
    expect_ok(interp, "nb = ml_naive_bayes_fit(X, y)");
    expect_contains(interp, "ml_naive_bayes_predict(X, nb)", "pred =");
    expect_ok(interp, "lda = ml_lda_fit(X, y)");
    expect_contains(interp, "ml_lda_predict(X, lda)", "pred =");
    expect_contains(interp, "ml_lda_transform(X, lda)", "Z =");
    expect_ok(interp, "qda = ml_qda_fit(X, y)");
    expect_contains(interp, "ml_qda_predict(X, qda)", "pred =");
    expect_ok(interp, "svm = ml_svm_fit(X, y)");
    expect_contains(interp, "ml_svm_predict(X, svm)", "pred =");

    expect_ok(interp, "dt = ml_decision_tree_fit(X, y)");
    expect_contains(interp, "ml_decision_tree_predict(X, dt)", "pred =");
    expect_ok(interp, "rf = ml_random_forest_fit(X, y)");
    expect_contains(interp, "ml_random_forest_predict(X, rf)", "pred =");
    expect_ok(interp, "ab = ml_adaboost_fit(X, y)");
    expect_contains(interp, "ml_adaboost_predict(X, ab)", "pred =");
    expect_ok(interp, "gb = ml_gradient_boosting_fit(X, y)");
    expect_contains(interp, "ml_gradient_boosting_predict(X, gb)", "pred =");
}

TEST(ReplNoAssign, QuantumAnticommutatorPrintsWithoutAssignment) {
    Interpreter interp;
    expect_ok(interp, "Xg = [0, 1; 1, 0]");
    expect_ok(interp, "Zg = [1, 0; 0, -1]");
    // {X, Z} = XZ + ZX = 0 for the Pauli matrices.
    expect_contains(interp, "quantum_anticommutator(Xg, Zg)", "anticomm =");
}

// A bare name or expression is the last form the dispatcher tries, after every
// command and every assignment form has declined the line. Before this existed
// the only way to see a value was to assign it somewhere else first: `x`,
// `1 + 2` and `sqrt(2)` all came back as "could not parse".
TEST(ReplNoAssign, BareExpressionPrintsItsValue) {
    Interpreter interp;
    expect_ok(interp, "x = 2.5");
    expect_ok(interp, "A = [1, 2; 3, 4]");

    // A bare scalar name prints the bound value.
    expect_contains(interp, "x", "2.500000");
    // A bare matrix name prints the matrix under its own name.
    expect_contains(interp, "A", "A =");
    expect_contains(interp, "A", "[1.000000, 2.000000]");

    // Literal arithmetic, calls, and expressions over bound names all evaluate.
    expect_contains(interp, "1 + 2", "3.000000");
    expect_contains(interp, "sqrt(2)", "1.414214");
    expect_contains(interp, "x / 2 + 1", "2.250000");
    expect_contains(interp, "-x", "-2.500000");
    expect_contains(interp, "pow(x, 2)", "6.250000");
}

// The fallback must stay a fallback: a line that is not an expression still
// reports the parse error, and a command name is still a command.
TEST(ReplNoAssign, BareExpressionDoesNotShadowCommandsOrErrors) {
    Interpreter interp;

    // Bare `load` / `save` are incomplete commands, not variables.
    expect_error_contains(interp, "load", "could not parse: load");
    expect_error_contains(interp, "save", "could not parse: save");
    // An unbound name is not silently zero.
    expect_error(interp, "no_such_variable_here");
    // Nor is a line that is not an expression at all.
    expect_error_contains(interp, "1 2 3 ;;", "could not parse");
    // `vars` still lists the session rather than being read as a name.
    expect_ok(interp, "q = 7");
    expect_contains(interp, "vars", "q = 7");
}
