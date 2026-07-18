// MathScript Integration Tests: REPL Interpreter – Wave 234 Bindings Pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

void expect_near_in_output(Interpreter& interp, const std::string& cmd, double expected,
                           double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const std::size_t start = result->find('[');
    ASSERT_NE(start, std::string::npos) << *result;
    const std::size_t end = result->find(']', start);
    ASSERT_NE(end, std::string::npos) << *result;
    const double x_opt = std::stod(result->substr(start + 1, end - start - 1));
    EXPECT_NEAR(x_opt, expected, tol) << cmd << " output: " << *result;
}

void expect_near_scalar(Interpreter& interp, const std::string& cmd, double expected, double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const double value = std::stod(*result);
    EXPECT_NEAR(value, expected, tol) << cmd << " output: " << *result;
}

void expect_near_matrix_entry(Interpreter& interp, const std::string& assign_cmd, size_t row,
                              double expected, double tol) {
    expect_ok(interp, assign_cmd);
    const std::string name = Interpreter::trim(assign_cmd.substr(0, assign_cmd.find('=')));
    const auto& m = interp.state().matrices.at(name);
    ASSERT_GT(m.rows(), row);
    EXPECT_NEAR(m(row, 0), expected, tol) << assign_cmd;
}

} // namespace

TEST(ReplWave234Pipeline, FinancePortfolioBindings) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0; 0, 0.01]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    expect_near_scalar(interp, "finance_portfolio_return(w_min, [0.08; 0.12])", 0.112, 1e-2);

    expect_ok(interp, "cov3 = [0.10, 0.02, 0.01; 0.02, 0.08, 0.03; 0.01, 0.03, 0.06]");
    expect_ok(interp, "mu3 = [0.08; 0.12; 0.10]");
    expect_ok(interp, "w_sharpe = finance_max_sharpe_portfolio(cov3, mu3, 0.02)");
    expect_contains(interp, "finance_portfolio_return(w_sharpe, mu3)", "0.");

    expect_ok(interp, "finance_heston_call(100, 100, 1, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)");
    expect_contains(interp, "finance_heston_call(100, 100, 1, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)",
                    "1");

    expect_contains(interp, "help", "finance_min_variance_portfolio(cov)");
    expect_contains(interp, "help", "finance_max_sharpe_portfolio(cov,mu,risk_free)");
    expect_contains(interp, "help", "finance_heston_call(S,K,T,r,v0,kappa,theta,sigma_v,rho)");
}

TEST(ReplWave234Pipeline, GraphCommunityBindings) {
    Interpreter interp;

    // Barbell graph: two triangles joined by edge 2-3; articulation points at 2 and 3.
    expect_ok(interp,
              "A = [0,1,1,0,0,0; 1,0,1,0,0,0; 1,1,0,1,0,0; 0,0,1,0,1,1; 0,0,0,1,0,1; 0,0,0,1,1,0]");

    expect_ok(interp, "louv = graph_louvain(A)");
    ASSERT_GT(interp.state().matrices.count("louv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    ASSERT_GT(interp.state().matrices.count("ec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_contains(interp, "graph_articulation_points(A)", "2");
    expect_contains(interp, "graph_articulation_points(A)", "3");

    expect_contains(interp, "help", "graph_louvain(A)");
    expect_contains(interp, "help", "graph_eigenvector_centrality(A)");
    expect_contains(interp, "help", "graph_articulation_points(A)");
}

TEST(ReplWave234Pipeline, ImageMorphologyBindings) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "D = imdilate(M, 3)");
    expect_ok(interp, "E = imerode(M, 3)");
    expect_ok(interp, "O = imopen(M, 3)");
    expect_ok(interp, "C = imclose(M, 3)");

    const auto& dil = interp.state().matrices.at("D");
    const auto& ero = interp.state().matrices.at("E");
    EXPECT_GE(dil(1, 2), ero(1, 2));

    expect_contains(interp, "help", "imdilate(M,k)");
    expect_contains(interp, "help", "imerode(M,k)");
    expect_contains(interp, "help", "imopen(M,k)");
    expect_contains(interp, "help", "imclose(M,k)");
}

TEST(ReplWave234Pipeline, Rgb2HsvBinding) {
    Interpreter interp;

    expect_ok(interp, "RGB = [1,0,0; 0,1,0; 0,0,1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB)");
    const auto& hsv = interp.state().matrices.at("HSV");
    ASSERT_EQ(hsv.rows(), 3u);
    ASSERT_EQ(hsv.cols(), 3u);
    EXPECT_NEAR(hsv(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(hsv(1, 0), 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(hsv(2, 0), 2.0 / 3.0, 1e-6);

    expect_contains(interp, "help", "rgb2hsv(M)");
}

TEST(ReplWave234Pipeline, LinearRegressionBindings) {
    Interpreter interp;

    expect_ok(interp, "X = [1;2;3;4;5]");
    expect_ok(interp, "y = [3;5;7;9;11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_near_matrix_entry(interp, "pred = ml_linear_predict([6], model)", 0, 13.0, 0.2);

    expect_contains(interp, "help", "ml_linear_fit(X,y)");
    expect_contains(interp, "help", "ml_linear_predict(X,model)");
}

TEST(ReplWave234Pipeline, RidgeRegressionBindings) {
    Interpreter interp;

    expect_ok(interp, "X = [0;1;2;3]");
    expect_ok(interp, "y = [1;3;5;7]");
    expect_ok(interp, "model = ml_ridge_fit(X, y, 0.001)");
    expect_near_matrix_entry(interp, "pred = ml_ridge_predict([4], model)", 0, 9.0, 0.5);

    expect_contains(interp, "help", "ml_ridge_fit(X,y,alpha)");
    expect_contains(interp, "help", "ml_ridge_predict(X,model)");
}

TEST(ReplWave234Pipeline, LogisticRegressionBindings) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    expect_ok(interp, "pred_pos = ml_logistic_predict([2], model)");
    expect_ok(interp, "pred_neg = ml_logistic_predict([-2], model)");

    const auto& pos = interp.state().matrices.at("pred_pos");
    const auto& neg = interp.state().matrices.at("pred_neg");
    EXPECT_GT(pos(0, 0), neg(0, 0));

    expect_contains(interp, "help", "ml_logistic_fit(X,y)");
    expect_contains(interp, "help", "ml_logistic_predict(X,model)");
}
