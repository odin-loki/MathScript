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
