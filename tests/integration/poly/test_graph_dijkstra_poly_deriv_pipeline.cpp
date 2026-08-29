
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationPoly,  DijkstraPolyDerivTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_dijkstra(A,source)");
    expect_contains(interp, "help", "poly_deriv(coeffs)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(IntegrationPoly,  Hypergeo0f1Scalar) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}
