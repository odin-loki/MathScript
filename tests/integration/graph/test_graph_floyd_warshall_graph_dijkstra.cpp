
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

TEST(IntegrationGraph,  FloydDijkstraTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_floyd_warshall");
    expect_contains(interp, "help", "graph_dijkstra");

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);
}

TEST(IntegrationGraph,  Theta4Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}
