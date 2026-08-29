
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

TEST(IntegrationGraph,  BetweennessClosenessDegreeTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_betweenness");
    expect_contains(interp, "help", "graph_closeness");
    expect_contains(interp, "help", "graph_degree_centrality");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);
}

TEST(IntegrationGraph,  JacobiDsScalar) {
    Interpreter interp;
    expect_ok(interp, "ds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}
