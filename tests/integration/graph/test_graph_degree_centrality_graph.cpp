
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

TEST(IntegrationGraph,  DegreeTopoSortTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_degree_centrality(A)");
    expect_contains(interp, "help", "graph_topological_sort(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(IntegrationGraph,  Theta3Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}
