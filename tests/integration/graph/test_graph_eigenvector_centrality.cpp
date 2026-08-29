
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

TEST(IntegrationGraph,  EigenvectorKatzTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_katz_centrality(A)");

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(IntegrationGraph,  AngerJScalar) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}
