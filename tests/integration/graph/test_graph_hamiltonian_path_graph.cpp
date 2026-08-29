
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

TEST(IntegrationGraph,  HamiltonianTspTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_hamiltonian_path(A)");
    expect_contains(interp, "help", "graph_tsp_heuristic(D)");

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(IntegrationGraph,  GeoVec2dLengthScalar) {
    Interpreter interp;
    expect_ok(interp, "len = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("len"), 5.0, 1e-8);
}
