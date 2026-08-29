
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/quantum/quantum.hpp"

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

TEST(IntegrationGraph,  SpectrumLaplacianTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_adjacency_spectrum");
    expect_contains(interp, "help", "graph_laplacian");

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(IntegrationGraph,  GroverItersScalar) {
    Interpreter interp;
    expect_contains(interp, "quantum_grover_optimal_iterations(3, 1)", "2");
}
