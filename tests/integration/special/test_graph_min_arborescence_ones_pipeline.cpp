
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

TEST(IntegrationSpecial,  ArborescenceImfilterTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_min_arborescence");
    expect_contains(interp, "help", "imfilter");

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(IntegrationSpecial,  ChebyshevVScalar) {
    Interpreter interp;
    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-8);
}
