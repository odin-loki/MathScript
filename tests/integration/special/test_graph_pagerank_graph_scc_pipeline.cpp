
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

TEST(IntegrationSpecial,  GraphImagePoly) {
    Interpreter interp;

    expect_contains(interp, "help", "graph_pagerank");
    expect_contains(interp, "help", "prewitt");

    expect_ok(interp, "A = [0, 1, 1; 1, 0, 1; 1, 1, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 3u);

    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_GT(interp.state().matrices.at("S").rows(), 0u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 2u);

    expect_ok(interp, "p = [1; 2; 3]");
    expect_ok(interp, "dp = poly_deriv(p)");
    EXPECT_EQ(interp.state().matrices.at("dp").rows(), 2u);
}

TEST(IntegrationSpecial,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-9);

    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-9);
}
