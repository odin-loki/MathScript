
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

TEST(IntegrationSpecial,  GraphStatsGeoImage) {
    Interpreter interp;

    expect_contains(interp, "help", "graph_laplacian");
    expect_contains(interp, "help", "kruskal_wallis");
    expect_contains(interp, "help", "geo_delaunay_2d");
    expect_contains(interp, "help", "adapthisteq");

    expect_ok(interp, "A = [0, 1, 1; 1, 0, 1; 1, 1, 0]");
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);

    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    EXPECT_EQ(interp.state().matrices.at("kw").rows(), 3u);

    expect_ok(interp, "T = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    EXPECT_GT(interp.state().matrices.at("T").rows(), 0u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(IntegrationSpecial,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "ll = laguerre_l(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ll"), ms::laguerre_l(2, 0.5), 1e-9);

    expect_ok(interp, "ct = chebyshev_t(3, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(3, 0.25), 1e-9);
}
