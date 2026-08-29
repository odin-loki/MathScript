
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

TEST(IntegrationSpecial,  ImageTopoStats) {
    Interpreter interp;

    expect_contains(interp, "help", "imgaussfilt");
    expect_contains(interp, "help", "topo_pairwise_distances");
    expect_contains(interp, "help", "stats_linear_regression");

    expect_ok(interp, "G = [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 4u);

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [2; 4; 6; 8; 10]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_EQ(interp.state().matrices.at("lr").rows(), 1u);
}

TEST(IntegrationSpecial,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "cv = chebyshev_v(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), ms::chebyshev_v(2, 0.5), 1e-9);

    expect_ok(interp, "cw = chebyshev_w(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(1, 0.25), 1e-9);
}
