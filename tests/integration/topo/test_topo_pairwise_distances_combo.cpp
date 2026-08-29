
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

TEST(IntegrationTopo,  PairwiseNextPermTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_pairwise_distances(P)");
    expect_contains(interp, "help", "combo_next_perm(v)");

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    const auto& dist = interp.state().matrices.at("D");
    EXPECT_EQ(dist.rows(), 3u);
    EXPECT_EQ(dist.cols(), 3u);
    EXPECT_NEAR(dist(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(dist(0, 2), 1.0, 1e-9);
    EXPECT_NEAR(dist(1, 2), std::sqrt(2.0), 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    const auto& np = interp.state().matrices.at("np");
    EXPECT_NEAR(np(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(np(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(np(2, 0), 2.0, 1e-9);
}

TEST(IntegrationTopo,  JacobiNdScalar) {
    Interpreter interp;
    expect_ok(interp, "jnd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}
