
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

TEST(IntegrationTopo,  WatershedPairwiseTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "watershed");
    expect_contains(interp, "help", "topo_pairwise_distances");

    expect_ok(interp, "G = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
}

TEST(IntegrationTopo,  HermiteHfScalar) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}
