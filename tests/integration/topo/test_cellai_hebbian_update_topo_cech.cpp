
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

TEST(IntegrationTopo,  CellaiTopoTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cellai_hebbian_update");
    expect_contains(interp, "help", "topo_cech_complex");
    expect_contains(interp, "help", "topo_vietoris_rips");

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(IntegrationTopo,  ChebyshevWScalar) {
    Interpreter interp;
    expect_ok(interp, "w = chebyshev_w(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("w"), ms::chebyshev_w(1, 0.5), 1e-8);
}
