
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

TEST(IntegrationTopo,  TopoAlphaWitnessLandscapeTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_alpha_complex");
    expect_contains(interp, "help", "topo_select_landmarks");
    expect_contains(interp, "help", "topo_witness_complex");

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(IntegrationTopo,  SphBesselYScalar) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}
