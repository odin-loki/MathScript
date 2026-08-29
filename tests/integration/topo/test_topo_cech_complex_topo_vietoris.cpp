
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

TEST(IntegrationTopo,  TopoCechVietorisBetti) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_cech_complex");
    expect_contains(interp, "help", "topo_vietoris_rips");
    expect_contains(interp, "help", "topo_simplicial_betti");

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(IntegrationTopo,  Theta1PrimeScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}
