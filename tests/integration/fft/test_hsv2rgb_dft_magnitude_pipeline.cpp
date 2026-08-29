
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

TEST(IntegrationFft,  Hsv2rgbDftTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "hsv2rgb");
    expect_contains(interp, "help", "dft_magnitude");

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(IntegrationFft,  SphBesselJScalar) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}
