// MathScript Integration Tests: REPL Interpreter – Wave 302 Pipeline
//
// Wave 302 REPL smoke: image tail11 extensions, polylog/debye scalar validation.

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

TEST(ReplWave302Pipeline, Image) {
    Interpreter interp;

    expect_contains(interp, "help", "imfilter");
    expect_contains(interp, "help", "sobel_x");
    expect_contains(interp, "help", "laplacian_of_gaussian");

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);

    expect_ok(interp, "Sx = sobel_x(G)");
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);

    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "D = dft_magnitude(G)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplWave302Pipeline, SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-9);

    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-9);
}
