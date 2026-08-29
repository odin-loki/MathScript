
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

TEST(IntegrationPoly,  NumthyPolyGraph) {
    Interpreter interp;

    expect_contains(interp, "help", "numthy_lucas_sequence");
    expect_contains(interp, "help", "poly_interp_hermite");
    expect_contains(interp, "help", "graph_connected_components");

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_EQ(interp.state().matrices.at("lu").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("lu").cols(), 2u);

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_EQ(interp.state().matrices.at("c").rows(), 2u);

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    EXPECT_EQ(interp.state().matrices.at("rr").rows(), 2u);
}

TEST(IntegrationPoly,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "ff = special_falling_factorial(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("ff"), ms::falling_factorial(5.0, 2), 1e-9);
}
