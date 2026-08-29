
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

TEST(IntegrationSpecial,  ImresizeWatershedTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "imresize(M,r,c)");
    expect_contains(interp, "help", "watershed(G,M)");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(IntegrationSpecial,  JacobiDcScalar) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}
