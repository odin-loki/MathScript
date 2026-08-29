
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

TEST(IntegrationSpecial,  TophatImadjustTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "imtophat");
    expect_contains(interp, "help", "imadjust");

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);
}

TEST(IntegrationSpecial,  JacobiScScalar) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.2, 0.3), 1e-8);
}
