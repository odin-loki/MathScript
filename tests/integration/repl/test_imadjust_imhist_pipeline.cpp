
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

TEST(IntegrationRepl,  ImadjustImhistTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "imadjust(M,in_lo,in_hi");
    expect_contains(interp, "help", "imhist(M[,nbins])");

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(IntegrationRepl,  MathieuBScalar) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}
