
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

TEST(IntegrationImage,  BilateralCannyTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "bilateral(M,sigma_s,sigma_r)");
    expect_contains(interp, "help", "canny(M,low,high)");

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 5u);
}

TEST(IntegrationImage,  JacobiNcScalar) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}
