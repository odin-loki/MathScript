
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

TEST(IntegrationSpecial,  FftshiftIfftshiftTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "fftshift");
    expect_contains(interp, "help", "ifftshift");

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(IntegrationSpecial,  JacobiNcScalar) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}
