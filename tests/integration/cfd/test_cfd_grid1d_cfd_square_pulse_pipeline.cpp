
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

TEST(IntegrationCfd,  CfdGrid1dSquarePulse) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_grid1d");
    expect_contains(interp, "help", "cfd_square_pulse");

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(IntegrationCfd,  Theta1Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}
