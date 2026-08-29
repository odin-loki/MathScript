
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

TEST(IntegrationSpecial,  Poisson1d) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_poisson1d");

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(IntegrationSpecial,  BesselIScalar) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}
