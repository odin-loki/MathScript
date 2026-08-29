
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

TEST(IntegrationSpecial,  FemPoissonCfdAdv1d) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_poisson2d");
    expect_contains(interp, "help", "cfd_advection1d");

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(IntegrationSpecial,  ChebyshevWScalar) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}
