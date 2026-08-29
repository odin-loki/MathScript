
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

TEST(IntegrationFem,  FemPoissonCfdAdvectionTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_poisson3d");
    expect_contains(interp, "help", "cfd_advection3d");

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(IntegrationFem,  DebyeScalar) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}
