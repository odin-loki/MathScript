// MathScript Integration Tests: REPL Interpreter – Wave 236 FEM 3D Poisson Pipeline

#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave236Pipeline, FemPoisson3dBinding) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 27u);
    EXPECT_GT(interp.state().matrices.at("u")(13, 0), 0.0);

    expect_contains(interp, "help", "fem_poisson3d");
}
