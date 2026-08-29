
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

TEST(IntegrationFrameworks,  CellaiBoltzmannCypha) {
    Interpreter interp;
    expect_contains(interp, "help", "cellai_boltzmann_weights");
    expect_contains(interp, "help", "cellai_cell_to_cypha_features");

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm691, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm691, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm691, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(IntegrationFrameworks,  BesselZeroYnuScalar) {
    Interpreter interp;
    expect_ok(interp, "yz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}
