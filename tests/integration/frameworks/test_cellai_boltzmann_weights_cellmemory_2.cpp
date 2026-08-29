
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

TEST(IntegrationFrameworks,  CellaiBoltzmannCyphaTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cellai_boltzmann_weights");
    expect_contains(interp, "help", "cellai_cell_to_cypha_features");

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm334, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm334, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm334, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(IntegrationFrameworks,  HermiteHnScalar) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}
