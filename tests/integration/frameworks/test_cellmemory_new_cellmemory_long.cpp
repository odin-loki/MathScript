
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

TEST(IntegrationFrameworks,  CellmemoryPartialTrace) {
    Interpreter interp;
    expect_contains(interp, "help", "cellmemory_long_term_state");
    expect_contains(interp, "help", "quantum_partial_trace");

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(IntegrationFrameworks,  AngerJScalar) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}
