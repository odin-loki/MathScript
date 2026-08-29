
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

TEST(IntegrationSpecial,  CfdPulseDaggerTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_square_pulse_2d");
    expect_contains(interp, "help", "quantum_dagger");

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
}

TEST(IntegrationSpecial,  JacobiCsScalar) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.2, 0.3), 1e-8);
}
