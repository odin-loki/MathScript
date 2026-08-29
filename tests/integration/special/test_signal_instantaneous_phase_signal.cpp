
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

TEST(IntegrationSpecial,  PhaseUnwrapTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_instantaneous_phase");
    expect_contains(interp, "help", "signal_unwrap");

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x)");
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);
}

TEST(IntegrationSpecial,  JacobiDnScalar) {
    Interpreter interp;
    expect_ok(interp, "dn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}
