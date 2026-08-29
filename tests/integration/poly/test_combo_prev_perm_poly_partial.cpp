
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

TEST(IntegrationPoly,  PrevPermPartialTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_prev_perm");
    expect_contains(interp, "help", "poly_partial_fractions");

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(IntegrationPoly,  StruveHnScalar) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}
