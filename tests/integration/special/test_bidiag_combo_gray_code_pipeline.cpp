
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

TEST(IntegrationSpecial,  LinalgCombo) {
    Interpreter interp;

    expect_contains(interp, "help", "bidiag");
    expect_contains(interp, "help", "combo_gray_code");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 2u);

    expect_ok(interp, "gc = combo_gray_code(3)");
    EXPECT_GT(interp.state().matrices.at("gc").rows(), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_EQ(interp.state().matrices.at("X").rows(), 2u);
}

TEST(IntegrationSpecial,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "tn = chebyshev_tn(3, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("tn"), ms::chebyshev_tn(3, 0, 0.5), 1e-9);

    expect_ok(interp, "un = chebyshev_un(2, 1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("un"), ms::chebyshev_un(2, 1, 0.25), 1e-9);
}
