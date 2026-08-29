
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(IntegrationNumthy,  ComboConvergentsTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_next_perm");
    expect_contains(interp, "help", "numthy_convergents");

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);
}

TEST(IntegrationNumthy,  CplxJoukowskiInvScalar) {
    Interpreter interp;
    expect_ok(interp, "zmag = cplx_joukowski_inv(2, 1)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("zmag")));
}
