
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

TEST(IntegrationNumthy,  ConvergentsImcropTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_convergents(cf)");
    expect_contains(interp, "help", "imcrop(M,r0,c0,r1,c1)");

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(IntegrationNumthy,  JacobiCdScalar) {
    Interpreter interp;
    expect_ok(interp, "jcd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}
