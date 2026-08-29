
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

TEST(IntegrationSpecial,  MinresCgTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "minres");
    expect_contains(interp, "help", "cg");

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(IntegrationSpecial,  LegendrePScalar) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}
