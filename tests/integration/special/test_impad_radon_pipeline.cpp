
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

TEST(IntegrationSpecial,  ImpadRadonTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "impad");
    expect_contains(interp, "help", "radon");

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(IntegrationSpecial,  JacobiNcScalar) {
    Interpreter interp;
    expect_ok(interp, "jnc = jacobi_nc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jnc"), ms::jacobi_nc(0.2, 0.3), 1e-8);
}
