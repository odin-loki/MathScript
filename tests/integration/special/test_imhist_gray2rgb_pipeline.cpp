
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

TEST(IntegrationSpecial,  ImhistGray2rgbTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "imhist");
    expect_contains(interp, "help", "gray2rgb");

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);
}

TEST(IntegrationSpecial,  JacobiSdScalar) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.2, 0.3), 1e-8);
}
