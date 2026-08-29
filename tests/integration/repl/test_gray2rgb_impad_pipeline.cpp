
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

TEST(IntegrationRepl,  Gray2rgbImpadTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "gray2rgb(M)");
    expect_contains(interp, "help", "impad(M,pad");

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(IntegrationRepl,  DawsonxScalar) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}
