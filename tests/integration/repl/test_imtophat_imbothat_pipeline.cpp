
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

TEST(IntegrationRepl,  MorphTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "imtophat(M[,k])");
    expect_contains(interp, "help", "imbothat(M[,k])");
    expect_contains(interp, "help", "imgradient_morph(M[,k])");

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(IntegrationRepl,  AiryAipScalar) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}
