
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

TEST(IntegrationCompress,  RleCryptoFromHex) {
    Interpreter interp;
    expect_contains(interp, "help", "run_length_encode_vec");
    expect_contains(interp, "help", "run_length_decode_vec");
    expect_contains(interp, "help", "crypto_from_hex");

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(IntegrationCompress,  EllipEIncScalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}
