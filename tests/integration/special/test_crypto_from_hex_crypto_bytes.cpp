
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

TEST(IntegrationSpecial,  CryptoHexBwtDecode) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_bytes_to_hex");
    expect_contains(interp, "help", "bwt_decode_vec");

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(IntegrationSpecial,  StruveLScalar) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-8);
}
