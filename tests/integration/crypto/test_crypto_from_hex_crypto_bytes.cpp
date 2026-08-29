
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

TEST(IntegrationCrypto,  CryptoBwtEquityTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_bytes_to_hex");
    expect_contains(interp, "help", "bwt_decode_vec");
    expect_contains(interp, "help", "run_backtest_equity");

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);
}

TEST(IntegrationCrypto,  PolylogScalar) {
    Interpreter interp;
    expect_ok(interp, "p = polylog(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("p"), ms::polylog(2, 0.5), 1e-8);
}
