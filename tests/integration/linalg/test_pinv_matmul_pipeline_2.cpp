
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

TEST(IntegrationLinalg,  PinvNullOrthKronTail18) {
    Interpreter interp;
    expect_contains(interp, "help", "pinv");
    expect_contains(interp, "help", "cplx_joukowski_inv");

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    expect_ok(interp, "AP = matmul(A, P)");
    const auto& ap = interp.state().matrices.at("AP");
    EXPECT_NEAR(ap(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(ap(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(ap(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(ap(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    const auto& n = interp.state().matrices.at("N");
    EXPECT_EQ(n.rows(), 4u);
    EXPECT_GE(n.cols(), 1u);
    expect_ok(interp, "WN = matmul(W, N)");
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    const auto& k = interp.state().matrices.at("K");
    EXPECT_EQ(k.rows(), 4u);
    EXPECT_EQ(k.cols(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(k(i, j), (i == j) ? 1.0 : 0.0, 1e-8);
        }
    }
}

TEST(IntegrationLinalg,  CplxJoukowskiInvScalar) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}
