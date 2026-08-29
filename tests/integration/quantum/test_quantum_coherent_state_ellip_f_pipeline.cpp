
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

TEST(IntegrationQuantum,  QuantumCoherentState) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_coherent_state(alpha_re,alpha_im,n_max)");

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    const auto& s = interp.state().matrices.at("s");
    EXPECT_EQ(s.rows(), 21u);
    EXPECT_EQ(s.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < s.rows(); ++i) {
        norm_sq += s(i, 0) * s(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);
}

TEST(IntegrationQuantum,  EllipFScalar) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}
