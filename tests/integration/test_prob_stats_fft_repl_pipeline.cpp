// MathScript Integration Tests: REPL prob/stats/fft/optim backlog bindings (Waves 182-185)

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"
#include "ms/stats/stats.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " -> " << ms::format_error(result.error());
}

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

ms::Result<std::string> run(Interpreter& interp, const std::string& cmd) {
    return interp.execute(cmd);
}

} // namespace

TEST(ProbStatsFftReplPipeline, GammaCdfReplBinding) {
    Interpreter interp;
    expect_ok(interp, "gc = gamma_cdf(1, 1, 1)");
    const double expected = 1.0 - std::exp(-1.0);
    EXPECT_NEAR(interp.state().scalars.at("gc"), expected, 1e-6);
    expect_error(interp, "gamma_cdf(1, )");
}

TEST(ProbStatsFftReplPipeline, BetaPdfCdfReplBindings) {
    Interpreter interp;
    expect_ok(interp, "bp = beta_pdf(0.5, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 1.0, 1e-9);
    expect_ok(interp, "bc = beta_cdf(0.5, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bc"), 0.5, 1e-9);
}

TEST(ProbStatsFftReplPipeline, FPdfCdfReplBindings) {
    Interpreter interp;
    expect_ok(interp, "fp = f_pdf(1, 5, 10)");
    EXPECT_GT(interp.state().scalars.at("fp"), 0.0);
    expect_ok(interp, "fc = f_cdf(0, 5, 10)");
    EXPECT_NEAR(interp.state().scalars.at("fc"), 0.0, 1e-10);
}

TEST(ProbStatsFftReplPipeline, KruskalWallisReplBinding) {
    Interpreter interp;
    expect_ok(interp, "kw = kruskal_wallis([10,11,12; 20,21,22; 30,31,32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
    const auto& m = interp.state().matrices.at("kw");
    ASSERT_EQ(m.rows(), 3u);
    EXPECT_GT(m(0, 0), 5.0);
    EXPECT_NEAR(m(1, 0), 2.0, 1e-9);
    EXPECT_LT(m(2, 0), 0.05);
    const auto direct = run(interp, "kruskal_wallis([5,6,7; 5.1,6.1,7.1; 4.9,5.9,6.9])");
    ASSERT_TRUE(direct.has_value());
    EXPECT_NE(direct->find("h_stat"), std::string::npos);
}

TEST(ProbStatsFftReplPipeline, CmaesReplBinding) {
    Interpreter interp;
    const auto out = run(interp, "cmaes(\"x0*x0+x1*x1\", [2, 3], 0.5, 500, 42)");
    ASSERT_TRUE(out.has_value()) << ms::format_error(out.error());
    EXPECT_NE(out->find("f_val"), std::string::npos);
    const auto f_pos = out->find("f_val = ");
    ASSERT_NE(f_pos, std::string::npos);
    const double f_val = std::stod((*out).substr(f_pos + 8));
    EXPECT_LT(f_val, 1e-3);
    expect_error(interp, "cmaes(\"x0*\", [1], 0.5, 100)");
}

TEST(ProbStatsFftReplPipeline, Ifft2ReplBinding) {
    Interpreter interp;
    expect_ok(interp, "S = [1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    const auto& m = interp.state().matrices.at("back");
    ASSERT_EQ(m.rows(), 4u);
    ASSERT_EQ(m.cols(), 2u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(m(i, 0), static_cast<double>(i + 1), 1e-5);
        EXPECT_NEAR(m(i, 1), 0.0, 1e-5);
    }
}

TEST(ProbStatsFftReplPipeline, Idst2ReplBinding) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "back = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    const auto& m = interp.state().matrices.at("back");
    ASSERT_EQ(m.rows(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(m(i, 0), static_cast<double>(i + 1), 1e-5);
    }
}
