// MathScript Integration Tests: REPL framework bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

void expect_rng_seed_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_FALSE(result.has_value()) << cmd;
    const std::string message = ms::format_error(result.error());
    EXPECT_NE(message.find("izaac seed"), std::string::npos) << cmd << " error: " << message;
}

ms::Result<std::string> run(Interpreter& interp, const std::string& cmd) {
    return interp.execute(cmd);
}

double parse_scalar_output(const std::string& text) {
    return std::stod(Interpreter::trim(text));
}

} // namespace

TEST(FrameworksReplPipeline, FrameworkBindingsPipeline) {
    Interpreter interp;
    ms::izaac::clear_session();

    const auto entropy = run(interp, "gria_entropy([1,2,2,3,3,3], 4)");
    ASSERT_TRUE(entropy.has_value());
    const double h = parse_scalar_output(*entropy);
    EXPECT_GT(h, 0.0);
    EXPECT_TRUE(std::isfinite(h));
    expect_error(interp, "gria_entropy()");

    const auto matrix_alpha =
        run(interp, "gria_matrix_alpha([1,2;3,4], [2,4;6,8])");
    ASSERT_TRUE(matrix_alpha.has_value());
    const double alpha = parse_scalar_output(*matrix_alpha);
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    EXPECT_TRUE(std::isfinite(alpha));

    const auto is_critical = run(interp, "gria_is_critical(0.5, 0.05)");
    ASSERT_TRUE(is_critical.has_value());
    EXPECT_EQ(Interpreter::trim(*is_critical), "true");
    const auto not_critical = run(interp, "gria_is_critical(0.1)");
    ASSERT_TRUE(not_critical.has_value());
    EXPECT_EQ(Interpreter::trim(*not_critical), "false");

    const auto classify = run(interp, "gria_classify(0.5)");
    ASSERT_TRUE(classify.has_value());
    EXPECT_EQ(Interpreter::trim(*classify), "critical");

    const auto nig_fit = run(interp, "cypha_nig_fit([1,2,3,4])");
    ASSERT_TRUE(nig_fit.has_value());
    EXPECT_NE(nig_fit->find("mu="), std::string::npos);
    EXPECT_NE(nig_fit->find("delta="), std::string::npos);

    const auto nig_pdf = run(interp, "cypha_nig_pdf(2.5, 0, 1, 0, 1)");
    ASSERT_TRUE(nig_pdf.has_value());
    EXPECT_GT(parse_scalar_output(*nig_pdf), 0.0);

    const auto nig_cdf = run(interp, "cypha_nig_cdf(2.5, 0, 1, 0, 1)");
    ASSERT_TRUE(nig_cdf.has_value());
    const double cdf = parse_scalar_output(*nig_cdf);
    EXPECT_GE(cdf, 0.0);
    EXPECT_LE(cdf, 1.0);

    expect_rng_seed_error(interp, "cypha_nig_sample(0, 1, 0, 1, 5)");
    expect_rng_seed_error(interp, "izaac_estimate_pi(100)");
    expect_rng_seed_error(interp, "izaac_laplace_noise(5, 1, 1)");
    expect_rng_seed_error(interp, "izaac_gaussian_noise(5, 1, 1e-5, 1)");

    expect_ok(interp, "izaac seed 42");

    const auto nig_sample = run(interp, "cypha_nig_sample(0, 1, 0, 1, 5)");
    ASSERT_TRUE(nig_sample.has_value());
    EXPECT_NE(nig_sample->find("samples"), std::string::npos);

    const auto pi_est = run(interp, "izaac_estimate_pi(1000)");
    ASSERT_TRUE(pi_est.has_value());
    const double pi = parse_scalar_output(*pi_est);
    EXPECT_GT(pi, 0.0);
    EXPECT_LT(pi, 4.0);

    const auto laplace = run(interp, "izaac_laplace_noise(5, 1, 1)");
    ASSERT_TRUE(laplace.has_value());
    EXPECT_TRUE(std::isfinite(parse_scalar_output(*laplace)));

    const auto gaussian = run(interp, "izaac_gaussian_noise(5, 1, 1e-5, 1)");
    ASSERT_TRUE(gaussian.has_value());
    EXPECT_TRUE(std::isfinite(parse_scalar_output(*gaussian)));

    const auto hebbian =
        run(interp, "cellai_hebbian_update([0.1], [1], [0.8], 0.1)");
    ASSERT_TRUE(hebbian.has_value());
    EXPECT_NE(hebbian->find("w ="), std::string::npos);

    const auto energy =
        run(interp, "cellai_energy([1,2;3,4], [1;0], [1;2])");
    ASSERT_TRUE(energy.has_value());
    EXPECT_NEAR(parse_scalar_output(*energy), -5.0, 1e-9);

    expect_error(interp, "cellai_energy([1,2;3,4], [1;0])");
    expect_error(interp, "cypha_nig_pdf(1, 2, 3)");
}
