#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, prob_norm_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_norm_cdf(x,mu,sigma)");

    expect_ok(interp, "p = prob_norm_cdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("p"), 0.5, 1e-9);

    expect_contains(interp, "prob_norm_cdf(0, 0, 1)", "0.5");
}

TEST(ReplCommandsTest, prob_norm_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_norm_pdf(x,mu,sigma)");

    expect_ok(interp, "d = prob_norm_pdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 0.3989422804014327, 1e-6);

    expect_contains(interp, "prob_norm_pdf(0, 0, 1)", "0.398942");
}

TEST(ReplCommandsTest, prob_norm_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_norm_ppf(p,mu,sigma)");

    expect_ok(interp, "q = prob_norm_ppf(0.5, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("q"), 0.0, 1e-9);

    expect_contains(interp, "prob_norm_ppf(0.5, 0, 1)", "0");
}

TEST(ReplCommandsTest, prob_binom_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_binom_pdf(k,n,p)");

    expect_ok(interp, "pk = prob_binom_pdf(2, 4, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("pk"), 0.375, 1e-9);

    expect_contains(interp, "prob_binom_pdf(2, 4, 0.5)", "0.375");
}

TEST(ReplCommandsTest, prob_exp_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_exp_cdf(x,lambda)");

    expect_ok(interp, "c = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 1.0 - std::exp(-1.0), 1e-9);

    expect_contains(interp, "prob_exp_cdf(1, 1)", "0.632");
}

TEST(ReplCommandsTest, prob_binom_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_binom_cdf(k,n,p)");

    expect_ok(interp, "cdf = prob_binom_cdf(2, 4, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cdf"), 0.6875, 1e-9);

    expect_contains(interp, "prob_binom_cdf(2, 4, 0.5)", "0.6875");
}

TEST(ReplCommandsTest, prob_pois_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_pois_pdf(k,lambda)");

    expect_ok(interp, "pk = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pk"), 2.0 * std::exp(-2.0), 1e-9);

    expect_ok(interp, "prob_pois_pdf(2, 2)");
}

TEST(ReplCommandsTest, prob_uniform_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_uniform_cdf(x,a,b)");

    expect_ok(interp, "ucdf = prob_uniform_cdf(3, 0, 10)");
    EXPECT_NEAR(interp.state().scalars.at("ucdf"), 0.3, 1e-9);

    expect_contains(interp, "prob_uniform_cdf(3, 0, 10)", "0.3");
}

TEST(ReplCommandsTest, prob_pois_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_pois_cdf(k,lambda)");

    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_GT(interp.state().scalars.at("pc"), 0.0);
    EXPECT_LT(interp.state().scalars.at("pc"), 1.0);

    expect_ok(interp, "prob_pois_cdf(2, 2)");
}

TEST(ReplCommandsTest, prob_exp_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_exp_pdf(x,lambda)");

    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), std::exp(-1.0), 1e-9);

    expect_ok(interp, "prob_exp_pdf(1, 1)");
}

TEST(ReplCommandsTest, prob_chi2_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_chi2_cdf(x,df)");

    expect_ok(interp, "cc = prob_chi2_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), 0.682689492, 1e-3);

    expect_ok(interp, "prob_chi2_cdf(1, 1)");
}

TEST(ReplCommandsTest, prob_gamma_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_gamma_pdf(x,shape,scale)");

    expect_ok(interp, "gp = prob_gamma_pdf(2, 3, 1)");
    EXPECT_GT(interp.state().scalars.at("gp"), 0.0);

    expect_ok(interp, "prob_gamma_pdf(2, 3, 1)");
}

TEST(ReplCommandsTest, prob_t_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_t_cdf(x,df)");

    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), 0.5, 1e-6);

    expect_ok(interp, "prob_t_cdf(0, 5)");
}

TEST(ReplCommandsTest, prob_chi2_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_chi2_pdf(x,df)");

    expect_ok(interp, "cp = prob_chi2_pdf(1, 3)");
    EXPECT_GT(interp.state().scalars.at("cp"), 0.0);

    expect_ok(interp, "prob_chi2_pdf(1, 3)");
}

TEST(ReplCommandsTest, prob_uniform_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_uniform_pdf(x,a,b)");

    // uniform_pdf on [1,2] is constant 1/(2-1) = 1.
    const double ref = ms::uniform_pdf(1.5, 1.0, 2.0);
    EXPECT_NEAR(ref, 1.0, 1e-9);
    expect_ok(interp, "y = prob_uniform_pdf(1.5, 1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_uniform_pdf(1.5, 1, 2)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_t_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_t_pdf(x,df)");

    const double ref = ms::t_pdf(1.0, 5.0);
    expect_ok(interp, "y = prob_t_pdf(1, 5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_t_pdf(1, 5)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_t_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_t_ppf(p,df)");

    // The Student-t distribution is symmetric, so its median (p=0.5) is 0.
    const double ref = ms::t_ppf(0.5, 10.0);
    EXPECT_NEAR(ref, 0.0, 1e-9);
    expect_ok(interp, "y = prob_t_ppf(0.5, 10)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_t_ppf(0.5, 10)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_gamma_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_gamma_ppf(p,shape,scale)");

    const double ref = ms::gamma_ppf(0.5, 2.0, 1.0);
    expect_ok(interp, "y = prob_gamma_ppf(0.5, 2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_gamma_ppf(0.5, 2, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_beta_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_beta_ppf(p,alpha,beta)");

    // Beta(1,1) is the uniform distribution on [0,1], so its ppf is the identity.
    const double ref = ms::beta_ppf(0.5, 1.0, 1.0);
    EXPECT_NEAR(ref, 0.5, 1e-9);
    expect_ok(interp, "y = prob_beta_ppf(0.5, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_beta_ppf(0.5, 1, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_f_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_f_pdf(x,d1,d2)");

    const double ref = ms::f_pdf(1.0, 5.0, 5.0);
    expect_ok(interp, "y = prob_f_pdf(1, 5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_f_pdf(1, 5, 5)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_f_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_f_ppf(p,d1,d2)");

    const double ref = ms::f_ppf(0.5, 5.0, 5.0);
    expect_ok(interp, "y = prob_f_ppf(0.5, 5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_f_ppf(0.5, 5, 5)", std::to_string(ref));
}

TEST(ReplCommandsTest, prob_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_lognormal_pdf(x,mu,sigma)");
    expect_contains(interp, "help", "prob_weibull_cdf(x,lambda,k)");
    expect_contains(interp, "help", "prob_laplace_ppf(p,mu,b)");
    expect_contains(interp, "help", "prob_logistic_cdf(x,mu,s)");
    expect_contains(interp, "help", "prob_gumbel_ppf(p,mu,beta)");
    expect_contains(interp, "help", "prob_cauchy_pdf(x,x0,gamma)");
    expect_contains(interp, "help", "prob_pareto_cdf(x,x_m,alpha)");
    expect_contains(interp, "help", "prob_rayleigh_cdf(x,sigma)");
    expect_contains(interp, "help", "prob_gamma_cdf(x,shape,scale)");
    expect_contains(interp, "help", "prob_beta_cdf(x,alpha,beta)");
    expect_contains(interp, "help", "prob_f_cdf(x,d1,d2)");

    // Closed-form spot checks (assignment path).
    expect_ok(interp, "lnq = prob_lognormal_ppf(0.5, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lnq"), 1.0, 1e-9);  // median = exp(mu)

    expect_ok(interp, "wc = prob_weibull_cdf(1, 1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("wc"), 1.0 - std::exp(-1.0), 1e-9);

    expect_ok(interp, "lp = prob_laplace_pdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), 0.5, 1e-9);  // 1/(2b)

    expect_ok(interp, "lc = prob_logistic_cdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 0.5, 1e-9);

    expect_ok(interp, "gq = prob_gumbel_ppf(0.5, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("gq"), -std::log(std::log(2.0)), 1e-9);

    expect_ok(interp, "cp = prob_cauchy_pdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), 1.0 / 3.14159265358979323846, 1e-9);

    expect_ok(interp, "pc = prob_pareto_cdf(2, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), 0.5, 1e-9);  // 1-(x_m/x)^alpha

    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), 1.0 - std::exp(-0.5), 1e-9);

    expect_ok(interp, "bc = prob_beta_cdf(0.5, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bc"), 0.5, 1e-9);  // Beta(1,1)=U[0,1]

    expect_ok(interp, "gc = prob_gamma_cdf(0, 2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("gc"), 0.0, 1e-9);

    const double f_ref = ms::f_cdf(1.0, 5.0, 5.0);
    expect_ok(interp, "fc = prob_f_cdf(1, 5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("fc"), f_ref, 1e-9);

    // Direct-call path smoke.
    expect_contains(interp, "prob_laplace_cdf(0, 0, 1)", "0.5");
    expect_contains(interp, "prob_cauchy_cdf(0, 0, 1)", "0.5");
}

TEST(ReplCommandsTest, prob_chi2_exp_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_exp_ppf(p,lambda)");
    expect_contains(interp, "help", "prob_chi2_ppf(p,df)");

    const double p_exp = 0.6321205588289387;
    const double lambda = 1.0;
    const double x_exp = ms::exp_ppf(p_exp, lambda);
    EXPECT_NEAR(x_exp, 1.0, 1e-9);
    expect_ok(interp, "x = prob_exp_ppf(0.6321205588289387, 1)");
    EXPECT_NEAR(interp.state().scalars.at("x"), x_exp, 1e-9);
    expect_contains(interp, "prob_exp_ppf(0.6321205588289387, 1)", std::to_string(x_exp));

    expect_ok(interp, "q = prob_exp_ppf(0.5, 2)");
    const double q_exp = interp.state().scalars.at("q");
    expect_ok(interp, "c = prob_exp_cdf(q, 2)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 0.5, 1e-6);
    EXPECT_NEAR(ms::exp_cdf(q_exp, 2.0), 0.5, 1e-6);

    const double p_chi2 = 0.5;
    const double df = 3.0;
    const double x_chi2 = ms::chi2_ppf(p_chi2, df);
    expect_ok(interp, "y = prob_chi2_ppf(0.5, 3)");
    EXPECT_NEAR(interp.state().scalars.at("y"), x_chi2, 1e-9);
    expect_contains(interp, "prob_chi2_ppf(0.5, 3)", std::to_string(x_chi2));

    expect_ok(interp, "cc = prob_chi2_cdf(y, 3)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), 0.5, 1e-6);
    EXPECT_NEAR(ms::chi2_cdf(x_chi2, df), p_chi2, 1e-6);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar) {
    Interpreter interp;
    expect_ok(interp, "eq = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eq"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar) {
    Interpreter interp;
    expect_ok(interp, "q = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("q"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "eq = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eq"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_ppf_scalar) {
    Interpreter interp;
    expect_ok(interp, "rq = prob_rayleigh_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rq"), ms::rayleigh_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "eq = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eq"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_ppf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "rq = prob_rayleigh_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rq"), ms::rayleigh_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_ppf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_ppf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_pdf_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_t_ppf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_cdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_pdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_pois_cdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_exp_pdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_cdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_t_cdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}

TEST(ReplCommandsTest, prob_chi2_pdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}

TEST(ReplCommandsTest, prob_lognormal_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_lognormal_pdf(1, 0, 1)");
    expect_error_contains(interp, "prob_lognormal_pdf(1, 0, missing)", "prob_lognormal_pdf");
}

TEST(ReplCommandsTest, prob_lognormal_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_lognormal_cdf(1, 0, 1)");
    expect_error_contains(interp, "prob_lognormal_cdf(1, 0, missing)", "prob_lognormal_cdf");
}

TEST(ReplCommandsTest, prob_lognormal_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_lognormal_ppf(0.5, 0, 1)");
    expect_error_contains(interp, "prob_lognormal_ppf(0.5, 0, missing)", "prob_lognormal_ppf");
}

TEST(ReplCommandsTest, prob_weibull_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_weibull_pdf(1, 1, 2)");
    expect_error_contains(interp, "prob_weibull_pdf(1, 1, missing)", "prob_weibull_pdf");
}

TEST(ReplCommandsTest, prob_weibull_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_weibull_cdf(1, 1, 2)");
    expect_error_contains(interp, "prob_weibull_cdf(1, 1, missing)", "prob_weibull_cdf");
}

TEST(ReplCommandsTest, prob_weibull_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_weibull_ppf(0.5, 1, 2)");
    expect_error_contains(interp, "prob_weibull_ppf(0.5, 1, missing)", "prob_weibull_ppf");
}

TEST(ReplCommandsTest, prob_laplace_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_laplace_pdf(0, 0, 1)");
    expect_error_contains(interp, "prob_laplace_pdf(0, 0, missing)", "prob_laplace_pdf");
}

TEST(ReplCommandsTest, prob_laplace_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_laplace_ppf(0.5, 0, 1)");
    expect_error_contains(interp, "prob_laplace_ppf(0.5, 0, missing)", "prob_laplace_ppf");
}

TEST(ReplCommandsTest, prob_logistic_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_logistic_pdf(0, 0, 1)");
    expect_error_contains(interp, "prob_logistic_pdf(0, 0, missing)", "prob_logistic_pdf");
}

TEST(ReplCommandsTest, prob_logistic_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_logistic_cdf(0, 0, 1)");
    expect_error_contains(interp, "prob_logistic_cdf(0, 0, missing)", "prob_logistic_cdf");
}

TEST(ReplCommandsTest, prob_logistic_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_logistic_ppf(0.5, 0, 1)");
    expect_error_contains(interp, "prob_logistic_ppf(0.5, 0, missing)", "prob_logistic_ppf");
}

TEST(ReplCommandsTest, prob_gumbel_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_gumbel_pdf(0, 0, 1)");
    expect_error_contains(interp, "prob_gumbel_pdf(0, 0, missing)", "prob_gumbel_pdf");
}

TEST(ReplCommandsTest, prob_gumbel_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_gumbel_cdf(0, 0, 1)");
    expect_error_contains(interp, "prob_gumbel_cdf(0, 0, missing)", "prob_gumbel_cdf");
}

TEST(ReplCommandsTest, prob_gumbel_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_gumbel_ppf(0.5, 0, 1)");
    expect_error_contains(interp, "prob_gumbel_ppf(0.5, 0, missing)", "prob_gumbel_ppf");
}

TEST(ReplCommandsTest, prob_cauchy_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_cauchy_pdf(0, 0, 1)");
    expect_error_contains(interp, "prob_cauchy_pdf(0, 0, missing)", "prob_cauchy_pdf");
}

TEST(ReplCommandsTest, prob_cauchy_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_cauchy_ppf(0.5, 0, 1)");
    expect_error_contains(interp, "prob_cauchy_ppf(0.5, 0, missing)", "prob_cauchy_ppf");
}

TEST(ReplCommandsTest, prob_pareto_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_pareto_pdf(2, 1, 1)");
    expect_error_contains(interp, "prob_pareto_pdf(2, 1, missing)", "prob_pareto_pdf");
}

TEST(ReplCommandsTest, prob_pareto_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_pareto_cdf(2, 1, 1)");
    expect_error_contains(interp, "prob_pareto_cdf(2, 1, missing)", "prob_pareto_cdf");
}

TEST(ReplCommandsTest, prob_pareto_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_pareto_ppf(0.5, 1, 1)");
    expect_error_contains(interp, "prob_pareto_ppf(0.5, 1, missing)", "prob_pareto_ppf");
}

TEST(ReplCommandsTest, prob_gamma_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_gamma_cdf(0, 2, 1)");
    expect_error_contains(interp, "prob_gamma_cdf(0, 2, missing)", "prob_gamma_cdf");
}

TEST(ReplCommandsTest, prob_beta_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_beta_cdf(0.5, 1, 1)");
    expect_error_contains(interp, "prob_beta_cdf(0.5, 1, missing)", "prob_beta_cdf");
}

TEST(ReplCommandsTest, prob_f_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_f_cdf(1, 5, 5)");
    expect_error_contains(interp, "prob_f_cdf(1, 5, missing)", "prob_f_cdf");
}

TEST(ReplCommandsTest, prob_rayleigh_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_rayleigh_pdf(1, 1)");
    expect_error_contains(interp, "prob_rayleigh_pdf(1, missing)", "prob_rayleigh_pdf");
}

TEST(ReplCommandsTest, prob_rayleigh_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_rayleigh_cdf(1, 1)");
    expect_error_contains(interp, "prob_rayleigh_cdf(1, missing)", "prob_rayleigh_cdf");
}

TEST(ReplCommandsTest, prob_rayleigh_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_rayleigh_ppf(0.5, 1)");
    expect_error_contains(interp, "prob_rayleigh_ppf(0.5, missing)", "prob_rayleigh_ppf");
}

TEST(ReplCommandsTest, prob_norm_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_norm_cdf(0, 0, 1)");
    expect_error_contains(interp, "prob_norm_cdf(0, 0, missing)", "expected prob_norm_cdf(x,mu,sigma)");
}

TEST(ReplCommandsTest, prob_norm_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_norm_pdf(0, 0, 1)");
    expect_error_contains(interp, "prob_norm_pdf(0, 0, missing)", "expected prob_norm_pdf(x,mu,sigma)");
}

TEST(ReplCommandsTest, prob_norm_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_norm_ppf(0.5, 0, 1)");
    expect_error_contains(interp, "prob_norm_ppf(0.5, 0, missing)",
                          "expected prob_norm_ppf(p,mu,sigma)");
}

TEST(ReplCommandsTest, prob_binom_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_binom_pdf(2, 4, 0.5)");
    expect_error_contains(interp, "prob_binom_pdf(2, 4, missing)",
                          "expected prob_binom_pdf(k, n, p)");
    expect_error_contains(interp, "prob_binom_pdf(1.5, 4, 0.5)", "expected integer k and n");
}

TEST(ReplCommandsTest, prob_binom_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_binom_cdf(2, 4, 0.5)");
    expect_error_contains(interp, "prob_binom_cdf(2, 4, missing)",
                          "expected prob_binom_cdf(k, n, p)");
    expect_error_contains(interp, "prob_binom_cdf(1.5, 4, 0.5)", "expected integer k and n");
}

TEST(ReplCommandsTest, prob_exp_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_exp_cdf(1, 1)");
    expect_error_contains(interp, "prob_exp_cdf(1, missing)", "expected prob_exp_cdf(x,lambda)");
}

TEST(ReplCommandsTest, prob_exp_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_exp_pdf(1, 1)");
    expect_error_contains(interp, "prob_exp_pdf(1, missing)", "expected prob_exp_pdf(x,lambda)");
}

TEST(ReplCommandsTest, prob_exp_ppf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_exp_ppf(0.5, 1)");
    expect_error_contains(interp, "prob_exp_ppf(0.5, missing)", "expected prob_exp_ppf(p,lambda)");
}

TEST(ReplCommandsTest, prob_pois_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_pois_pdf(2, 2)");
    expect_error_contains(interp, "prob_pois_pdf(2, missing)", "expected prob_pois_pdf(k,lambda)");
}

TEST(ReplCommandsTest, prob_pois_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_pois_cdf(2, 2)");
    expect_error_contains(interp, "prob_pois_cdf(2, missing)", "expected prob_pois_cdf(k,lambda)");
}

TEST(ReplCommandsTest, prob_uniform_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_uniform_cdf(3, 0, 10)");
    expect_error_contains(interp, "prob_uniform_cdf(3, 0, missing)",
                          "expected prob_uniform_cdf(x, a, b)");
}

TEST(ReplCommandsTest, prob_uniform_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_uniform_pdf(1.5, 1, 2)");
    expect_error_contains(interp, "prob_uniform_pdf(1.5, 1, missing)",
                          "expected prob_uniform_pdf(x,a,b)");
}

TEST(ReplCommandsTest, prob_gamma_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_gamma_pdf(2, 3, 1)");
    expect_error_contains(interp, "prob_gamma_pdf(2, 3, missing)",
                          "expected prob_gamma_pdf(x, shape, scale)");
}

TEST(ReplCommandsTest, prob_t_pdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_t_pdf(1, 5)");
    expect_error_contains(interp, "prob_t_pdf(1, missing)", "expected prob_t_pdf(x,df)");
}

TEST(ReplCommandsTest, prob_laplace_cdf_noassign) {
    Interpreter interp;
    expect_ok(interp, "prob_laplace_cdf(0, 0, 1)");
    expect_error_contains(interp, "prob_laplace_cdf(0, 0, missing)",
                          "expected prob_laplace_cdf(x,mu,b)");
}
