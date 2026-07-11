#pragma once

namespace ms {

double norm_pdf(double x, double mu, double sigma);
double norm_cdf(double x, double mu, double sigma);
double norm_ppf(double p, double mu, double sigma);

double exp_pdf(double x, double lambda);
double exp_cdf(double x, double lambda);
double exp_ppf(double p, double lambda);

double binom_pdf(int k, int n, double p);
double binom_cdf(int k, int n, double p);

double pois_pdf(double k, double lambda);
double pois_cdf(double k, double lambda);

double chi2_pdf(double x, double df);
double chi2_cdf(double x, double df);
double chi2_ppf(double p, double df);

double uniform_pdf(double x, double a, double b);
double uniform_cdf(double x, double a, double b);

double t_pdf(double x, double df);
double t_cdf(double x, double df);
double t_ppf(double p, double df);

double gamma_pdf(double x, double shape, double scale);
double gamma_cdf(double x, double shape, double scale);
double gamma_ppf(double p, double shape, double scale);

double beta_pdf(double x, double alpha, double beta);
double beta_cdf(double x, double alpha, double beta);
double beta_ppf(double p, double alpha, double beta);

double f_pdf(double x, double d1, double d2);
double f_cdf(double x, double d1, double d2);
double f_ppf(double p, double d1, double d2);

/// Lognormal(mu, sigma): X = exp(Z) where Z ~ Normal(mu, sigma). mu/sigma are the mean/stddev
/// of the underlying normal, not of X itself. Support x > 0.
double lognormal_pdf(double x, double mu, double sigma);
double lognormal_cdf(double x, double mu, double sigma);
double lognormal_ppf(double p, double mu, double sigma);

/// Weibull(lambda, k): scale lambda > 0, shape k > 0. Support x >= 0. k=1 reduces to
/// Exponential(rate=1/lambda).
double weibull_pdf(double x, double lambda, double k);
double weibull_cdf(double x, double lambda, double k);
double weibull_ppf(double p, double lambda, double k);

/// Laplace(mu, b): location mu, scale b > 0 (double exponential distribution).
double laplace_pdf(double x, double mu, double b);
double laplace_cdf(double x, double mu, double b);
double laplace_ppf(double p, double mu, double b);

} // namespace ms
