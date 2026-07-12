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

/// Logistic(mu, s): location mu, scale s > 0. Support all reals. Sigmoid CDF:
/// 1 / (1 + exp(-(x-mu)/s)). Used for growth curves, dose-response, and logistic
/// regression link functions.
double logistic_pdf(double x, double mu, double s);
double logistic_cdf(double x, double mu, double s);
double logistic_ppf(double p, double mu, double s);

/// Gumbel(mu, beta): location mu, scale beta > 0. Support all reals. The standard
/// extreme-value distribution (Type I), used for modeling the distribution of maxima
/// (e.g. maximum river levels, extreme rainfall). CDF: exp(-exp(-(x-mu)/beta)).
double gumbel_pdf(double x, double mu, double beta);
double gumbel_cdf(double x, double mu, double beta);
double gumbel_ppf(double p, double mu, double beta);

/// Cauchy(x0, gamma): location x0, scale gamma > 0. Support all reals. Heavy-tailed
/// distribution with UNDEFINED mean/variance (the classic pathological example). PDF:
/// 1 / (pi*gamma*(1+((x-x0)/gamma)^2)). CDF: 1/pi * atan((x-x0)/gamma) + 0.5.
double cauchy_pdf(double x, double x0, double gamma);
double cauchy_cdf(double x, double x0, double gamma);
double cauchy_ppf(double p, double x0, double gamma);

/// Pareto(x_m, alpha): scale (minimum value) x_m > 0, shape alpha > 0. Support x >= x_m.
/// Classic power-law/"80-20 rule" distribution. PDF: alpha*x_m^alpha / x^(alpha+1) for
/// x >= x_m, 0 otherwise. CDF: 1 - (x_m/x)^alpha for x >= x_m, 0 otherwise.
double pareto_pdf(double x, double x_m, double alpha);
double pareto_cdf(double x, double x_m, double alpha);
double pareto_ppf(double p, double x_m, double alpha);

/// Rayleigh(sigma): scale sigma > 0. Support x >= 0. The distribution of the magnitude
/// of a 2D vector whose components are i.i.d. Normal(0, sigma) (equivalently
/// Weibull(lambda=sigma*sqrt(2), k=2)); used for wind speed, wave height, and
/// signal-magnitude modeling. PDF: (x/sigma^2)*exp(-x^2/(2*sigma^2)). CDF:
/// 1 - exp(-x^2/(2*sigma^2)).
double rayleigh_pdf(double x, double sigma);
double rayleigh_cdf(double x, double sigma);
double rayleigh_ppf(double p, double sigma);

} // namespace ms
