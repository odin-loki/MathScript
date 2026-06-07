#pragma once

namespace ms {

double norm_pdf(double x, double mu, double sigma);
double norm_cdf(double x, double mu, double sigma);
double norm_ppf(double p, double mu, double sigma);

double exp_pdf(double x, double lambda);
double exp_cdf(double x, double lambda);

double binom_pdf(int k, int n, double p);
double binom_cdf(int k, int n, double p);

double pois_pdf(double k, double lambda);
double pois_cdf(double k, double lambda);

double chi2_pdf(double x, double df);
double chi2_cdf(double x, double df);

double uniform_pdf(double x, double a, double b);
double uniform_cdf(double x, double a, double b);

double t_pdf(double x, double df);
double t_cdf(double x, double df);

double gamma_pdf(double x, double shape, double scale);

} // namespace ms
