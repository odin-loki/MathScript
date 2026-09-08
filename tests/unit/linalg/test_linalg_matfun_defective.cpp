// Matrix functions at repeated / clustered eigenvalues.
//
// The unblocked Parlett recurrence divides by T(j,j) - T(i,i), and when two
// eigenvalues coincide it used to take a shortcut: if the numerator was also
// negligible it set F(i,j) = 0 and called the block "decoupled". That is not
// what happens at a repeated eigenvalue -- f(A) there depends on the DERIVATIVES
// of f -- so logm, sinm and cosm returned a plausible but wrong matrix on every
// defective input. For J = [[2,1],[0,2]] logm returned diag(ln 2, ln 2), whose
// exponential is diag(2, 2), not J.
//
// The blocked Schur-Parlett fixes it, and every expectation below is an exact
// closed form or an identity the implementation never uses.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "ms/linalg/linalg.hpp"

namespace {

using ms::Matrix;

double max_abs_diff(const Matrix<double>& a, const Matrix<double>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
        }
    }
    return worst;
}

// J = [[2,1],[0,2]]: a single Jordan block, the smallest defective matrix.
Matrix<double> jordan2() {
    Matrix<double> j(2, 2);
    j(0, 0) = 2.0;
    j(0, 1) = 1.0;
    j(1, 0) = 0.0;
    j(1, 1) = 2.0;
    return j;
}

Matrix<double> identity(std::size_t n) {
    Matrix<double> i(n, n);
    for (std::size_t k = 0; k < n; ++k) i(k, k) = 1.0;
    return i;
}

}  // namespace

TEST(LinalgMatfunDefective, LogmOfAJordanBlockMatchesTheClosedForm) {
    // log(lambda I + N) = ln(lambda) I + N/lambda - N^2/(2 lambda^2) + ...
    // For the 2x2 block N^2 = 0, so the answer is exactly [[ln 2, 1/2],[0, ln 2]].
    const auto l = ms::logm(jordan2());
    ASSERT_TRUE(l.has_value());
    Matrix<double> want(2, 2);
    want(0, 0) = std::log(2.0);
    want(0, 1) = 0.5;
    want(1, 0) = 0.0;
    want(1, 1) = std::log(2.0);
    EXPECT_LT(max_abs_diff(*l, want), 1e-14);

    // The defining property. Before the fix this was diag(2,2).
    const auto back = ms::expm(*l);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT(max_abs_diff(*back, jordan2()), 1e-13);
}

TEST(LinalgMatfunDefective, SinmAndCosmOfAJordanBlockMatchTheClosedForm) {
    const auto s = ms::sinm(jordan2());
    const auto c = ms::cosm(jordan2());
    ASSERT_TRUE(s.has_value());
    ASSERT_TRUE(c.has_value());

    Matrix<double> want_s(2, 2);
    want_s(0, 0) = std::sin(2.0);
    want_s(0, 1) = std::cos(2.0);  // d/dx sin x at 2
    want_s(1, 1) = std::sin(2.0);
    Matrix<double> want_c(2, 2);
    want_c(0, 0) = std::cos(2.0);
    want_c(0, 1) = -std::sin(2.0);
    want_c(1, 1) = std::cos(2.0);
    EXPECT_LT(max_abs_diff(*s, want_s), 1e-14);
    EXPECT_LT(max_abs_diff(*c, want_c), 1e-14);

    // sin^2 + cos^2 == I. The old answers were diag(sin 2, sin 2) and
    // diag(cos 2, cos 2), which satisfy this identity too -- so it alone would
    // not have caught the bug; the closed forms above are what does.
    const Matrix<double> sum = (*s) * (*s) + (*c) * (*c);
    EXPECT_LT(max_abs_diff(sum, identity(2)), 1e-14);
}

TEST(LinalgMatfunDefective, LogmOfAThreeByThreeJordanBlock) {
    // log(2I + N) with N the shift: ln2 I + N/2 - N^2/8, and N^3 = 0.
    Matrix<double> a(3, 3);
    for (std::size_t i = 0; i < 3; ++i) a(i, i) = 2.0;
    a(0, 1) = 1.0;
    a(1, 2) = 1.0;

    const auto l = ms::logm(a);
    ASSERT_TRUE(l.has_value());
    Matrix<double> want(3, 3);
    for (std::size_t i = 0; i < 3; ++i) want(i, i) = std::log(2.0);
    want(0, 1) = 0.5;
    want(1, 2) = 0.5;
    want(0, 2) = -1.0 / 8.0;
    EXPECT_LT(max_abs_diff(*l, want), 1e-14);

    const auto back = ms::expm(*l);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT(max_abs_diff(*back, a), 1e-13);
}

TEST(LinalgMatfunDefective, TwoClustersInterleavedOnTheDiagonalAreReordered) {
    // The eigenvalues 2, 7, 2, 7 alternate, so the two groups are not contiguous
    // and the Schur form has to be reordered before it can be blocked.
    Matrix<double> a(4, 4);
    a(0, 0) = 2.0; a(1, 1) = 7.0; a(2, 2) = 2.0; a(3, 3) = 7.0;
    a(0, 1) = 1.0; a(0, 2) = 1.0; a(0, 3) = 2.0;
    a(1, 2) = 3.0; a(1, 3) = 1.0; a(2, 3) = 1.0;

    const auto l = ms::logm(a);
    ASSERT_TRUE(l.has_value());
    const auto back = ms::expm(*l);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT(max_abs_diff(*back, a), 1e-11);

    const auto s = ms::sinm(a);
    const auto c = ms::cosm(a);
    ASSERT_TRUE(s.has_value());
    ASSERT_TRUE(c.has_value());
    const Matrix<double> sum = (*s) * (*s) + (*c) * (*c);
    EXPECT_LT(max_abs_diff(sum, identity(4)), 1e-12);
}

TEST(LinalgMatfunDefective, SeparatedSpectraAreUnchanged) {
    // The blocked path must not engage for a well-separated spectrum; this pins
    // the plain Parlett recurrence that was already there.
    Matrix<double> a(3, 3);
    a(0, 0) = 1.0; a(1, 1) = 3.0; a(2, 2) = 7.0;
    a(0, 1) = 2.0; a(0, 2) = 1.0; a(1, 2) = 4.0;

    const auto l = ms::logm(a);
    ASSERT_TRUE(l.has_value());
    const auto back = ms::expm(*l);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT(max_abs_diff(*back, a), 1e-12);
}

TEST(LinalgMatfunDefective, PlainFunmReportsTheCaseItCannotAnswer) {
    // funm receives f but not its derivatives, and f alone does not determine
    // f(A) at a repeated eigenvalue. Reporting is the only honest answer -- what
    // it must NOT do is return the diagonal-only matrix it used to.
    const auto bad = ms::funm(jordan2(), std::function<double(double)>(
                                             [](double x) { return std::log(x); }));
    EXPECT_FALSE(bad.has_value());

    // A separated spectrum still works through funm.
    Matrix<double> sep(2, 2);
    sep(0, 0) = 2.0; sep(0, 1) = 1.0; sep(1, 1) = 5.0;
    const auto ok = ms::funm(sep, std::function<double(double)>(
                                      [](double x) { return std::log(x); }));
    ASSERT_TRUE(ok.has_value());
    const auto back = ms::expm(*ok);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT(max_abs_diff(*back, sep), 1e-12);
}

TEST(LinalgMatfunDefective, FunmTaylorAnswersWhatFunmCannot) {
    // The Taylor coefficients of log about x: f^(k)(x)/k! = (-1)^(k+1)/(k x^k).
    const auto coeffs = std::function<double(double, unsigned)>([](double x, unsigned k) {
        if (k == 0) return std::log(x);
        const double sign = (k % 2 == 1) ? 1.0 : -1.0;
        return sign / (static_cast<double>(k) * std::pow(x, static_cast<double>(k)));
    });

    const auto l = ms::funm_taylor(jordan2(), coeffs);
    ASSERT_TRUE(l.has_value());
    Matrix<double> want(2, 2);
    want(0, 0) = std::log(2.0);
    want(0, 1) = 0.5;
    want(1, 1) = std::log(2.0);
    EXPECT_LT(max_abs_diff(*l, want), 1e-14);

    // exp, whose coefficients are exp(x)/k!, on the same block: the closed form
    // is exp(2) * [[1, 1], [0, 1]].
    const auto exp_coeffs = std::function<double(double, unsigned)>([](double x, unsigned k) {
        return std::exp(x - std::lgamma(static_cast<double>(k) + 1.0));
    });
    const auto e = ms::funm_taylor(jordan2(), exp_coeffs);
    ASSERT_TRUE(e.has_value());
    Matrix<double> want_e(2, 2);
    want_e(0, 0) = std::exp(2.0);
    want_e(0, 1) = std::exp(2.0);
    want_e(1, 1) = std::exp(2.0);
    EXPECT_LT(max_abs_diff(*e, want_e), 1e-12);

    // Contract checks.
    EXPECT_FALSE(ms::funm_taylor(jordan2(), std::function<double(double, unsigned)>{}).has_value());
    Matrix<double> oblong(2, 3);
    EXPECT_FALSE(ms::funm_taylor(oblong, coeffs).has_value());

    // A symmetric matrix takes the exact eigenbasis path, not the Schur path.
    Matrix<double> sym(2, 2);
    sym(0, 0) = 4.0; sym(0, 1) = 1.0; sym(1, 0) = 1.0; sym(1, 1) = 4.0;
    const auto ls = ms::funm_taylor(sym, coeffs);
    ASSERT_TRUE(ls.has_value());
    const auto back = ms::expm(*ls);
    ASSERT_TRUE(back.has_value());
    EXPECT_LT(max_abs_diff(*back, sym), 1e-12);
}

TEST(LinalgMatfunDefective, ComplexEigenvaluesAreStillReported) {
    // A real rotation has a complex-conjugate pair, which a real-arithmetic
    // evaluation cannot represent; the blocking must not paper over that.
    Matrix<double> rot(2, 2);
    rot(0, 0) = 0.0; rot(0, 1) = -1.0; rot(1, 0) = 1.0; rot(1, 1) = 0.0;
    EXPECT_FALSE(ms::logm(rot).has_value());
    EXPECT_FALSE(ms::sinm(rot).has_value());
    EXPECT_FALSE(ms::cosm(rot).has_value());
}

TEST(LinalgMatfunDefective, NonPositiveEigenvalueStillHasNoRealLogarithm) {
    Matrix<double> a(2, 2);
    a(0, 0) = -2.0; a(0, 1) = 1.0; a(1, 1) = -2.0;  // defective AND negative
    EXPECT_FALSE(ms::logm(a).has_value());
}

TEST(LinalgMatfunDefective, RandomDefectiveMatricesRoundTripThroughExpm) {
    // Upper triangular with two repeated eigenvalues and random strictly-upper
    // entries: exp(log(A)) == A is an identity the implementation never uses.
    std::mt19937 rng(20260908u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double worst = 0.0;
    for (int trial = 0; trial < 40; ++trial) {
        Matrix<double> a(4, 4);
        const double diag[4] = {3.0, 3.0, 8.0, 8.0};
        for (std::size_t i = 0; i < 4; ++i) {
            a(i, i) = diag[i];
            for (std::size_t j = i + 1; j < 4; ++j) a(i, j) = u(rng);
        }
        const auto l = ms::logm(a);
        ASSERT_TRUE(l.has_value()) << "trial " << trial;
        const auto back = ms::expm(*l);
        ASSERT_TRUE(back.has_value()) << "trial " << trial;
        worst = std::max(worst, max_abs_diff(*back, a));
    }
    EXPECT_LT(worst, 1e-10);
}
