// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: `cheby2` is only ever designed at order 2.
//
// A sample of 24 mutants at seed 73 against the eleven suites that cover
// `src/signal/signal.cpp` scored 83.3% -- the highest first-run score of the
// files measured -- and all four survivors reduce to one sentence. Every
// cheby2 test in the tree passes order 2, and:
//
//   `cheb2ap`'s ODD-order branch places the stopband zeros in two loops that
//   skip m = 0 (where the zero would be at infinity). Both loop bounds mutate
//   freely -- `m = -order + 1` to `-order - 1`, and `m = 2` to `m = 3` --
//   because no test ever designs an odd-order filter. The even-order branch
//   three lines above them is asserted and its mutants die.
//
//   `lp2hp_zpk`'s gain normalisation multiplies by prod(-p) over the poles.
//   Turning that into prod(p) multiplies the gain by (-1)^order: identical for
//   even order, a SIGN FLIP for odd. Order 2 cannot see it, and neither can any
//   assertion on |H| -- only a signed one.
//
// So this file designs orders 2 through 8 and asserts what Chebyshev type II
// means, rather than reference coefficients:
//
//   |H| at the stopband edge is EXACTLY 10^(-rs/20)   -- the definition
//   H at the passband reference point is EXACTLY +1   -- signed, not magnitude
//   the stopband stays under its target across the band
//   the design is stable, so the impulse response decays

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/signal/signal.hpp"

using namespace ms;

namespace {

// H(e^{jw}) from the coefficients, by direct evaluation of both polynomials.
std::complex<double> response(const IirCoeffs& c, double w) {
    std::complex<double> num{0.0, 0.0};
    std::complex<double> den{0.0, 0.0};
    const std::complex<double> step = std::exp(std::complex<double>(0.0, -w));
    std::complex<double> z{1.0, 0.0};
    for (const double b : c.b) {
        num += b * z;
        z *= step;
    }
    z = std::complex<double>{1.0, 0.0};
    for (const double a : c.a) {
        den += a * z;
        z *= step;
    }
    return num / den;
}

// The impulse response, by running the difference equation. A stable design's
// tail is small; an unstable one's grows without bound.
double impulse_tail(const IirCoeffs& c, int steps) {
    std::vector<double> y(static_cast<std::size_t>(steps), 0.0);
    double worst_tail = 0.0;
    for (int n = 0; n < steps; ++n) {
        double acc = (n == 0) ? c.b[0] : 0.0;
        for (std::size_t k = 1; k < c.b.size(); ++k) {
            if (n >= static_cast<int>(k)) {
                acc += c.b[k] * ((n - static_cast<int>(k) == 0) ? 1.0 : 0.0);
            }
        }
        for (std::size_t k = 1; k < c.a.size(); ++k) {
            if (n >= static_cast<int>(k)) {
                acc -= c.a[k] * y[static_cast<std::size_t>(n - static_cast<int>(k))];
            }
        }
        y[static_cast<std::size_t>(n)] = acc / c.a[0];
        if (n > steps / 2) {
            worst_tail = std::max(worst_tail, std::abs(y[static_cast<std::size_t>(n)]));
        }
    }
    return worst_tail;
}

} // namespace

TEST(SignalCheby2Orders, EveryOrderMeetsItsStopbandSpecification) {
    const double fs = 2.0;
    for (const double rs : {20.0, 40.0, 60.0}) {
        const double target = std::pow(10.0, -rs / 20.0);
        for (const double cutoff : {0.15, 0.25, 0.4}) {
            const double wn = cutoff * M_PI / (fs / 2.0);
            for (int order = 2; order <= 8; ++order) {
                const IirCoeffs lo = cheby2(order, rs, cutoff, fs, FilterType::Lowpass);
                ASSERT_EQ(lo.b.size(), static_cast<std::size_t>(order) + 1)
                    << "order " << order << " rs " << rs;
                ASSERT_EQ(lo.a.size(), static_cast<std::size_t>(order) + 1);

                // The passband reference gain, SIGNED. A gain normalisation that
                // multiplies by (-1)^order leaves |H| alone and flips this.
                const std::complex<double> dc = response(lo, 0.0);
                EXPECT_NEAR(dc.real(), 1.0, 1e-9) << "lowpass H(0) at order " << order;
                EXPECT_NEAR(dc.imag(), 0.0, 1e-12);

                // Type II's defining property: the stopband edge sits exactly on
                // the attenuation asked for. Not approximately, not "at least" --
                // the design is equiripple and this is where the ripple touches.
                EXPECT_NEAR(std::abs(response(lo, wn)), target, target * 1e-9)
                    << "lowpass stopband edge at order " << order << ", rs " << rs;

                // And it stays there or below across the rest of the stopband.
                for (int i = 0; i <= 40; ++i) {
                    const double w = wn + (M_PI - wn) * static_cast<double>(i) / 40.0;
                    EXPECT_LE(std::abs(response(lo, w)), target * (1.0 + 1e-6))
                        << "lowpass stopband at w=" << w << ", order " << order;
                }

                EXPECT_LT(impulse_tail(lo, 400), 1e-3)
                    << "lowpass order " << order << " is not stable";
            }
        }
    }
}

TEST(SignalCheby2Orders, HighpassAtEveryOrderToo) {
    // The highpass path goes through `lp2hp_zpk`, whose gain normalisation is the
    // one an even-order-only test cannot see.
    const double fs = 2.0;
    for (const double rs : {30.0, 50.0}) {
        const double target = std::pow(10.0, -rs / 20.0);
        for (const double cutoff : {0.2, 0.35}) {
            const double wn = cutoff * M_PI / (fs / 2.0);
            for (int order = 2; order <= 7; ++order) {
                const IirCoeffs hi = cheby2(order, rs, cutoff, fs, FilterType::Highpass);
                ASSERT_EQ(hi.b.size(), static_cast<std::size_t>(order) + 1);
                ASSERT_EQ(hi.a.size(), static_cast<std::size_t>(order) + 1);

                // Nyquist is the highpass passband reference, and +1 is signed.
                const std::complex<double> nyq = response(hi, M_PI);
                EXPECT_NEAR(nyq.real(), 1.0, 1e-9)
                    << "highpass H(pi) at order " << order << ", rs " << rs;
                EXPECT_NEAR(nyq.imag(), 0.0, 1e-12);

                EXPECT_NEAR(std::abs(response(hi, wn)), target, target * 1e-9)
                    << "highpass stopband edge at order " << order;

                for (int i = 0; i <= 40; ++i) {
                    const double w = wn * static_cast<double>(i) / 40.0;
                    EXPECT_LE(std::abs(response(hi, w)), target * (1.0 + 1e-6))
                        << "highpass stopband at w=" << w << ", order " << order;
                }

                EXPECT_LT(impulse_tail(hi, 400), 1e-3)
                    << "highpass order " << order << " is not stable";
            }
        }
    }
}

TEST(SignalCheby2Orders, OddAndEvenOrdersDifferInTheirZeroCount) {
    // Type II places its stopband zeros on the imaginary axis at 1/sin(m*pi/2n).
    // For EVEN order every one of them is finite. For ODD order the m = 0 zero is
    // at s = infinity and the two loops skip it, leaving order-1 finite zeros --
    // and the bilinear transform maps s = infinity to z = -1. So an odd-order
    // design has a transmission zero AT NYQUIST and an even-order one does not.
    //
    // That is the structural difference between the two branches of `cheb2ap`,
    // it is visible in the coefficients, and it is the thing an order-2-only
    // suite can never look at. (Measured before it was asserted: the first
    // version of this test had the two cases the wrong way round, which is what
    // running it tells you and reading it does not.)
    const double fs = 2.0;
    for (int order = 2; order <= 7; ++order) {
        const IirCoeffs lo = cheby2(order, 40.0, 0.25, fs, FilterType::Lowpass);
        const double h_nyquist = std::abs(response(lo, M_PI));
        if (order % 2 == 0) {
            EXPECT_GT(h_nyquist, 1e-9)
                << "even order " << order << " should have no zero at Nyquist, got "
                << h_nyquist;
        } else {
            EXPECT_LT(h_nyquist, 1e-12)
                << "odd order " << order << " should have its infinite zero at Nyquist, got "
                << h_nyquist;
        }
        // The b polynomial says the same thing directly: a zero at z = -1 means
        // the alternating sum of the numerator coefficients vanishes.
        double alternating = 0.0;
        double sign = 1.0;
        for (const double b : lo.b) {
            alternating += sign * b;
            sign = -sign;
        }
        if (order % 2 != 0) {
            EXPECT_NEAR(alternating, 0.0, 1e-14)
                << "odd order " << order << ": b(-1) is not zero";
        } else {
            EXPECT_GT(std::abs(alternating), 1e-12)
                << "even order " << order << ": b(-1) vanished";
        }
    }
}
