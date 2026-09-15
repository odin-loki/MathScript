// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: the underdamped case, which is the one control theory is about.
//
// `src/control/control.cpp` scored 79.2%, and the most telling survivor was
// `double im = std::sqrt(-disc) / (2 * a);` becoming `std::sqrt(disc)`. That
// line only runs when `disc < 0` -- the complex-conjugate branch of the
// quadratic root finder -- so the mutant takes the square root of a NEGATIVE
// number and every pole comes back **NaN**. Nothing noticed, because
// `ControlTF.Poles` uses `s^2 + 3s + 2`, whose discriminant is 1: the branch
// that produces complex poles had no test at all, and complex poles are what a
// second-order system has whenever it is underdamped.
//
// Two more survivors are in the step-response metrics, at boundaries a smooth
// response never reaches: the FALLING crossing test, and a settling search whose
// only out-of-tolerance sample is the first one.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "ms/control/control.hpp"

using namespace ms::control;

TEST(ControlComplexPoles, UnderdampedPolesAreTheConjugatePairTheyShouldBe) {
    // s^2 + 2*zeta*wn*s + wn^2 has poles at -zeta*wn +/- j*wn*sqrt(1 - zeta^2)
    // whenever zeta < 1. Both parts are closed forms, so both are asserted.
    for (const double wn : {0.5, 1.0, 4.0, 25.0}) {
        for (const double zeta : {0.0, 0.1, 0.5, 0.707, 0.99}) {
            const auto sys = tf({wn * wn}, {1.0, 2.0 * zeta * wn, wn * wn});
            const auto p = poles(sys);
            ASSERT_EQ(p.size(), 2u) << "wn " << wn << " zeta " << zeta;

            const double want_re = -zeta * wn;
            const double want_im = wn * std::sqrt(1.0 - zeta * zeta);

            for (const auto& root : p) {
                ASSERT_TRUE(std::isfinite(root.real()))
                    << "wn " << wn << " zeta " << zeta << ": real part is not finite";
                ASSERT_TRUE(std::isfinite(root.imag()))
                    << "wn " << wn << " zeta " << zeta << ": imaginary part is not finite";
                EXPECT_NEAR(root.real(), want_re, 1e-9 * std::max(1.0, wn));
                EXPECT_NEAR(std::abs(root.imag()), want_im, 1e-9 * std::max(1.0, wn));
            }
            // A conjugate pair: same real part, opposite imaginary parts.
            EXPECT_NEAR(p[0].real(), p[1].real(), 1e-12 * std::max(1.0, wn));
            EXPECT_NEAR(p[0].imag(), -p[1].imag(), 1e-12 * std::max(1.0, wn));

            // The two relations a control engineer reads off a pole: its distance
            // from the origin is the natural frequency, and the cosine of its
            // angle from the negative real axis is the damping ratio.
            const double magnitude = std::abs(p[0]);
            EXPECT_NEAR(magnitude, wn, 1e-9 * std::max(1.0, wn))
                << "|pole| is not the natural frequency";
            if (magnitude > 1e-12) {
                EXPECT_NEAR(-p[0].real() / magnitude, zeta, 1e-9)
                    << "the damping ratio read off the pole is wrong";
            }
        }
    }

    // The real-root branch still answers, so the above is not passing because
    // every quadratic now takes the complex path.
    const auto real_roots = poles(tf({1.0}, {1.0, 3.0, 2.0}));
    ASSERT_EQ(real_roots.size(), 2u);
    for (const auto& root : real_roots) {
        EXPECT_NEAR(root.imag(), 0.0, 1e-12) << "a real root grew an imaginary part";
    }
    std::vector<double> re{real_roots[0].real(), real_roots[1].real()};
    std::sort(re.begin(), re.end());
    EXPECT_NEAR(re[0], -2.0, 1e-9);
    EXPECT_NEAR(re[1], -1.0, 1e-9);

    // A repeated root sits exactly on the boundary between the two branches.
    const auto repeated = poles(tf({1.0}, {1.0, 4.0, 4.0}));
    ASSERT_EQ(repeated.size(), 2u);
    for (const auto& root : repeated) {
        EXPECT_NEAR(root.real(), -2.0, 1e-9);
        EXPECT_NEAR(root.imag(), 0.0, 1e-9);
    }
}

TEST(ControlComplexPoles, SettlingTimeWhenOnlyTheFirstSampleIsOutside) {
    // `last_outside` is the index of the final sample outside the tolerance band,
    // and -1 when there is none. Testing it with `<= 0` instead of `< 0` folds
    // "only sample zero was outside" into "nothing was outside" -- and a smooth
    // step response is always outside for many samples, so no ordinary trace can
    // tell the two apart. This trace leaves the band after one sample.
    StepData data;
    data.t = {0.0, 1.0, 2.0, 3.0, 4.0};
    data.y = {0.0, 1.0, 1.0, 1.0, 1.0};
    const StepInfo info = step_info(data, 2.0);
    EXPECT_NEAR(info.settling_time, 1.0, 1e-12)
        << "the first sample is outside the band, so settling is at the second";

    // And when nothing is outside at all, settling is the first instant.
    StepData already;
    already.t = {0.0, 1.0, 2.0};
    already.y = {1.0, 1.0, 1.0};
    EXPECT_NEAR(step_info(already, 2.0).settling_time, 0.0, 1e-12);

    // A trace still outside at the end never settles.
    StepData never;
    never.t = {0.0, 1.0, 2.0};
    never.y = {0.0, 0.5, 1.0};
    const StepInfo late = step_info(never, 2.0);
    EXPECT_TRUE(std::isinf(late.settling_time) || late.settling_time >= 2.0)
        << "settling_time " << late.settling_time;
}

TEST(ControlComplexPoles, RiseTimeOfAStepThatGoesDown) {
    // The crossing search has two arms, and only the rising one had ever run:
    // every step response in the suite ends above where it started. Replacing
    // the falling arm's `y[i-1] > threshold && y[i] <= threshold` with `||`
    // makes it fire on the first sample of any descending trace, and then the
    // "interpolation" extrapolates off the end of the segment it was handed.
    //
    // A trace from 0 down to -1, crossing -10% between samples 1 and 2 and
    // -90% between samples 2 and 3, with deliberately unequal slopes so that
    // extrapolating from the wrong segment cannot land on the right answer.
    const std::vector<double> t{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> y{0.0, -0.02, -0.30, -0.95, -1.0, -1.0};

    const StepInfo info = step_info(t, y, -1.0, 2.0);

    // 10% of a step of -1 is -0.1, reached at t = 1 + (0.08/0.28) = 9/7.
    // 90% is -0.9, reached at t = 2 + (0.60/0.65) = 38/13.
    const double t_lo = 1.0 + 0.08 / 0.28;
    const double t_hi = 2.0 + 0.60 / 0.65;
    ASSERT_TRUE(std::isfinite(info.rise_time)) << "a descending step has a rise time too";
    EXPECT_NEAR(info.rise_time, t_hi - t_lo, 1e-12)
        << "10%-90% of a falling step, interpolated within the crossing segments";

    // The header defines the peak as the maximum sample and overshoot as
    // 100*(peak - final)/|final| whenever the peak is above the final value.
    // For a descending step the maximum IS the starting value, so overshoot
    // reads 100% by that definition. Pinned here so that reading it as
    // "undershoot" would have to be a deliberate change, not a silent one.
    EXPECT_NEAR(info.peak_value, 0.0, 1e-12);
    EXPECT_NEAR(info.peak_time, 0.0, 1e-12);
    EXPECT_NEAR(info.overshoot_pct, 100.0, 1e-9);

    // The rising arm on the mirror image of the same trace, so this test is not
    // passing because both arms now behave alike.
    std::vector<double> up;
    up.reserve(y.size());
    for (const double v : y) up.push_back(-v);
    const StepInfo mirrored = step_info(t, up, 1.0, 2.0);
    ASSERT_TRUE(std::isfinite(mirrored.rise_time));
    EXPECT_NEAR(mirrored.rise_time, info.rise_time, 1e-12)
        << "rise time should not depend on the sign of the step";
}

TEST(ControlComplexPoles, ATraceOfOneSampleIsTheResponseAtZeroNotNaN) {
    // dt = t_end / (n_pts - 1) and w-step = i / (n_pts - 1): with one sample
    // both divide by zero, and every one-point trace came back with NaN in the
    // independent variable. Four entry points shared the expression.
    const auto sys = tf({2.0, 3.0}, {1.0, 1.0});  // proper, so D = 2 and y(0) = 2

    const StepData s = step_response(sys, 10.0, 1);
    ASSERT_EQ(s.t.size(), 1u);
    EXPECT_EQ(s.t[0], 0.0) << "the only sample of a trace is at t = 0";
    EXPECT_NEAR(s.y[0], 2.0, 1e-12) << "y(0) of a proper system is its feedthrough";

    const StepData i = impulse_response(sys, 10.0, 1);
    ASSERT_EQ(i.t.size(), 1u);
    EXPECT_EQ(i.t[0], 0.0);
    EXPECT_TRUE(std::isfinite(i.y[0])) << "y " << i.y[0];

    const BodeData b = bode(sys, 0.01, 1000.0, 1);
    ASSERT_EQ(b.w.size(), 1u);
    EXPECT_NEAR(b.w[0], 0.01, 1e-15) << "one point sits at the start of the range";
    EXPECT_TRUE(std::isfinite(b.magnitude[0])) << "magnitude " << b.magnitude[0];
    EXPECT_TRUE(std::isfinite(b.phase[0])) << "phase " << b.phase[0];

    const auto ny = nyquist(sys, 0.01, 1000.0, 1);
    ASSERT_EQ(ny.size(), 1u);
    EXPECT_TRUE(std::isfinite(ny[0].first) && std::isfinite(ny[0].second))
        << ny[0].first << " + j" << ny[0].second;

    // And the multi-point sweeps still span the range they are given.
    const BodeData full = bode(sys, 0.01, 1000.0, 5);
    ASSERT_EQ(full.w.size(), 5u);
    EXPECT_NEAR(full.w.front(), 0.01, 1e-12);
    EXPECT_NEAR(full.w.back(), 1000.0, 1e-9);
    const StepData five = step_response(sys, 10.0, 5);
    EXPECT_NEAR(five.t.back(), 10.0, 1e-12);
    EXPECT_NEAR(five.t[1], 2.5, 1e-12);

    // Zero samples is an empty trace, not a crash.
    EXPECT_TRUE(step_response(sys, 10.0, 0).t.empty());
    EXPECT_TRUE(impulse_response(sys, 10.0, 0).t.empty());
    EXPECT_TRUE(bode(sys, 0.01, 1000.0, 0).w.empty());
    EXPECT_TRUE(nyquist(sys, 0.01, 1000.0, 0).empty());
}

TEST(ControlComplexPoles, TheFallbackMarchStillWritesTheFirstSample) {
    // When the augmented matrix exponential cannot be formed, step_response
    // falls back to an explicit-Euler march. Nothing had ever reached that
    // branch, so starting its loop at i = 1 -- leaving sample zero at the
    // vector's default 0.0 -- went unnoticed. A horizon so long that A*dt
    // overflows is the reachable way in: the march that follows is nonsense,
    // but the sample at t = 0 is exact on either path and must be written.
    const auto sys = tf({2.0, 0.0, 0.0}, {1.0, 3.0, 2.0});  // D = 2, C = (-4, -6)
    const double huge = 1e308;

    const StepData s = step_response(sys, huge, 2);
    ASSERT_EQ(s.t.size(), 2u);
    EXPECT_EQ(s.t[0], 0.0);
    EXPECT_NEAR(s.y[0], 2.0, 1e-12)
        << "the first sample of a step response is D*u, whichever march produced it";

    const StepData i = impulse_response(sys, huge, 2);
    ASSERT_EQ(i.t.size(), 2u);
    EXPECT_EQ(i.t[0], 0.0);
    EXPECT_NEAR(i.y[0], -6.0, 1e-12)
        << "the first sample of an impulse response is C*B";

    // The same system over a horizon the exponential can handle: the fallback
    // is not what produced the numbers above by accident.
    const StepData ok = step_response(sys, 10.0, 2);
    EXPECT_NEAR(ok.y[0], 2.0, 1e-12);
    EXPECT_TRUE(std::isfinite(ok.y[1])) << "y " << ok.y[1];
}

TEST(ControlComplexPoles, TwoSampleTracesAndTheLastSampleOutsideTheBand) {
    // Two boundary arithmetic sites that a 500-point smooth trace cannot reach.
    const auto sys = tf({2.0, 0.0, 0.0}, {1.0, 3.0, 2.0});

    // n_pts == 2 is the smallest trace with an interval in it: dt is the whole
    // horizon. Guarding with `n_pts > 2` collapses dt to zero and stamps both
    // samples at t = 0; dividing by `n_pts - 2` makes dt infinite instead.
    for (const auto& trace : {step_response(sys, 10.0, 2), impulse_response(sys, 10.0, 2)}) {
        ASSERT_EQ(trace.t.size(), 2u);
        EXPECT_EQ(trace.t[0], 0.0);
        EXPECT_NEAR(trace.t[1], 10.0, 1e-12) << "the second of two samples is the horizon";
        EXPECT_TRUE(std::isfinite(trace.y[1])) << "y " << trace.y[1];
    }
    // The second sample is a real propagation, not a copy of the first.
    const StepData two = step_response(sys, 10.0, 2);
    EXPECT_GT(std::abs(two.y[0] - two.y[1]), 1e-6);

    // Settling: `last_outside + 1 >= y.size()` means "still outside at the end".
    // Written with + 2 it also swallows the case where the LAST sample is the
    // first one inside the band -- the response settles exactly at the horizon,
    // and the answer becomes +inf instead of that time.
    const std::vector<double> t{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> y{0.0, 0.3, 0.7, 1.0};
    const StepInfo info = step_info(t, y, 1.0, 2.0);
    EXPECT_NEAR(info.settling_time, 3.0, 1e-12)
        << "the trace enters the band on its final sample, so that is when it settles";

    // One sample later out of band and it never settles, which is the case the
    // `>=` is there for.
    const std::vector<double> late_y{0.0, 0.3, 0.7, 0.5};
    EXPECT_TRUE(std::isinf(step_info(t, late_y, 1.0, 2.0).settling_time));
}

TEST(ControlComplexPoles, GainMarginIsAPhaseCrossingThatHasToBeUnwrappedFirst) {
    // margin() looked for `phase < -180` in std::arg's principal value, which
    // lives in (-180, 180]. A phase descending through -180 is reported as +180
    // and then counts down, so the comparison was never true: gain_margin_db
    // came back +inf and phase_crossover_freq 0 for EVERY plant, including the
    // textbook ones. With the sweep unwrapped, both are real numbers.
    //
    // G = 1/(s+1)^3 crosses -180 at w = sqrt(3), where |G| = 1/8, so the gain
    // margin is 20*log10(8) = 18.0618 dB. The sweep is 5000 log-spaced points
    // over [1e-3, 1e5], about 0.18% apart, which is the tolerance below.
    {
        const auto sys = tf({1.0}, {1.0, 3.0, 3.0, 1.0});
        const Margins m = margin(sys);
        ASSERT_TRUE(std::isfinite(m.gain_margin_db)) << "gm " << m.gain_margin_db;
        EXPECT_NEAR(m.phase_crossover_freq, std::sqrt(3.0), 0.01);
        EXPECT_NEAR(m.gain_margin_db, 20.0 * std::log10(8.0), 0.05);
    }
    // The same plant with a gain of 10 is past the crossing: both margins go
    // negative, which is how margin() says the closed loop is unstable.
    {
        const auto sys = tf({10.0}, {1.0, 3.0, 3.0, 1.0});
        const Margins m = margin(sys);
        EXPECT_NEAR(m.phase_crossover_freq, std::sqrt(3.0), 0.01);
        EXPECT_NEAR(m.gain_margin_db, 20.0 * std::log10(0.8), 0.05);
        EXPECT_LT(m.gain_margin_db, 0.0);
        EXPECT_LT(m.phase_margin_deg, 0.0);
        EXPECT_FALSE(is_stable(feedback(sys, tf({1.0}, {1.0}), -1)));
    }
    // A phase that only approaches -180 is not a crossing: 1/(s^2+s+1) gets to
    // within 0.0006 degrees of it at the top of the sweep and still has an
    // infinite gain margin.
    {
        const Margins m = margin(tf({1.0}, {1.0, 1.0, 1.0}));
        EXPECT_TRUE(std::isinf(m.gain_margin_db)) << "gm " << m.gain_margin_db;
        EXPECT_EQ(m.phase_crossover_freq, 0.0);
    }
    // 1/s^2 sits ON the line: every sampled phase is exactly -180.0, which is
    // the one input that can tell `phase < -180` from `phase <= -180`. Sitting
    // on the line is not crossing it, so the margin stays infinite.
    {
        const Margins m = margin(tf({1.0}, {1.0, 0.0, 0.0}));
        EXPECT_TRUE(std::isinf(m.gain_margin_db)) << "gm " << m.gain_margin_db;
        EXPECT_EQ(m.phase_crossover_freq, 0.0)
            << "a phase that never leaves -180 never crosses it";
        EXPECT_NEAR(m.phase_margin_deg, 0.0, 1e-9);
    }
    // First order never reaches -180 at all.
    {
        const Margins m = margin(tf({1.0}, {1.0, 1.0}));
        EXPECT_TRUE(std::isinf(m.gain_margin_db));
        EXPECT_EQ(m.phase_crossover_freq, 0.0);
    }
}

TEST(ControlComplexPoles, OvershootIsNotAPercentageOfAFinalValueThatIsZero) {
    // Overshoot is 100*(peak - final)/|final|, which is meaningless once |final|
    // is at the floor, so step_info() only computes it when |final| > 1e-12.
    // Written as >= it would divide by exactly 1e-12 and report a percentage
    // made entirely of rounding.
    const std::vector<double> t{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> y{0.0, 5e-12, 2e-12, 1e-12};

    // Exactly at the floor: no overshoot is reported, however far the peak is
    // above the final value in relative terms.
    const StepInfo at_floor = step_info(t, y, 1e-12, 2.0);
    EXPECT_EQ(at_floor.overshoot_pct, 0.0)
        << "a final value at the floor has no percentage to be a fraction of";
    EXPECT_NEAR(at_floor.peak_value, 5e-12, 1e-24) << "the peak itself is still reported";

    // Ten times above it, the same shape does report an overshoot, so the guard
    // is a threshold and not an unconditional zero.
    const std::vector<double> big_y{0.0, 5e-11, 2e-11, 1e-11};
    const StepInfo above = step_info(t, big_y, 1e-11, 2.0);
    EXPECT_NEAR(above.overshoot_pct, 400.0, 1e-6);

    // And an ordinary trace is unaffected.
    const std::vector<double> ord{0.0, 1.2, 1.05, 1.0};
    EXPECT_NEAR(step_info(t, ord, 1.0, 2.0).overshoot_pct, 20.0, 1e-9);
}
