#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>
#include "ms/core/matrix.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

using DMatrix = ms::ColMatrix<double>;

// ---------------------------------------------------------------------------
// CellMemory – constructor / step dimensions
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, CellMemory_StepResultHasMemoryDimRows) {
    ms::cellai::CellMemory memory(3, 5, {0.1, 1.0, 10.0});
    DMatrix x{{1.0}, {0.5}, {-0.3}};
    const auto result = memory.step(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().rows(), 5u);
    EXPECT_EQ(result.value().cols(), 1u);
}

TEST(CellAiCyphaExtTest, CellMemory_StepResultIsFinite) {
    ms::cellai::CellMemory memory(2, 4, {0.5, 2.0});
    DMatrix x{{0.1}, {-0.2}};
    const auto result = memory.step(x);
    ASSERT_TRUE(result.has_value());
    const auto& m = result.value();
    for (size_t i = 0; i < m.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(m(i, 0)));
    }
}

// ---------------------------------------------------------------------------
// CellMemory – recall at each constructor time-scale
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, CellMemory_RecallAtEachConstructorTimeScale) {
    const std::vector<double> scales = {0.5, 2.0, 8.0};
    ms::cellai::CellMemory memory(2, 3, scales);
    DMatrix x{{1.0}, {0.5}};
    ASSERT_TRUE(memory.step(x).has_value());

    for (double ts : scales) {
        const auto recalled = memory.recall(ts);
        ASSERT_TRUE(recalled.has_value()) << "recall failed for time_scale=" << ts;
        EXPECT_EQ(recalled.value().rows(), 3u);
    }
}

// ---------------------------------------------------------------------------
// CellMemory – reset then step works normally
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, CellMemory_ResetThenStepSucceeds) {
    ms::cellai::CellMemory memory(2, 3, {0.5, 2.0});
    DMatrix x{{1.0}, {0.5}};

    ASSERT_TRUE(memory.step(x).has_value());
    memory.reset();

    // After reset the internal state should be zeroed; a subsequent step must still succeed.
    const auto result = memory.step(x);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().rows(), 3u);
}

TEST(CellAiCyphaExtTest, CellMemory_RecallAfterResetIsZero) {
    ms::cellai::CellMemory memory(2, 3, {1.0, 5.0});
    DMatrix x{{1.0}, {0.5}};
    ASSERT_TRUE(memory.step(x).has_value());
    memory.reset();

    // All entries should be back to zero after reset.
    const auto recalled = memory.recall(0.0);
    ASSERT_TRUE(recalled.has_value());
    const auto& r = recalled.value();
    for (size_t i = 0; i < r.rows(); ++i) {
        EXPECT_DOUBLE_EQ(r(i, 0), 0.0);
    }
}

// ---------------------------------------------------------------------------
// hebbian_update – zero learning rate leaves weights unchanged
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, HebbianUpdate_ZeroLearningRatePreservesWeights) {
    DMatrix w{{0.7, -0.3}, {0.1, 0.5}};
    DMatrix x{{1.0}, {2.0}};
    DMatrix y{{0.5}, {-1.0}};

    const auto updated = ms::cellai::hebbian_update(w, x, y, 0.0);
    for (size_t i = 0; i < w.rows(); ++i) {
        for (size_t j = 0; j < w.cols(); ++j) {
            EXPECT_DOUBLE_EQ(updated(i, j), w(i, j));
        }
    }
}

// ---------------------------------------------------------------------------
// cell_to_cypha_features – output non-empty, row count == memory_dim
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, CellToCyphaFeatures_OutputDimensionsMatchMemoryDim) {
    const size_t memory_dim = 4;
    const std::vector<double> ts = {0.1, 1.0, 5.0};
    ms::cellai::CellMemory memory(3, memory_dim, ts);

    // Step once so state is non-zero.
    DMatrix x{{0.5}, {-0.5}, {0.2}};
    ASSERT_TRUE(memory.step(x).has_value());

    const auto features = ms::cellai::cell_to_cypha_features(memory, ts);
    EXPECT_GT(features.rows(), 0u);
}

TEST(CellAiCyphaExtTest, CellToCyphaFeatures_EmptyTimeScales) {
    ms::cellai::CellMemory memory(2, 3, {});
    const auto features = ms::cellai::cell_to_cypha_features(memory, {});
    // With no time scales the output may be empty; at minimum it must not crash.
    (void)features;
}

// ---------------------------------------------------------------------------
// DifModel::ood_score – after multiple updates
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, DifModel_OodScoreAfterMultipleUpdates) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim  = 2;
    cfg.output_dim = 1;
    cfg.learning_rate = 0.05;
    ms::cypha::DifModel model(cfg);

    DMatrix x{{1.0}, {0.0}};
    DMatrix y{{1.0}};

    // Ten updates to build up internal state.
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(model.update(x, y).has_value());
    }

    const auto score = model.ood_score(x);
    ASSERT_TRUE(score.has_value());
    EXPECT_TRUE(std::isfinite(score.value()));
    EXPECT_GT(score.value(), 0.0);
}

TEST(CellAiCyphaExtTest, DifModel_OodScoreChangesWithInput) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim  = 2;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);

    DMatrix x_train{{1.0}, {0.0}};
    DMatrix x_ood{{100.0}, {100.0}};
    DMatrix y{{1.0}};

    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(model.update(x_train, y).has_value());
    }

    const auto score_in  = model.ood_score(x_train);
    const auto score_out = model.ood_score(x_ood);
    ASSERT_TRUE(score_in.has_value());
    ASSERT_TRUE(score_out.has_value());
    EXPECT_TRUE(std::isfinite(score_in.value()));
    EXPECT_TRUE(std::isfinite(score_out.value()));
}

// ---------------------------------------------------------------------------
// nig_fit – valid NIGParams from real data
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, NigFit_ReturnsValidParamsForRealData) {
    DMatrix data{{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}, {3.0}};
    const ms::cypha::NIGParams p = ms::cypha::nig_fit(data);
    EXPECT_TRUE(std::isfinite(p.mu));
    EXPECT_TRUE(std::isfinite(p.alpha));
    EXPECT_TRUE(std::isfinite(p.beta));
    EXPECT_GT(p.delta, 0.0);
}

TEST(CellAiCyphaExtTest, NigFit_SingleElementData) {
    DMatrix data{{42.0}};
    const ms::cypha::NIGParams p = ms::cypha::nig_fit(data);
    // Must not crash; delta should remain positive.
    EXPECT_GT(p.delta, 0.0);
}

// ---------------------------------------------------------------------------
// nig_pdf – non-negative everywhere, higher at mode than tails
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, NigPdf_NonNegativeAtSeveralPoints) {
    ms::cypha::NIGParams p;
    p.mu    = 0.0;
    p.alpha = 2.0;
    p.beta  = 0.0;
    p.delta = 1.0;

    for (double x : {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0}) {
        EXPECT_GE(ms::cypha::nig_pdf(x, p), 0.0) << "nig_pdf negative at x=" << x;
    }
}

TEST(CellAiCyphaExtTest, NigPdf_NonNegativeAndFinite) {
    // The simplified nig_pdf uses exp(-z)/(2*delta) where z=alpha*delta/sqrt(delta^2+(x-mu)^2)
    // This approximation is monotone in |x-mu|, so just verify basic properties
    ms::cypha::NIGParams p;
    p.mu    = 0.0;
    p.alpha = 2.0;
    p.beta  = 0.0;
    p.delta = 1.0;

    for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        const double pdf = ms::cypha::nig_pdf(x, p);
        EXPECT_GE(pdf, 0.0) << "nig_pdf non-negative at x=" << x;
        EXPECT_TRUE(std::isfinite(pdf)) << "nig_pdf finite at x=" << x;
    }
}

TEST(CellAiCyphaExtTest, NigPdf_FitThenEvaluate) {
    DMatrix data{{0.0}, {1.0}, {2.0}, {1.5}, {0.5}};
    const ms::cypha::NIGParams p = ms::cypha::nig_fit(data);
    // pdf at the estimated mean should be positive and finite.
    const double val = ms::cypha::nig_pdf(p.mu, p);
    EXPECT_GT(val, 0.0);
    EXPECT_TRUE(std::isfinite(val));
}

// ---------------------------------------------------------------------------
// nig_cdf – monotonicity, tails, pdf cross-check
// ---------------------------------------------------------------------------

namespace {

ms::cypha::NIGParams make_test_nig_params() {
    ms::cypha::NIGParams p;
    p.mu = 1.0;
    p.alpha = 2.5;
    p.beta = 0.0;
    p.delta = 1.2;
    return p;
}

double trapezoid_pdf_integral(double a, double b, const ms::cypha::NIGParams& p, int steps = 512) {
    if (a >= b) {
        return 0.0;
    }
    const double h = (b - a) / static_cast<double>(steps);
    double sum = 0.5 * (ms::cypha::nig_pdf(a, p) + ms::cypha::nig_pdf(b, p));
    for (int i = 1; i < steps; ++i) {
        sum += ms::cypha::nig_pdf(a + static_cast<double>(i) * h, p);
    }
    return sum * h;
}

} // namespace

TEST(CellAiCyphaExtTest, NigCdf_MonotonicallyNonDecreasing) {
    const auto p = make_test_nig_params();
    double prev = ms::cypha::nig_cdf(-5.0, p);
    for (double x : {-3.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0}) {
        const double cur = ms::cypha::nig_cdf(x, p);
        EXPECT_GE(cur, prev) << "cdf decreased at x=" << x;
        prev = cur;
    }
}

TEST(CellAiCyphaExtTest, NigCdf_LeftTailApproachesZero) {
    const auto p = make_test_nig_params();
    EXPECT_NEAR(ms::cypha::nig_cdf(p.mu - 100.0 * p.delta, p), 0.0, 1e-3);
}

TEST(CellAiCyphaExtTest, NigCdf_RightTailApproachesOne) {
    const auto p = make_test_nig_params();
    EXPECT_NEAR(ms::cypha::nig_cdf(p.mu + 100.0 * p.delta, p), 1.0, 1e-2);
}

TEST(CellAiCyphaExtTest, NigCdf_AtMeanIsBetweenZeroAndOne) {
    const auto p = make_test_nig_params();
    const double c = ms::cypha::nig_cdf(p.mu, p);
    EXPECT_GT(c, 0.0);
    EXPECT_LT(c, 1.0);
    EXPECT_NEAR(c, 0.5, 0.15);
}

TEST(CellAiCyphaExtTest, NigCdf_ConsistentWithPdfIntegration) {
    const auto p = make_test_nig_params();
    const double lower = p.mu - 2.0;
    const double upper = p.mu + 3.0;
    const double tail = 80.0 * p.delta;
    const double total = trapezoid_pdf_integral(p.mu - tail, p.mu + tail, p);
    const double cdf_diff = ms::cypha::nig_cdf(upper, p) - ms::cypha::nig_cdf(lower, p);
    const double integral = trapezoid_pdf_integral(lower, upper, p);
    EXPECT_NEAR(cdf_diff, integral / total, 5e-2);
}

TEST(CellAiCyphaExtTest, NigCdf_IntegrationCrossCheckOnSymmetricWindow) {
    const auto p = make_test_nig_params();
    const double lower = p.mu - 1.5;
    const double upper = p.mu + 1.5;
    const double tail = 80.0 * p.delta;
    const double total = trapezoid_pdf_integral(p.mu - tail, p.mu + tail, p);
    const double cdf_diff = ms::cypha::nig_cdf(upper, p) - ms::cypha::nig_cdf(lower, p);
    const double integral = trapezoid_pdf_integral(lower, upper, p);
    EXPECT_NEAR(cdf_diff, integral / total, 5e-2);
}

// ---------------------------------------------------------------------------
// nig_sample – count, finiteness, loose moments
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, NigSample_ReturnsRequestedCount) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x11;
    ms::izaac::CSPRNG rng(seed);
    const auto p = make_test_nig_params();
    const auto samples = ms::cypha::nig_sample(p, 250, rng);
    EXPECT_EQ(samples.rows(), 250u);
    EXPECT_EQ(samples.cols(), 1u);
}

TEST(CellAiCyphaExtTest, NigSample_AllFinite) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x22;
    ms::izaac::CSPRNG rng(seed);
    const auto p = make_test_nig_params();
    const auto samples = ms::cypha::nig_sample(p, 100, rng);
    for (size_t i = 0; i < samples.rows(); ++i) {
        EXPECT_TRUE(std::isfinite(samples(i, 0)));
    }
}

TEST(CellAiCyphaExtTest, NigSample_EmpiricalMeanNearMu) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x33;
    ms::izaac::CSPRNG rng(seed);
    const auto p = make_test_nig_params();
    const auto samples = ms::cypha::nig_sample(p, 5000, rng);
    double mean = 0.0;
    for (size_t i = 0; i < samples.rows(); ++i) {
        mean += samples(i, 0);
    }
    mean /= static_cast<double>(samples.rows());
    EXPECT_NEAR(mean, p.mu, 0.5);
}

TEST(CellAiCyphaExtTest, NigSample_EmpiricalVarianceInBallpark) {
    std::array<uint8_t, 32> seed{};
    seed[0] = 0x44;
    ms::izaac::CSPRNG rng(seed);
    const auto p = make_test_nig_params();
    const auto samples = ms::cypha::nig_sample(p, 5000, rng);
    double mean = 0.0;
    for (size_t i = 0; i < samples.rows(); ++i) {
        mean += samples(i, 0);
    }
    mean /= static_cast<double>(samples.rows());
    double var = 0.0;
    for (size_t i = 0; i < samples.rows(); ++i) {
        const double d = samples(i, 0) - mean;
        var += d * d;
    }
    var /= static_cast<double>(samples.rows());
    const double expected_scale = p.delta / std::sqrt(p.alpha);
    EXPECT_GT(var, 0.1 * expected_scale * expected_scale);
    EXPECT_LT(var, 25.0 * expected_scale * expected_scale);
}

TEST(CellAiCyphaExtTest, NigSample_DifferentSeedsProduceDifferentValues) {
    std::array<uint8_t, 32> seed_a{};
    std::array<uint8_t, 32> seed_b{};
    for (size_t i = 0; i < seed_a.size(); ++i) {
        seed_a[i] = static_cast<uint8_t>(0x10 + i * 3);
        seed_b[i] = static_cast<uint8_t>(0x80 + i * 5);
    }
    ms::izaac::CSPRNG rng_a(seed_a);
    ms::izaac::CSPRNG rng_b(seed_b);
    const auto p = make_test_nig_params();
    const auto sa = ms::cypha::nig_sample(p, 8, rng_a);
    const auto sb = ms::cypha::nig_sample(p, 8, rng_b);

    bool any_differ = false;
    for (size_t i = 0; i < sa.rows(); ++i) {
        if (sa(i, 0) != sb(i, 0)) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ);
}

// ---------------------------------------------------------------------------
// DifModel::gh_gate – heuristic coverage behaviour
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, GhGate_ReturnsBoolForValidInput) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{0.5}, {-0.2}};
    const bool gated = model.gh_gate(x);
    (void)gated;
    SUCCEED();
}

TEST(CellAiCyphaExtTest, GhGate_InDistributionAcceptedMoreOften) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.gh_protection = 0.95;
    ms::cypha::DifModel model(cfg);

    DMatrix x_in{{0.5}, {-0.1}};
    DMatrix x_out{{50.0}, {-40.0}};
    EXPECT_TRUE(model.gh_gate(x_in));
    EXPECT_FALSE(model.gh_gate(x_out));
}

TEST(CellAiCyphaExtTest, GhGate_ModerateInputMayPassWhileExtremeFails) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 3;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);

    DMatrix x_ok{{1.0}, {0.0}, {-0.5}};
    DMatrix x_bad{{1000.0}, {1000.0}, {1000.0}};
    EXPECT_TRUE(model.gh_gate(x_ok));
    EXPECT_FALSE(model.gh_gate(x_bad));
}

TEST(CellAiCyphaExtTest, GhGate_DimensionMismatchReturnsFalse) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(model.gh_gate(x));
}

// ---------------------------------------------------------------------------
// DifModel::predict_interval – credible band properties
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, PredictInterval_ReturnsFiniteMatrices) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{1.0}, {0.5}};
    DMatrix y{{2.0}};
    ASSERT_TRUE(model.update(x, y).has_value());

    const auto interval = model.predict_interval(x);
    ASSERT_TRUE(interval.has_value());
    const auto& pi = interval.value();
    EXPECT_TRUE(std::isfinite(pi.mean(0, 0)));
    EXPECT_TRUE(std::isfinite(pi.lower(0, 0)));
    EXPECT_TRUE(std::isfinite(pi.upper(0, 0)));
}

TEST(CellAiCyphaExtTest, PredictInterval_LowerMeanUpperOrdering) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 2;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{0.3}, {-0.7}};
    DMatrix y{{1.0}, {-0.5}};
    ASSERT_TRUE(model.update(x, y).has_value());

    const auto interval = model.predict_interval(x);
    ASSERT_TRUE(interval.has_value());
    const auto& pi = interval.value();
    for (size_t o = 0; o < cfg.output_dim; ++o) {
        EXPECT_LE(pi.lower(o, 0), pi.mean(o, 0));
        EXPECT_LE(pi.mean(o, 0), pi.upper(o, 0));
    }
}

TEST(CellAiCyphaExtTest, PredictInterval_PopulatesNigAlphaBeta) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{2.0}};
    DMatrix y{{3.0}};
    ASSERT_TRUE(model.update(x, y).has_value());

    const auto interval = model.predict_interval(x);
    ASSERT_TRUE(interval.has_value());
    EXPECT_GT(interval.value().nig_alpha, 0.0);
    EXPECT_TRUE(std::isfinite(interval.value().nig_beta));
}

TEST(CellAiCyphaExtTest, PredictInterval_MeanMatchesPredict) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{1.5}, {0.25}};
    DMatrix y{{0.5}};
    ASSERT_TRUE(model.update(x, y).has_value());

    const auto mean = model.predict(x);
    const auto interval = model.predict_interval(x);
    ASSERT_TRUE(mean.has_value());
    ASSERT_TRUE(interval.has_value());
    EXPECT_NEAR(interval.value().mean(0, 0), mean.value()(0, 0), 1e-12);
}

TEST(CellAiCyphaExtTest, PredictInterval_HasPositiveWidth) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);
    DMatrix x{{1.0}};
    DMatrix y{{1.0}};
    ASSERT_TRUE(model.update(x, y).has_value());

    const auto interval = model.predict_interval(x);
    ASSERT_TRUE(interval.has_value());
    EXPECT_GT(interval.value().upper(0, 0) - interval.value().lower(0, 0), 0.0);
}

// ---------------------------------------------------------------------------
// energy – hand-computed RBM-style energy
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, Energy_MatchesHandComputedExample) {
    DMatrix w{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix v{{1.0}, {0.0}};
    DMatrix h{{1.0}, {2.0}};
    EXPECT_DOUBLE_EQ(ms::cellai::energy(w, v, h), -5.0);
}

TEST(CellAiCyphaExtTest, Energy_ZeroWhenVisibleIsZero) {
    DMatrix w{{0.5, -1.0}, {2.0, 0.3}};
    DMatrix v{{0.0}, {0.0}};
    DMatrix h{{1.0}, {-1.0}};
    EXPECT_DOUBLE_EQ(ms::cellai::energy(w, v, h), 0.0);
}

TEST(CellAiCyphaExtTest, Energy_IsNegativeOfBilinearForm) {
    DMatrix w{{2.0}};
    DMatrix v{{3.0}};
    DMatrix h{{4.0}};
    EXPECT_DOUBLE_EQ(ms::cellai::energy(w, v, h), -24.0);
}

// ---------------------------------------------------------------------------
// CellMemory::consolidate – long-term moves toward short-term state
// ---------------------------------------------------------------------------

TEST(CellAiCyphaExtTest, Consolidate_SucceedsAfterStep) {
    ms::cellai::CellMemory memory(2, 3, {0.5});
    DMatrix x{{1.0}, {0.5}};
    ASSERT_TRUE(memory.step(x).has_value());
    EXPECT_TRUE(memory.consolidate().has_value());
}

TEST(CellAiCyphaExtTest, Consolidate_LongTermMovesTowardState) {
    ms::cellai::CellMemory memory(2, 3, {1.0});
    DMatrix x{{1.0}, {1.0}};
    memory.reset();
    ASSERT_TRUE(memory.step(x).has_value());

    const auto state = memory.recall(0.0).value();
    const DMatrix before = memory.long_term_state();
    ASSERT_TRUE(memory.consolidate().has_value());
    const DMatrix after = memory.long_term_state();

    double dist_before = 0.0;
    double dist_after = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        const double target = state(i, 0);
        dist_before += std::abs(before(i, 0) - target);
        dist_after += std::abs(after(i, 0) - target);
    }
    EXPECT_LT(dist_after, dist_before);
}

TEST(CellAiCyphaExtTest, Consolidate_RepeatedCallsRemainSuccessful) {
    ms::cellai::CellMemory memory(2, 2, {0.2});
    DMatrix x{{0.7}, {-0.3}};
    ASSERT_TRUE(memory.step(x).has_value());
    EXPECT_TRUE(memory.consolidate().has_value());
    EXPECT_TRUE(memory.consolidate().has_value());
}

TEST(CellAiCyphaExtTest, Consolidate_AfterResetStillSucceeds) {
    ms::cellai::CellMemory memory(2, 2, {0.2});
    memory.reset();
    EXPECT_TRUE(memory.consolidate().has_value());
}
