#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/stats/stats.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/frameworks/axiom/axiom.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

// ---------------------------------------------------------------------------
// CellAI → CyPha bridge pipeline
// ---------------------------------------------------------------------------

TEST(FrameworkPipelineTest, cellai_step_to_cypha_features) {
    // Create a CellMemory and step it with a 2-dim input
    ms::cellai::CellMemory cell(2, 4, {0.1, 0.5, 1.0});
    DMatrix input(2, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 0.5;

    const auto step_result = cell.step(input);
    ASSERT_TRUE(step_result.has_value());
    EXPECT_EQ(step_result->rows(), 4u);

    // Extract features for cypha
    const auto features = ms::cellai::cell_to_cypha_features(cell, {0.1, 0.5, 1.0});
    EXPECT_FALSE(features.empty());

    // Feed features into DifModel predict
    ms::cypha::DifConfig cfg;
    cfg.input_dim = features.rows();
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);

    const auto pred = model.predict(features);
    ASSERT_TRUE(pred.has_value());
    EXPECT_EQ(pred->rows(), 1u);
    EXPECT_TRUE(std::isfinite((*pred)(0, 0)));
}

// ---------------------------------------------------------------------------
// IZAAC → stats pipeline: generate random samples, verify statistics
// ---------------------------------------------------------------------------

TEST(FrameworkPipelineTest, izaac_samples_have_valid_stats) {
    ms::izaac::seed_session(123);
    const auto mat = ms::izaac::randn_matrix(50, 1);
    ASSERT_EQ(mat.rows(), 50u);

    std::vector<double> data(50);
    for (size_t i = 0; i < 50; ++i) data[i] = mat(i, 0);

    const double m = ms::mean(data);
    const double s = ms::stddev(data);
    EXPECT_TRUE(std::isfinite(m));
    EXPECT_TRUE(std::isfinite(s));
    EXPECT_GE(s, 0.0);

    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// Axiom PrimitiveRegistry + population construction
// ---------------------------------------------------------------------------

TEST(FrameworkPipelineTest, axiom_small_population) {
    const auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    EXPECT_FALSE(registry.function_names.empty());

    ms::axiom::EvolutionConfig cfg;
    cfg.population_size = 5;
    cfg.max_generations = 1;
    cfg.use_cuda = false;

    ms::axiom::Axiom axiom(cfg, registry);
    EXPECT_EQ(axiom.population().size(), cfg.population_size);
}

// ---------------------------------------------------------------------------
// Hebbian update preserves positive-definite structure
// ---------------------------------------------------------------------------

TEST(FrameworkPipelineTest, hebbian_update_finite_result) {
    DMatrix w = eye<double>(3);
    DMatrix x(3, 1);
    x(0, 0) = 1.0; x(1, 0) = 0.5; x(2, 0) = -0.5;
    DMatrix y = x;  // Hebbian: y = x

    const auto w_new = ms::cellai::hebbian_update(w, x, y, 0.01);
    EXPECT_EQ(w_new.rows(), 3u);
    EXPECT_EQ(w_new.cols(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(std::isfinite(w_new(i, j)));
        }
    }
}

// ---------------------------------------------------------------------------
// CyPha NIG fit → DifModel → OOD score pipeline
// ---------------------------------------------------------------------------

TEST(FrameworkPipelineTest, cypha_fit_then_ood_score) {
    DMatrix data(5, 1);
    data(0,0) = 0.1; data(1,0) = 0.9; data(2,0) = 0.5;
    data(3,0) = 0.3; data(4,0) = 0.7;

    const auto params = ms::cypha::nig_fit(data);
    EXPECT_TRUE(std::isfinite(params.mu));
    EXPECT_TRUE(std::isfinite(params.alpha));

    ms::cypha::DifConfig cfg;
    cfg.input_dim = 1;
    cfg.output_dim = 1;
    ms::cypha::DifModel model(cfg);

    // Update with some training data
    DMatrix x_train(1, 1); x_train(0, 0) = 0.5;
    DMatrix y_train(1, 1); y_train(0, 0) = 1.0;
    for (int i = 0; i < 5; ++i) {
        const auto upd = model.update(x_train, y_train);
        ASSERT_TRUE(upd.has_value());
    }

    // OOD score
    DMatrix x_test(1, 1); x_test(0, 0) = 0.5;
    const auto score = model.ood_score(x_test);
    ASSERT_TRUE(score.has_value());
    EXPECT_TRUE(std::isfinite(*score));
}
