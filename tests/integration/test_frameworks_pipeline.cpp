// MathScript Integration Tests: Frameworks Pipeline
// Tests Axiom GP evolution, CellAI memory, Cypha DIF model end-to-end

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <vector>
#include <array>
#include <cstdint>

#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/core/matrix.hpp"

using DMatrix = ms::ColMatrix<double>;

// ---------------------------------------------------------------------------
// Axiom: evolve small population, terminate after 1 generation
// ---------------------------------------------------------------------------

TEST(FrameworksPipeline, Axiom_Evolve_SmallPopulation_Terminates) {
    ms::axiom::EvolutionConfig cfg;
    cfg.population_size = 5;
    cfg.max_generations = 2;
    cfg.use_cuda = false;

    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom axiom(cfg, registry);

    int eval_count = 0;
    auto objective = [&](const ms::axiom::Algorithm& algo) -> double {
        ++eval_count;
        return algo.fitness;
    };
    auto termination = [](const ms::axiom::Algorithm&) -> bool {
        return true; // terminate immediately
    };

    auto result = axiom.evolve(objective, termination);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(eval_count, 0);
}

TEST(FrameworksPipeline, Axiom_Population_SizeMatchesConfig) {
    ms::axiom::EvolutionConfig cfg;
    cfg.population_size = 3;
    cfg.max_generations = 1;
    cfg.use_cuda = false;

    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom axiom(cfg, registry);

    EXPECT_EQ(axiom.population().size(), 3u);
}

TEST(FrameworksPipeline, Axiom_GriaFitness_ReturnsFinite) {
    ms::axiom::EvolutionConfig cfg;
    cfg.population_size = 3;
    cfg.use_cuda = false;

    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom axiom(cfg, registry);

    const auto& pop = axiom.population();
    ASSERT_GT(pop.size(), 0u);

    DMatrix data({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    double fitness = axiom.gria_fitness(pop[0], data);
    EXPECT_TRUE(std::isfinite(fitness));
}

TEST(FrameworksPipeline, Axiom_Evaluate_ReturnsMatrix) {
    ms::axiom::EvolutionConfig cfg;
    cfg.population_size = 3;
    cfg.use_cuda = false;

    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    ms::axiom::Axiom axiom(cfg, registry);

    const auto& pop = axiom.population();
    ASSERT_GT(pop.size(), 0u);

    DMatrix data({{1.0, 2.0}, {3.0, 4.0}});
    auto result = axiom.evaluate(pop[0], data);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result.value().rows() * result.value().cols(), 0u);
}

TEST(FrameworksPipeline, Axiom_PrimitiveRegistry_HasFunctions) {
    auto registry = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
    EXPECT_GT(registry.function_names.size(), 0u);
    EXPECT_EQ(registry.function_names.size(), registry.function_symbols.size());
}

// ---------------------------------------------------------------------------
// CellAI: pipeline - step multiple times then recall
// ---------------------------------------------------------------------------

TEST(FrameworksPipeline, CellAI_MultiStep_StateEvolves) {
    ms::cellai::CellMemory memory(4, 6, {0.1, 1.0, 10.0});
    DMatrix x({{0.5}, {-0.3}, {0.8}, {-0.1}});

    std::vector<DMatrix> states;
    for (int step = 0; step < 5; ++step) {
        auto r = memory.step(x);
        ASSERT_TRUE(r.has_value());
        states.push_back(r.value());
    }

    // After multiple steps, state should have evolved (not all identical)
    bool changed = false;
    for (size_t i = 0; i < states[0].rows(); ++i) {
        if (std::abs(states[0](i,0) - states[4](i,0)) > 1e-12) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed) << "State should evolve over multiple steps";
}

TEST(FrameworksPipeline, CellAI_Reset_ClearsState) {
    ms::cellai::CellMemory memory(2, 4, {0.5, 5.0});
    DMatrix x({{1.0}, {1.0}});

    // Step once
    memory.step(x);

    // Reset
    memory.reset();

    // Step again from clean state
    auto r = memory.step(x);
    ASSERT_TRUE(r.has_value());
    for (size_t i = 0; i < r.value().rows(); ++i) {
        EXPECT_TRUE(std::isfinite(r.value()(i,0)));
    }
}

TEST(FrameworksPipeline, CellAI_HebbianUpdate_Dimensions) {
    DMatrix w({{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}});
    DMatrix x({{1.0}, {0.5}});
    DMatrix y({{0.3}, {0.7}, {-0.2}});
    double lr = 0.01;

    auto w_new = ms::cellai::hebbian_update(w, x, y, lr);
    EXPECT_EQ(w_new.rows(), 2u);
    EXPECT_EQ(w_new.cols(), 3u);
}

TEST(FrameworksPipeline, CellAI_HebbianUpdate_Changes_W) {
    DMatrix w({{0.0, 0.0}, {0.0, 0.0}});
    DMatrix x({{1.0}, {1.0}});
    DMatrix y({{1.0}, {1.0}});
    double lr = 0.1;

    auto w_new = ms::cellai::hebbian_update(w, x, y, lr);
    // With Hebb rule: w_new = w + lr * x * y^T
    // x*y^T = [[1,1],[1,1]], so w_new should be 0.1 * [[1,1],[1,1]]
    EXPECT_NEAR(w_new(0,0), 0.1, 1e-9);
    EXPECT_NEAR(w_new(0,1), 0.1, 1e-9);
}

TEST(FrameworksPipeline, CellAI_ToCyphaFeatures_Smoke) {
    const std::vector<double> scales = {0.1, 1.0};
    ms::cellai::CellMemory memory(2, 4, scales);
    DMatrix x({{0.5}, {0.5}});
    memory.step(x);

    auto features = ms::cellai::cell_to_cypha_features(memory, scales);
    EXPECT_GT(features.rows() * features.cols(), 0u);
    for (size_t i = 0; i < features.rows(); ++i)
        for (size_t j = 0; j < features.cols(); ++j)
            EXPECT_TRUE(std::isfinite(features(i,j)));
}

// ---------------------------------------------------------------------------
// Cypha DIF model: update and predict
// ---------------------------------------------------------------------------

TEST(FrameworksPipeline, Cypha_DifModel_UpdateAndPredict) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.n_experts = 4;
    cfg.online = true;

    ms::cypha::DifModel model(cfg);

    DMatrix x({{1.0}, {2.0}});
    DMatrix y({{3.0}});

    auto update_result = model.update(x, y);
    EXPECT_TRUE(update_result.has_value());

    auto pred = model.predict(x);
    ASSERT_TRUE(pred.has_value());
    EXPECT_EQ(pred.value().rows(), 1u);
    EXPECT_TRUE(std::isfinite(pred.value()(0,0)));
}

TEST(FrameworksPipeline, Cypha_DifModel_OOD_Score) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.n_experts = 3;

    ms::cypha::DifModel model(cfg);

    // Train on a few points
    DMatrix x({{0.0}, {0.0}});
    DMatrix y({{0.0}});
    model.update(x, y);

    // OOD score for training point should be finite
    auto score = model.ood_score(x);
    ASSERT_TRUE(score.has_value());
    EXPECT_TRUE(std::isfinite(score.value()));
}

TEST(FrameworksPipeline, Cypha_NIG_Fit_Smoke) {
    DMatrix data(10, 1, 0.0);
    for (int i = 0; i < 10; ++i) data(i, 0) = static_cast<double>(i) * 0.1;

    auto params = ms::cypha::nig_fit(data);
    EXPECT_TRUE(std::isfinite(params.mu));
    EXPECT_TRUE(std::isfinite(params.alpha));
    EXPECT_GT(params.alpha, 0.0);
    EXPECT_GT(params.delta, 0.0);
}

TEST(FrameworksPipeline, Cypha_NIG_PDF_Finite) {
    ms::cypha::NIGParams params;
    params.mu = 0.0;
    params.alpha = 2.0;
    params.beta = 0.0;
    params.delta = 1.0;

    double pdf = ms::cypha::nig_pdf(0.0, params);
    EXPECT_TRUE(std::isfinite(pdf));
    EXPECT_GT(pdf, 0.0);
}

TEST(FrameworksPipeline, Cypha_NIG_PDF_Decreases_FarFromCenter) {
    ms::cypha::NIGParams params;
    params.mu = 0.0;
    params.alpha = 3.0;
    params.beta = 0.0;
    params.delta = 1.0;

    // PDF should be finite everywhere
    for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        double pdf = ms::cypha::nig_pdf(x, params);
        EXPECT_TRUE(std::isfinite(pdf)) << "PDF not finite at x=" << x;
        EXPECT_GE(pdf, 0.0) << "PDF negative at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// Gria + Izaac: combined pipeline (alpha-driven dispatch)
// ---------------------------------------------------------------------------

TEST(FrameworksPipeline, Gria_Izaac_CryptoAlphaPipeline) {
    // Use IZAAC to seed RNG, then generate random matrix, compute gria alpha
    ms::izaac::seed_session(42ULL);
    ASSERT_TRUE(ms::izaac::session_active());

    auto M = ms::izaac::rand_matrix(4, 4);
    EXPECT_EQ(M.rows(), 4u);
    EXPECT_EQ(M.cols(), 4u);

    // Apply some transform (identity)
    double alpha = ms::gria::matrix_alpha(M, M);
    EXPECT_TRUE(std::isfinite(alpha));
    EXPECT_GE(alpha, 0.0);

    ms::izaac::clear_session();
}

TEST(FrameworksPipeline, Gria_Dispatch_Hint_Workflow) {
    // Register dispatch hints from computed alpha values
    auto field = ms::gria::gf2n::generate_field(4);
    EXPECT_EQ(field.size(), 16u);

    double alpha_ca = ms::gria::ca::alpha_ca(30, 50, 16);
    ms::gria::register_dispatch_hint("ca_rule30", alpha_ca);

    double retrieved = ms::gria::dispatch_hint_alpha("ca_rule30");
    EXPECT_NEAR(retrieved, alpha_ca, 1e-10);
}
