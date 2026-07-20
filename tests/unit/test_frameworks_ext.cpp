#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;

TEST(FrameworksExtTest, gria_classify_buckets) {
    EXPECT_EQ(gria::classify(0.1), gria::ComputeClass::Reversible);
    EXPECT_EQ(gria::classify(0.5), gria::ComputeClass::Critical);
    EXPECT_EQ(gria::classify(0.9), gria::ComputeClass::Irreversible);
}

TEST(FrameworksExtTest, gria_entropy_and_matrix_alpha) {
    const std::vector<double> data{1, 1, 2, 2, 3, 3};
    EXPECT_GT(gria::entropy(data, 4), 0.0);
    EXPECT_DOUBLE_EQ(gria::entropy({}, 4), 0.0);

    ColMatrix<double> x{{1, 2}, {3, 4}};
    ColMatrix<double> fx = x;
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            fx(i, j) = std::sin(x(i, j));
        }
    }
    const double alpha = gria::matrix_alpha(x, fx);
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    EXPECT_TRUE(gria::is_critical(0.5, 0.05));
}

TEST(FrameworksExtTest, gria_ca_and_lfsr) {
    const std::vector<uint8_t> state{0, 1, 0, 1, 1};
    const auto next = gria::ca::step(state, 30);
    ASSERT_EQ(next.size(), state.size());
    EXPECT_GE(gria::ca::langton_lambda(30), 0.0);
    EXPECT_GE(gria::ca::alpha_ca(30, 8, 16), 0.0);

    const uint64_t poly = 0xB;
    const auto stepped = gria::lfsr::step(0x5, poly);
    EXPECT_NE(stepped, 0x5u);
    EXPECT_GE(gria::lfsr::alpha_lfsr(poly, 32), 0.0);
    EXPECT_FALSE(gria::lfsr::is_maximal(0, 4));
}

TEST(FrameworksExtTest, gria_gf2n_pow_inv_and_field) {
    const uint64_t poly = 0x11B;
    const uint64_t squared = gria::gf2n::pow(3, 2, poly);
    EXPECT_EQ(squared, gria::gf2n::mul(3, 3, poly));
    const uint64_t inv3 = gria::gf2n::inv(3, poly);
    EXPECT_NE(inv3, 0u);
    EXPECT_EQ(gria::gf2n::mul(3, inv3, poly), 1u);
    EXPECT_EQ(gria::gf2n::mul(0x53, 0xCA, poly), 0x01u);

    const auto field = gria::gf2n::generate_field(4);
    EXPECT_EQ(field.size(), 16u);
    EXPECT_EQ(gria::gf2n::generate_field(0).size(), 0u);
}

TEST(FrameworksExtTest, gria_dispatch_hint_override) {
    gria::register_dispatch_hint("custom_op", 0.42);
    EXPECT_NEAR(gria::dispatch_hint_alpha("custom_op"), 0.42, 1e-12);
    EXPECT_LT(gria::dispatch_hint_alpha("missing_op"), 0.0);
}

TEST(FrameworksExtTest, izaac_rand_matrix) {
    izaac::seed_session(99);
    const auto R = rand<double>(3, 2, 0);
    EXPECT_EQ(R.rows(), 3u);
    EXPECT_EQ(R.cols(), 2u);
    izaac::clear_session();
}

TEST(FrameworksExtTest, izaac_vrf_and_csprng) {
    const auto key = izaac::keygen();
    const std::vector<uint8_t> msg{1, 2, 3, 4};
    const auto proof = izaac::prove(key, msg);
    const auto reproof = izaac::prove(key, msg);
    EXPECT_EQ(std::memcmp(proof.output.data(), reproof.output.data(), proof.output.size()), 0);
    (void)izaac::verify(key.public_key, msg, proof);

    izaac::CSPRNG rng(proof);
    std::array<uint8_t, 16> buf{};
    rng.fill(buf);
    EXPECT_GT(rng.next_u64(), 0u);
    EXPECT_GE(rng.next_f64(), 0.0);
    EXPECT_LE(rng.next_f64(), 1.0);
    EXPECT_TRUE(std::isfinite(rng.next_normal()));
}

TEST(FrameworksExtTest, izaac_randn_and_mc_pi) {
    izaac::seed_session(42);
    const auto m = izaac::randn_matrix(2, 2);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_TRUE(izaac::session_active());
    std::array<uint8_t, 32> seed{};
    seed[0] = 42;
    izaac::CSPRNG rng(seed);
    const double pi = izaac::mc::estimate_pi(5000, rng);
    EXPECT_GT(pi, 0.0);
    EXPECT_LT(pi, 4.0);
    izaac::clear_session();
    EXPECT_FALSE(izaac::session_active());
}

TEST(FrameworksExtTest, cypha_nig_and_dimension_guards) {
    cypha::DifModel model(cypha::DifConfig{.input_dim = 2, .output_dim = 1});
    ColMatrix<double> bad_x{{1.0}, {2.0}, {3.0}};
    ColMatrix<double> good_x{{1.0}, {2.0}};
    ColMatrix<double> y{{3.0}};
    EXPECT_FALSE(model.update(bad_x, y).has_value());
    EXPECT_FALSE(model.predict(bad_x).has_value());
    EXPECT_FALSE(model.ood_score(bad_x).has_value());

    ASSERT_TRUE(model.update(good_x, y).has_value());
    const auto pred = model.predict(good_x).value();
    EXPECT_TRUE(std::isfinite(pred(0, 0)));
    const auto ood = model.ood_score(good_x).value();
    EXPECT_GT(ood, 0.0);

    const ColMatrix<double> empty;
    const cypha::NIGParams empty_fit = cypha::nig_fit(empty);
    EXPECT_DOUBLE_EQ(empty_fit.mu, 0.0);

    const ColMatrix<double> samples{{1.0}, {2.0}, {3.0}, {4.0}};
    const cypha::NIGParams fit = cypha::nig_fit(samples);
    EXPECT_GT(fit.delta, 0.0);
    EXPECT_GT(cypha::nig_pdf(2.5, fit), 0.0);
}

TEST(FrameworksExtTest, cypha_multi_output_and_mismatch_guards) {
    cypha::DifModel model(cypha::DifConfig{.input_dim = 2, .output_dim = 2, .learning_rate = 0.05});
    ColMatrix<double> x{{1.0}, {2.0}};
    ColMatrix<double> y{{3.0}, {4.0}};
    ColMatrix<double> bad_y{{3.0}};

    ASSERT_TRUE(model.update(x, y).has_value());
    EXPECT_FALSE(model.update(x, bad_y).has_value());

    const auto pred = model.predict(x).value();
    EXPECT_EQ(pred.rows(), 2u);
    EXPECT_TRUE(std::isfinite(pred(0, 0)));
    EXPECT_TRUE(std::isfinite(pred(1, 0)));

    ASSERT_TRUE(model.update(x, y).has_value());
    const auto pred2 = model.predict(x).value();
    EXPECT_NE(pred(0, 0), pred2(0, 0));

    ColMatrix<double> bad_x{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(model.predict(bad_x).has_value());
    EXPECT_FALSE(model.ood_score(bad_x).has_value());
}

TEST(FrameworksExtTest, axiom_evaluate_and_registry) {
    const auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
    ASSERT_FALSE(registry.function_names.empty());
    EXPECT_EQ(registry.function_names.size(), registry.function_symbols.size());
    EXPECT_NE(
        std::find(registry.function_names.begin(), registry.function_names.end(), "sin"),
        registry.function_names.end());

    axiom::Axiom engine(axiom::EvolutionConfig{.population_size = 4}, registry);
    ColMatrix<double> data{{1.0, 2.0}, {3.0, 4.0}};
    axiom::Algorithm algo = engine.population().front();
    algo.representation = ms::Sym("x0");
    const auto out = engine.evaluate(algo, data).value();
    EXPECT_EQ(out.rows(), data.rows());
    EXPECT_EQ(out.cols(), 1u);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(out(1, 0), 3.0);
}

TEST(FrameworksExtTest, axiom_gria_fitness_and_empty_population) {
    auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
    axiom::Axiom engine(
        axiom::EvolutionConfig{.population_size = 6, .target_alpha = 0.5},
        std::move(registry));
    ColMatrix<double> data{{0.1, 0.2}, {0.3, 0.4}};
    const double fitness = engine.gria_fitness(engine.population().front(), data);
    EXPECT_GE(fitness, 0.0);
    EXPECT_LE(fitness, 1.0);

    axiom::Axiom empty(
        axiom::EvolutionConfig{.population_size = 0},
        axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto result = empty.evolve(
        [&](const axiom::Algorithm& a) { return a.fitness; },
        [](const axiom::Algorithm&) { return false; });
    EXPECT_FALSE(result.has_value());
}

TEST(FrameworksExtTest, axiom_evolve_with_izaac_session) {
    izaac::seed_session(7);
    auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
    axiom::Axiom engine(axiom::EvolutionConfig{.population_size = 8, .max_generations = 5}, registry);
    ColMatrix<double> data{{1.0, 0.0}, {0.0, 1.0}};
    const auto best = engine
                          .evolve(
                              [&](const axiom::Algorithm& a) { return engine.gria_fitness(a, data); },
                              [](const axiom::Algorithm& a) { return a.fitness > 0.9; })
                          .value();
    EXPECT_GT(best.fitness, 0.0);
    izaac::clear_session();
}

TEST(FrameworksExtTest, cellai_memory_guards_and_bridge) {
    cellai::CellMemory memory(2, 3, {0.5, 2.0});
    ColMatrix<double> x{{1.0}, {0.5}};
    ColMatrix<double> bad_x{{1.0}};
    EXPECT_FALSE(memory.step(bad_x).has_value());

    ASSERT_TRUE(memory.step(x).has_value());
    const auto recalled = memory.recall(1.0).value();
    EXPECT_EQ(recalled.rows(), 3u);
    EXPECT_GT(recalled(0, 0), 0.0);

    memory.reset();
    const auto after_reset = memory.recall(0.0).value();
    EXPECT_DOUBLE_EQ(after_reset(0, 0), 0.0);

    const auto features = cellai::cell_to_cypha_features(memory, {0.1, 1.0, 5.0});
    EXPECT_EQ(features.rows(), 3u);

    ColMatrix<double> w{{0.0}};
    ColMatrix<double> small_x{{1.0}};
    ColMatrix<double> small_y{{0.5}};
    const auto updated = cellai::hebbian_update(w, small_x, small_y, 0.2);
    EXPECT_DOUBLE_EQ(updated(0, 0), 0.1);
}
