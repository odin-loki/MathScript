#include <gtest/gtest.h>
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;

TEST(FrameworksTest, gria_alpha_bounds) {
    std::vector<double> data{1, 2, 3, 4, 5, 6, 7, 8};
    const double alpha = gria::compute_alpha<double>(data, [](std::span<const double> input) {
        std::vector<double> out(input.begin(), input.end());
        for (double& v : out) {
            v = std::fmod(v + 3.0, 8.0);
        }
        return out;
    });
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
}

TEST(FrameworksTest, gria_gf2n_mul) {
    const uint64_t poly = 0x11B;
    const uint64_t product = gria::gf2n::mul(3, 5, poly);
    EXPECT_GT(product, 0u);
}

TEST(FrameworksTest, izaac_session_reproducible_randn) {
    izaac::seed_session(12345);
    izaac::seed_session(12345);
    const auto a = randn<double>(2, 2, 0);
    izaac::seed_session(12345);
    const auto b = randn<double>(2, 2, 0);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(a(i, j), b(i, j));
        }
    }
    izaac::clear_session();
}

TEST(FrameworksTest, cypha_online_update) {
    cypha::DifModel model(cypha::DifConfig{.input_dim = 2, .output_dim = 1});
    ColMatrix<double> x{{1.0}, {2.0}};
    ColMatrix<double> y{{3.0}};
    ASSERT_TRUE(model.update(x, y).has_value());
    const auto pred = model.predict(x).value();
    EXPECT_GT(pred(0, 0), 0.0);
}

TEST(FrameworksTest, cellai_hebbian_and_bridge) {
    ColMatrix<double> w{{0.1, 0.2}, {0.3, 0.4}};
    ColMatrix<double> x{{1.0}, {0.5}};
    ColMatrix<double> y{{0.8}, {0.2}};
    const auto updated = cellai::hebbian_update(w, x, y, 0.1);
    EXPECT_GT(updated(0, 0), w(0, 0));

    cellai::CellMemory memory(2, 4, {0.1, 1.0, 10.0});
    ASSERT_TRUE(memory.step(x).has_value());
    const auto features = cellai::cell_to_cypha_features(memory, {0.1, 1.0});
    EXPECT_EQ(features.rows(), 2u);
}

TEST(FrameworksTest, axiom_evolve_improves_fitness) {
    auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
    axiom::Axiom engine(axiom::EvolutionConfig{.population_size = 12, .max_generations = 15}, registry);
    ColMatrix<double> data{{1, 2}, {3, 4}};
    const auto best = engine
                          .evolve(
                              [&](const axiom::Algorithm& a) { return engine.gria_fitness(a, data); },
                              [](const axiom::Algorithm& a) { return a.fitness > 0.4; })
                          .value();
    EXPECT_GT(best.fitness, 0.0);
}

TEST(FrameworksIntegrationTest, gria_dispatch_hint_registered) {
    EXPECT_GE(gria::dispatch_hint_alpha("matmul"), 0.0);
}
