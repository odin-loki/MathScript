// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Axiom: per-individual provenance of the evolutionary operators.
//
// The Algorithm::evaluation / selection / mutation fields used to be the three
// constants Sym("f(x)"), Sym("tournament") and Sym("mutate"), stamped onto every
// individual of every generation. They now record what actually produced each
// individual: the applied form evaluated for fitness, the selection operator
// with its parameter, and the variation chain. These tests pin that grammar and
// the point-mutation operator introduced alongside it.

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

namespace {

ms::axiom::EvolutionConfig cfg_of(
    size_t population,
    size_t generations,
    size_t tournament,
    double mutation_rate,
    double crossover_rate,
    size_t max_depth) {
    ms::axiom::EvolutionConfig cfg{};
    cfg.population_size = population;
    cfg.max_generations = generations;
    cfg.tournament_size = tournament;
    cfg.mutation_rate = mutation_rate;
    cfg.crossover_rate = crossover_rate;
    cfg.max_depth = max_depth;
    return cfg;
}

double flat_objective(const ms::axiom::Algorithm&) {
    return 1.0;
}

bool never_terminate(const ms::axiom::Algorithm&) {
    return false;
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

/// "<variation>[+subtree(r)|+point(r)][+regrow(d)]" with variation one of
/// none / clone(elite) / clone / crossover.
bool mutation_label_is_well_formed(const std::string& label) {
    std::string rest = label;
    const std::vector<std::string> variations = {"clone(elite)", "crossover", "clone", "none"};
    bool matched = false;
    for (const auto& variation : variations) {
        if (starts_with(rest, variation)) {
            rest.erase(0, variation.size());
            matched = true;
            break;
        }
    }
    if (!matched) {
        return false;
    }
    if (starts_with(rest, "+subtree(") || starts_with(rest, "+point(")) {
        const size_t close = rest.find(')');
        if (close == std::string::npos) {
            return false;
        }
        rest.erase(0, close + 1);
    }
    if (starts_with(rest, "+regrow(")) {
        const size_t close = rest.find(')');
        if (close == std::string::npos) {
            return false;
        }
        rest.erase(0, close + 1);
    }
    return rest.empty();
}

} // namespace

// ---------------------------------------------------------------------------
// Initial population: grown, not bred.
// ---------------------------------------------------------------------------

TEST(AxiomOperators, InitialPopulationRecordsGrowInitialisation) {
    ms::izaac::seed_session(static_cast<uint64_t>(31337));
    ms::axiom::Axiom engine(
        cfg_of(12, 0, 5, 0.1, 0.7, 6), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());

    std::set<std::string> evaluations;
    for (const auto& algo : engine.population()) {
        EXPECT_EQ(algo.selection.to_string(), "init(grow)");
        EXPECT_EQ(algo.mutation.to_string(), "none");
        const std::string evaluation = algo.evaluation.to_string();
        EXPECT_FALSE(evaluation.empty());
        EXPECT_TRUE(starts_with(evaluation, "f(")) << evaluation;
        EXPECT_TRUE(contains(evaluation, ")=")) << evaluation;
        evaluations.insert(evaluation);
    }
    EXPECT_GE(evaluations.size(), 3u);
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// evaluation == "f(<bound vars>)=<representation>".
// ---------------------------------------------------------------------------

TEST(AxiomOperators, EvaluationFormIsSignaturePlusRepresentation) {
    ms::izaac::seed_session(static_cast<uint64_t>(4711));
    ms::axiom::Axiom engine(
        cfg_of(10, 2, 4, 0.3, 0.6, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    for (const auto& algo : engine.population()) {
        const std::string evaluation = algo.evaluation.to_string();
        const std::string representation = algo.representation.to_string();
        const size_t split = evaluation.find(")=");
        ASSERT_NE(split, std::string::npos) << evaluation;
        EXPECT_TRUE(starts_with(evaluation, "f(")) << evaluation;
        EXPECT_EQ(evaluation.substr(split + 2), representation);

        const std::string signature = evaluation.substr(0, split);
        for (const std::string& var : {std::string("x0"), std::string("x1")}) {
            EXPECT_EQ(contains(representation, var), contains(signature, var))
                << "signature '" << signature << "' disagrees with '" << representation << "'";
        }
    }
    ms::izaac::clear_session();
}

TEST(AxiomOperators, ConstantOnlyIndividualsHaveEmptySignature) {
    ms::izaac::seed_session(static_cast<uint64_t>(6180));
    ms::axiom::Axiom engine(
        cfg_of(24, 0, 3, 0.0, 0.0, 4), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());

    for (const auto& algo : engine.population()) {
        const std::string evaluation = algo.evaluation.to_string();
        const std::string representation = algo.representation.to_string();
        const bool uses_variables = contains(representation, "x0") || contains(representation, "x1");
        if (!uses_variables) {
            EXPECT_TRUE(starts_with(evaluation, "f()=")) << evaluation;
        }
    }
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// Selection: elitism vs tournament, and the effective tournament size.
// ---------------------------------------------------------------------------

TEST(AxiomOperators, EliteCloneIsLabelledAndOthersAreNot) {
    ms::izaac::seed_session(static_cast<uint64_t>(2718));
    ms::axiom::Axiom engine(
        cfg_of(6, 3, 3, 0.0, 0.0, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    const auto& population = engine.population();
    ASSERT_EQ(population.size(), 6u);
    EXPECT_EQ(population[0].selection.to_string(), "elitism(1)");
    EXPECT_EQ(population[0].mutation.to_string(), "clone(elite)");
    for (size_t i = 1; i < population.size(); ++i) {
        EXPECT_EQ(population[i].selection.to_string(), "tournament(3)");
        EXPECT_EQ(population[i].mutation.to_string(), "clone");
    }
    ms::izaac::clear_session();
}

TEST(AxiomOperators, SelectionRecordsEffectiveTournamentSize) {
    ms::izaac::seed_session(static_cast<uint64_t>(555));
    {
        ms::axiom::Axiom engine(
            cfg_of(4, 1, 9, 0.0, 0.0, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
        const auto best = engine.evolve(flat_objective, never_terminate);
        ASSERT_TRUE(best.has_value());
        for (size_t i = 1; i < engine.population().size(); ++i) {
            EXPECT_EQ(engine.population()[i].selection.to_string(), "tournament(4)");
        }
    }
    {
        ms::axiom::Axiom engine(
            cfg_of(8, 1, 2, 0.0, 0.0, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
        const auto best = engine.evolve(flat_objective, never_terminate);
        ASSERT_TRUE(best.has_value());
        for (size_t i = 1; i < engine.population().size(); ++i) {
            EXPECT_EQ(engine.population()[i].selection.to_string(), "tournament(2)");
        }
    }
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// Variation chain: crossover, mutation operator and rate.
// ---------------------------------------------------------------------------

TEST(AxiomOperators, CrossoverChildrenAreLabelledCrossover) {
    ms::izaac::seed_session(static_cast<uint64_t>(90210));
    ms::axiom::Axiom engine(
        cfg_of(8, 2, 3, 0.0, 1.0, 4), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    const auto& population = engine.population();
    EXPECT_EQ(population[0].mutation.to_string(), "clone(elite)");
    for (size_t i = 1; i < population.size(); ++i) {
        const std::string label = population[i].mutation.to_string();
        EXPECT_TRUE(starts_with(label, "crossover")) << label;
        EXPECT_FALSE(contains(label, "subtree")) << label;
        EXPECT_FALSE(contains(label, "point")) << label;
    }
    ms::izaac::clear_session();
}

TEST(AxiomOperators, MutationLabelNamesOperatorAndRate) {
    ms::izaac::seed_session(static_cast<uint64_t>(13579));
    ms::axiom::Axiom engine(
        cfg_of(12, 2, 4, 1.0, 0.0, 6), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    const auto& population = engine.population();
    for (size_t i = 1; i < population.size(); ++i) {
        const std::string label = population[i].mutation.to_string();
        EXPECT_TRUE(starts_with(label, "clone")) << label;
        EXPECT_TRUE(contains(label, "+subtree(1)") || contains(label, "+point(1)")) << label;
    }
    ms::izaac::clear_session();
}

TEST(AxiomOperators, PointMutationIsActuallyUsed) {
    ms::izaac::seed_session(static_cast<uint64_t>(246810));
    ms::axiom::Axiom engine(
        cfg_of(40, 4, 3, 1.0, 0.0, 6), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    size_t point_labels = 0;
    size_t subtree_labels = 0;
    for (const auto& algo : engine.population()) {
        const std::string label = algo.mutation.to_string();
        point_labels += contains(label, "+point(") ? 1u : 0u;
        subtree_labels += contains(label, "+subtree(") ? 1u : 0u;
    }
    EXPECT_GT(point_labels, 0u);
    EXPECT_GT(subtree_labels, 0u);
    ms::izaac::clear_session();
}

TEST(AxiomOperators, MutationLabelsAreAlwaysWellFormed) {
    ms::izaac::seed_session(static_cast<uint64_t>(31415));
    ms::axiom::Axiom engine(
        cfg_of(20, 6, 4, 0.5, 0.8, 4), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    for (const auto& algo : engine.population()) {
        const std::string label = algo.mutation.to_string();
        EXPECT_TRUE(mutation_label_is_well_formed(label)) << label;
    }
    EXPECT_TRUE(mutation_label_is_well_formed(best->mutation.to_string()))
        << best->mutation.to_string();
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// The fields are per-individual, not constants.
// ---------------------------------------------------------------------------

TEST(AxiomOperators, FieldsDifferAcrossThePopulation) {
    ms::izaac::seed_session(static_cast<uint64_t>(24680));
    ms::axiom::Axiom engine(
        cfg_of(16, 4, 5, 0.1, 0.7, 6), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    std::set<std::string> selections;
    std::set<std::string> mutations;
    std::set<std::string> evaluations;
    for (const auto& algo : engine.population()) {
        EXPECT_FALSE(algo.selection.to_string().empty());
        EXPECT_FALSE(algo.mutation.to_string().empty());
        EXPECT_FALSE(algo.evaluation.to_string().empty());
        selections.insert(algo.selection.to_string());
        mutations.insert(algo.mutation.to_string());
        evaluations.insert(algo.evaluation.to_string());
    }
    EXPECT_GE(selections.size(), 2u);
    EXPECT_GE(mutations.size(), 2u);
    EXPECT_GE(evaluations.size(), 2u);
    ms::izaac::clear_session();
}

TEST(AxiomOperators, NoPlaceholderStringsRemain) {
    ms::izaac::seed_session(static_cast<uint64_t>(777));
    ms::axiom::Axiom engine(
        cfg_of(9, 2, 3, 0.5, 0.5, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    for (const auto& algo : engine.population()) {
        EXPECT_NE(algo.evaluation.to_string(), "f(x)");
        EXPECT_NE(algo.selection.to_string(), "tournament");
        EXPECT_NE(algo.mutation.to_string(), "mutate");
    }
    EXPECT_NE(best->evaluation.to_string(), "f(x)");
    EXPECT_NE(best->selection.to_string(), "tournament");
    EXPECT_NE(best->mutation.to_string(), "mutate");
    ms::izaac::clear_session();
}

// ---------------------------------------------------------------------------
// evolve()'s contract around the new fields.
// ---------------------------------------------------------------------------

TEST(AxiomOperators, EvolveOnTrivialObjectiveReturnsPopulatedBest) {
    ms::izaac::seed_session(static_cast<uint64_t>(101));
    ms::axiom::Axiom engine(
        cfg_of(5, 3, 2, 0.2, 0.5, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    EXPECT_FALSE(best->representation.to_string().empty());
    EXPECT_FALSE(best->evaluation.to_string().empty());
    EXPECT_FALSE(best->selection.to_string().empty());
    EXPECT_FALSE(best->mutation.to_string().empty());
    const std::set<std::string> allowed_selections = {
        "init(grow)", "elitism(1)", "tournament(2)"};
    EXPECT_EQ(allowed_selections.count(best->selection.to_string()), 1u)
        << best->selection.to_string();
    EXPECT_DOUBLE_EQ(best->fitness, 1.0);
    ms::izaac::clear_session();
}

TEST(AxiomOperators, ProvenanceFieldsAreLabelsNotArithmetic) {
    ms::izaac::seed_session(static_cast<uint64_t>(8642));
    ms::axiom::Axiom engine(
        cfg_of(6, 2, 3, 0.4, 0.6, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    // The three provenance Syms carry text, not an evaluable expression: the
    // grammar has no '=' and no "tournament"/"subtree" call, so eval() is the
    // documented malformed-expression 0.0 rather than a throw.
    for (const auto& algo : engine.population()) {
        EXPECT_DOUBLE_EQ(algo.evaluation.eval({{"x0", 1.0}, {"x1", 2.0}}), 0.0);
        EXPECT_DOUBLE_EQ(algo.selection.eval({}), 0.0);
        EXPECT_DOUBLE_EQ(algo.mutation.eval({}), 0.0);
    }
    ms::izaac::clear_session();
}

TEST(AxiomOperators, EmptyPopulationStillReportsDomainError) {
    ms::izaac::seed_session(static_cast<uint64_t>(4242));
    ms::axiom::Axiom engine(
        cfg_of(0, 3, 3, 0.1, 0.7, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_FALSE(best.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(best.error()));
    ms::izaac::clear_session();
}

TEST(AxiomOperators, SinglePopulationIsAllElite) {
    ms::izaac::seed_session(static_cast<uint64_t>(99));
    ms::axiom::Axiom engine(
        cfg_of(1, 3, 3, 0.9, 0.9, 5), ms::axiom::PrimitiveRegistry::build_from_ms_namespace());
    const auto best = engine.evolve(flat_objective, never_terminate);
    ASSERT_TRUE(best.has_value());

    ASSERT_EQ(engine.population().size(), 1u);
    EXPECT_EQ(engine.population()[0].selection.to_string(), "elitism(1)");
    EXPECT_EQ(engine.population()[0].mutation.to_string(), "clone(elite)");
    ms::izaac::clear_session();
}
