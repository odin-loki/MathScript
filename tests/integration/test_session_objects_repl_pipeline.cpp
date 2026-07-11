// MathScript Integration Tests: REPL session-object registry pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

void expect_rng_seed_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_FALSE(result.has_value()) << cmd;
    const std::string message = ms::format_error(result.error());
    EXPECT_NE(message.find("izaac seed"), std::string::npos) << cmd << " error: " << message;
}

ms::Result<std::string> run(Interpreter& interp, const std::string& cmd) {
    return interp.execute(cmd);
}

double parse_scalar_output(const std::string& text) {
    return std::stod(Interpreter::trim(text));
}

std::string trim_output(const std::string& text) {
    return Interpreter::trim(text);
}

} // namespace

TEST(SessionObjectsReplPipeline, SessionObjectRegistryPipeline) {
    Interpreter interp;
    ms::izaac::clear_session();

    // Bloom filter: RNG required before creation.
    expect_rng_seed_error(interp, "bloom_new(bf, 100, 0.01)");
    expect_ok(interp, "izaac seed 42");

    const auto bloom_created = run(interp, "bloom_new(bf, 100, 0.01)");
    ASSERT_TRUE(bloom_created.has_value());
    EXPECT_NE(bloom_created->find("BloomFilter"), std::string::npos);

    expect_ok(interp, "bloom_insert(bf, \"alpha\")");
    expect_ok(interp, "bloom_insert(bf, \"beta\")");
    expect_ok(interp, "bloom_insert(bf, \"gamma\")");

    const auto check_alpha = run(interp, "bloom_check(bf, \"alpha\")");
    ASSERT_TRUE(check_alpha.has_value());
    EXPECT_EQ(trim_output(*check_alpha), "true");

    const auto check_beta = run(interp, "bloom_check(bf, \"beta\")");
    ASSERT_TRUE(check_beta.has_value());
    EXPECT_EQ(trim_output(*check_beta), "true");

    const auto check_gamma = run(interp, "bloom_check(bf, \"gamma\")");
    ASSERT_TRUE(check_gamma.has_value());
    EXPECT_EQ(trim_output(*check_gamma), "true");

    const auto check_unknown = run(interp, "bloom_check(bf, \"not_inserted_xyz\")");
    ASSERT_TRUE(check_unknown.has_value());
    // False negatives are impossible; false positives are allowed for bloom filters.

    // Token bucket lifecycle.
    const auto tb_created = run(interp, "tokenbucket_new(tb, 10, 1)");
    ASSERT_TRUE(tb_created.has_value());
    EXPECT_NE(tb_created->find("TokenBucket"), std::string::npos);

    const auto consume_ok = run(interp, "tokenbucket_consume(tb, 5, 0)");
    ASSERT_TRUE(consume_ok.has_value());
    EXPECT_EQ(trim_output(*consume_ok), "true");

    const auto consume_fail = run(interp, "tokenbucket_consume(tb, 10, 0)");
    ASSERT_TRUE(consume_fail.has_value());
    EXPECT_EQ(trim_output(*consume_fail), "false");

    const auto available = run(interp, "tokenbucket_available(tb, 0)");
    ASSERT_TRUE(available.has_value());
    const double tokens = parse_scalar_output(*available);
    EXPECT_GE(tokens, 0.0);
    EXPECT_LE(tokens, 10.0);

    const auto available_refill = run(interp, "tokenbucket_available(tb, 5)");
    ASSERT_TRUE(available_refill.has_value());
    EXPECT_GT(parse_scalar_output(*available_refill), tokens);

    // CellMemory lifecycle.
    const auto cm_created = run(interp, "cellmemory_new(cm, 2, 4, [0.1, 1, 10])");
    ASSERT_TRUE(cm_created.has_value());
    EXPECT_NE(cm_created->find("CellMemory"), std::string::npos);

    const auto step = run(interp, "cellmemory_step(cm, [1;0])");
    ASSERT_TRUE(step.has_value());
    EXPECT_NE(step->find("state ="), std::string::npos);
    EXPECT_NE(step->find("["), std::string::npos);

    const auto recall = run(interp, "cellmemory_recall(cm, 1.0)");
    ASSERT_TRUE(recall.has_value());
    EXPECT_NE(recall->find("recall ="), std::string::npos);

    const auto consolidated = run(interp, "cellmemory_consolidate(cm)");
    ASSERT_TRUE(consolidated.has_value());
    EXPECT_EQ(trim_output(*consolidated), "consolidated");

    // Introspection and clear.
    const auto listed = run(interp, "session_objects()");
    ASSERT_TRUE(listed.has_value());
    EXPECT_NE(listed->find("bf bloom"), std::string::npos);
    EXPECT_NE(listed->find("tb tokenbucket"), std::string::npos);
    EXPECT_NE(listed->find("cm cellmemory"), std::string::npos);

    const auto cleared = run(interp, "session_object_clear(tb)");
    ASSERT_TRUE(cleared.has_value());
    EXPECT_NE(cleared->find("tb"), std::string::npos);

    const auto listed_after = run(interp, "session_objects()");
    ASSERT_TRUE(listed_after.has_value());
    EXPECT_NE(listed_after->find("bf bloom"), std::string::npos);
    EXPECT_EQ(listed_after->find("tb tokenbucket"), std::string::npos);
    EXPECT_NE(listed_after->find("cm cellmemory"), std::string::npos);

    // Error cases: missing handle after clear.
    expect_error(interp, "tokenbucket_consume(tb, 1, 0)");
    const auto missing_tb = run(interp, "tokenbucket_consume(tb, 1, 0)");
    ASSERT_FALSE(missing_tb.has_value());
    EXPECT_NE(ms::format_error(missing_tb.error()).find("not found"), std::string::npos);

    // Wrong type for handle.
    expect_error(interp, "tokenbucket_consume(bf, 1, 0)");
    const auto wrong_type = run(interp, "tokenbucket_consume(bf, 1, 0)");
    ASSERT_FALSE(wrong_type.has_value());
    EXPECT_NE(ms::format_error(wrong_type.error()).find("not a TokenBucket"), std::string::npos);

    // Nonexistent handle.
    expect_error(interp, "bloom_insert(missing, \"x\")");
    const auto missing_bf = run(interp, "bloom_insert(missing, \"x\")");
    ASSERT_FALSE(missing_bf.has_value());
    EXPECT_NE(ms::format_error(missing_bf.error()).find("not found"), std::string::npos);

    // Clear nonexistent handle.
    expect_error(interp, "session_object_clear(ghost)");
}
