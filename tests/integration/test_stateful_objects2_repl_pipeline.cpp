// MathScript Integration Tests: REPL session-object registry (DifModel + Cluster)

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
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

ms::Result<std::string> run(Interpreter& interp, const std::string& cmd) {
    return interp.execute(cmd);
}

double parse_scalar_output(const std::string& text) {
    return std::stod(Interpreter::trim(text));
}

std::string trim_output(const std::string& text) {
    return Interpreter::trim(text);
}

bool output_has_finite_matrix_values(const std::string& text) {
    std::stringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        const auto open = line.find('[');
        const auto close = line.find(']');
        if (open == std::string::npos || close == std::string::npos || close <= open) {
            continue;
        }
        std::stringstream values(line.substr(open + 1, close - open - 1));
        std::string cell;
        while (std::getline(values, cell, ',')) {
            cell = Interpreter::trim(cell);
            if (cell.empty()) {
                continue;
            }
            const double value = std::stod(cell);
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST(StatefulObjects2ReplPipeline, DifModelCreate) {
    Interpreter interp;
    const auto created = run(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    ASSERT_TRUE(created.has_value());
    EXPECT_NE(created->find("DifModel"), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, DifModelUpdate) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    const auto updated = run(interp, "difmodel_update(dm, [1], [0.5])");
    ASSERT_TRUE(updated.has_value());
    EXPECT_EQ(trim_output(*updated), "ok");

    expect_ok(interp, "difmodel_update(dm, [0.5], [0.25])");
    expect_ok(interp, "difmodel_update(dm, [2], [1.0])");
}

TEST(StatefulObjects2ReplPipeline, DifModelPredict) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    expect_ok(interp, "difmodel_update(dm, [1], [0.5])");
    const auto predicted = run(interp, "difmodel_predict(dm, [1])");
    ASSERT_TRUE(predicted.has_value());
    EXPECT_NE(predicted->find("["), std::string::npos);
    EXPECT_TRUE(output_has_finite_matrix_values(*predicted));
}

TEST(StatefulObjects2ReplPipeline, DifModelPredictInterval) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    expect_ok(interp, "difmodel_update(dm, [1], [0.5])");
    const auto interval = run(interp, "difmodel_predict_interval(dm, [1])");
    ASSERT_TRUE(interval.has_value());
    EXPECT_NE(interval->find("mean ="), std::string::npos);
    EXPECT_NE(interval->find("lower ="), std::string::npos);
    EXPECT_NE(interval->find("upper ="), std::string::npos);
    EXPECT_NE(interval->find("nig_alpha ="), std::string::npos);
    EXPECT_NE(interval->find("nig_beta ="), std::string::npos);
    EXPECT_TRUE(output_has_finite_matrix_values(*interval));
}

TEST(StatefulObjects2ReplPipeline, DifModelOodScore) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    const auto score = run(interp, "difmodel_ood_score(dm, [1])");
    ASSERT_TRUE(score.has_value());
    const double value = parse_scalar_output(*score);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GE(value, 0.0);
}

TEST(StatefulObjects2ReplPipeline, DifModelGhGate) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    const auto gate = run(interp, "difmodel_gh_gate(dm, [1])");
    ASSERT_TRUE(gate.has_value());
    const std::string value = trim_output(*gate);
    EXPECT_TRUE(value == "true" || value == "false");
}

TEST(StatefulObjects2ReplPipeline, ClusterCreate) {
    Interpreter interp;
    const auto created = run(interp, "cluster_new(cl, 3, 42)");
    ASSERT_TRUE(created.has_value());
    EXPECT_NE(created->find("Cluster"), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, ClusterRunElection) {
    Interpreter interp;
    expect_ok(interp, "cluster_new(cl, 3, 42)");
    const auto leader = run(interp, "cluster_run_election(cl)");
    ASSERT_TRUE(leader.has_value());
    const int leader_id = static_cast<int>(parse_scalar_output(*leader));
    EXPECT_GE(leader_id, 0);
    EXPECT_LT(leader_id, 3);
}

TEST(StatefulObjects2ReplPipeline, ClusterReplicate) {
    Interpreter interp;
    expect_ok(interp, "cluster_new(cl, 3, 42)");
    const auto leader = run(interp, "cluster_run_election(cl)");
    ASSERT_TRUE(leader.has_value());
    const int leader_id = static_cast<int>(parse_scalar_output(*leader));
    ASSERT_GE(leader_id, 0);

    const auto replicated =
        run(interp, "cluster_replicate(cl, " + std::to_string(leader_id) + ", \"set x=1\")");
    ASSERT_TRUE(replicated.has_value());
    EXPECT_EQ(trim_output(*replicated), "true");
}

TEST(StatefulObjects2ReplPipeline, ClusterCurrentLeader) {
    Interpreter interp;
    expect_ok(interp, "cluster_new(cl, 3, 42)");
    expect_ok(interp, "cluster_run_election(cl)");
    const auto current = run(interp, "cluster_current_leader(cl)");
    ASSERT_TRUE(current.has_value());
    const int leader_id = static_cast<int>(parse_scalar_output(*current));
    EXPECT_GE(leader_id, 0);
    EXPECT_LT(leader_id, 3);
}

TEST(StatefulObjects2ReplPipeline, ClusterStatus) {
    Interpreter interp;
    expect_ok(interp, "cluster_new(cl, 5, 7)");
    expect_ok(interp, "cluster_run_election(cl)");
    const auto status = run(interp, "cluster_status(cl)");
    ASSERT_TRUE(status.has_value());
    EXPECT_NE(status->find("id=0 role="), std::string::npos);
    EXPECT_NE(status->find("id=4 role="), std::string::npos);
    EXPECT_NE(status->find("log_size="), std::string::npos);
    EXPECT_NE(status->find("commit_index="), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, SessionObjectsListsNewTypes) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    expect_ok(interp, "cluster_new(cl, 3, 42)");
    const auto listed = run(interp, "session_objects()");
    ASSERT_TRUE(listed.has_value());
    EXPECT_NE(listed->find("dm difmodel"), std::string::npos);
    EXPECT_NE(listed->find("cl cluster"), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, SessionObjectClearRemovesHandles) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    expect_ok(interp, "cluster_new(cl, 3, 42)");
    expect_ok(interp, "session_object_clear(dm)");
    const auto listed = run(interp, "session_objects()");
    ASSERT_TRUE(listed.has_value());
    EXPECT_EQ(listed->find("dm difmodel"), std::string::npos);
    EXPECT_NE(listed->find("cl cluster"), std::string::npos);

    expect_error(interp, "difmodel_predict(dm, [1])");
    const auto missing = run(interp, "difmodel_predict(dm, [1])");
    ASSERT_FALSE(missing.has_value());
    EXPECT_NE(ms::format_error(missing.error()).find("not found"), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, WrongTypeError) {
    Interpreter interp;
    expect_ok(interp, "difmodel_new(dm, 1, 1, 2, 0.1)");
    expect_error(interp, "cluster_run_election(dm)");
    const auto wrong_type = run(interp, "cluster_run_election(dm)");
    ASSERT_FALSE(wrong_type.has_value());
    EXPECT_NE(ms::format_error(wrong_type.error()).find("not a Cluster"), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, MissingHandleError) {
    Interpreter interp;
    expect_error(interp, "cluster_status(missing)");
    const auto missing = run(interp, "cluster_status(missing)");
    ASSERT_FALSE(missing.has_value());
    EXPECT_NE(ms::format_error(missing.error()).find("not found"), std::string::npos);
}

TEST(StatefulObjects2ReplPipeline, ReplicateNonLeaderFails) {
    Interpreter interp;
    expect_ok(interp, "cluster_new(cl, 3, 42)");
    const auto leader = run(interp, "cluster_run_election(cl)");
    ASSERT_TRUE(leader.has_value());
    const int leader_id = static_cast<int>(parse_scalar_output(*leader));
    ASSERT_GE(leader_id, 0);

    int follower_id = 0;
    if (follower_id == leader_id) {
        follower_id = 1;
    }
    const auto failed =
        run(interp, "cluster_replicate(cl, " + std::to_string(follower_id) + ", \"noop\")");
    ASSERT_TRUE(failed.has_value());
    EXPECT_EQ(trim_output(*failed), "false");
}
