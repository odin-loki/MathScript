#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/cuda/nccl.hpp"

using namespace ms;
using namespace ms::interp;

TEST(ReplSessionTest, save_and_load_roundtrip) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_test_session.ms";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 2.5").has_value());
    ASSERT_TRUE(interp.execute("A = [1, 2; 3, 4]").has_value());
    ASSERT_TRUE(interp.save_session(path.string()).has_value());

    Interpreter loaded;
    ASSERT_TRUE(loaded.load_session(path.string()).has_value());
    EXPECT_DOUBLE_EQ(loaded.state().scalars.at("x"), 2.5);
    EXPECT_EQ(loaded.state().matrices.at("A").rows(), 2u);
    EXPECT_DOUBLE_EQ(loaded.state().matrices.at("A")(1, 1), 4.0);

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, history_save_load_roundtrip) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_history_session.ms";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 2.5").has_value());
    ASSERT_TRUE(interp.execute("y = x + 1").has_value());
    ASSERT_EQ(interp.state().history.size(), 2u);
    ASSERT_TRUE(interp.save_session(path.string()).has_value());

    Interpreter loaded;
    ASSERT_TRUE(loaded.load_session(path.string()).has_value());
    ASSERT_EQ(loaded.state().history.size(), 2u);
    EXPECT_EQ(loaded.state().history[0], "x = 2.5");
    EXPECT_EQ(loaded.state().history[1], "y = x + 1");

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, history_load_backward_compatible_without_history_lines) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_legacy_session.ms";

    {
        std::ofstream out(path);
        out << "# MathScript session\n";
        out << "scalar x = 3\n";
    }

    Interpreter loaded;
    ASSERT_TRUE(loaded.load_session(path.string()).has_value());
    EXPECT_DOUBLE_EQ(loaded.state().scalars.at("x"), 3.0);
    EXPECT_TRUE(loaded.state().history.empty());

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, export_history_writes_commands) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_history_export.txt";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("a = 1").has_value());
    ASSERT_TRUE(interp.execute("b = 2").has_value());
    ASSERT_TRUE(interp.export_history(path.string()).has_value());

    std::ifstream in(path);
    std::string line1;
    std::string line2;
    ASSERT_TRUE(static_cast<bool>(std::getline(in, line1)));
    ASSERT_TRUE(static_cast<bool>(std::getline(in, line2)));
    EXPECT_EQ(line1, "a = 1");
    EXPECT_EQ(line2, "b = 2");

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, export_history_meta_command) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_history_meta.txt";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("z = 9").has_value());
    const auto result = interp.execute("export history " + path.string());
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("exported history"), std::string::npos);

    std::ifstream in(path);
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(in, line)));
    EXPECT_EQ(line, "z = 9");

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, save_history_meta_command_alias) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_save_history.txt";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("k = 4").has_value());
    const auto result = interp.execute("save_history " + path.string());
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("exported history"), std::string::npos);

    std::ifstream in(path);
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(in, line)));
    EXPECT_EQ(line, "k = 4");

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, plot_command_sets_series) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("plot([1, 4, 2, 8])").has_value());
    EXPECT_TRUE(interp.has_plot());
    ASSERT_EQ(interp.plot().y.size(), 4u);
    EXPECT_DOUBLE_EQ(interp.plot().y[2], 2.0);
}

TEST(ReplSessionTest, plot_xy_command) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("plot([0, 1], [5, 10])").has_value());
    ASSERT_EQ(interp.plot().x.size(), 2u);
    EXPECT_DOUBLE_EQ(interp.plot().x[1], 1.0);
    EXPECT_DOUBLE_EQ(interp.plot().y[1], 10.0);
}

TEST(ReplSessionTest, hist_command_sets_bar_plot) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("hist([1, 2, 2, 10])").has_value());
    EXPECT_TRUE(interp.has_plot());
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Bar);
    EXPECT_EQ(interp.plot().y.size(), 10u);
}

TEST(ReplSessionTest, empty_matrix_save_load_roundtrip) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_empty_matrix.ms";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("E = []").has_value());
    ASSERT_TRUE(interp.save_session(path.string()).has_value());

    Interpreter loaded;
    ASSERT_TRUE(loaded.load_session(path.string()).has_value());
    EXPECT_EQ(loaded.state().matrices.at("E").rows(), 0u);
    EXPECT_EQ(loaded.state().matrices.at("E").cols(), 0u);

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, hist_via_named_matrix) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("D = [1, 3, 3, 9]").has_value());
    ASSERT_TRUE(interp.execute("hist(D)").has_value());
    EXPECT_TRUE(interp.has_plot());
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Bar);
}

TEST(ReplSessionTest, plot_save_load_roundtrip) {
    const auto path = std::filesystem::temp_directory_path() / "mathscript_plot_session.ms";

    Interpreter interp;
    ASSERT_TRUE(interp.execute("plot([1, 4, 2, 8])").has_value());
    EXPECT_TRUE(interp.has_plot());
    ASSERT_TRUE(interp.save_session(path.string()).has_value());

    Interpreter loaded;
    ASSERT_TRUE(loaded.load_session(path.string()).has_value());
    EXPECT_TRUE(loaded.has_plot());
    EXPECT_EQ(loaded.plot().y.size(), 4u);
    EXPECT_DOUBLE_EQ(loaded.plot().y[2], 2.0);

    std::filesystem::remove(path);
}

TEST(ReplSessionTest, unary_minus_scalar_expr) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = -sin(0)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 0.0);
    ASSERT_TRUE(interp.execute("y = 2 * -3").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), -6.0);
}

TEST(ReplEngineTest, reset_clears_state) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 5").has_value());
    ASSERT_TRUE(interp.execute("M = [1, 2; 3, 4]").has_value());
    ASSERT_TRUE(interp.execute("plot([1, 2, 3])").has_value());
    EXPECT_TRUE(interp.has_plot());

    interp.reset();

    EXPECT_TRUE(interp.state().scalars.empty());
    EXPECT_TRUE(interp.state().matrices.empty());
    EXPECT_FALSE(interp.has_plot());
}

TEST(ReplEngineTest, print_matrix_formats_output) {
    Matrix<double> M{{{1.0, 2.0}, {3.0, 4.0}}};
    std::ostringstream os;
    print_matrix(os, M);
    EXPECT_NE(os.str().find("1"), std::string::npos);
    EXPECT_NE(os.str().find("4"), std::string::npos);
}

TEST(ReplEngineTest, eval_scalar_call_erf) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("y = erf(0.5)").has_value());
    EXPECT_NEAR(interp.state().scalars.at("y"), 0.5205, 0.001);
}

TEST(ReplEngineTest, eval_scalar_call_unknown) {
    Interpreter interp;
    const auto result = interp.execute("z = nosuch(1.0)");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(interp.state().scalars.count("z"), 0u);
}

TEST(ReplEngineTest, divide_by_zero_scalar) {
    Interpreter interp;
    const auto result = interp.execute("w = 1 / 0");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(interp.state().scalars.count("w"), 0u);
}

TEST(ReplEngineTest, assign_overwrite) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 3").has_value());
    ASSERT_TRUE(interp.execute("x = 7").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 7.0);
}

TEST(ReplEngineTest, multiline_independence) {
    Interpreter first;
    Interpreter second;
    ASSERT_TRUE(first.execute("x = 99").has_value());
    EXPECT_DOUBLE_EQ(first.state().scalars.at("x"), 99.0);
    EXPECT_EQ(second.state().scalars.count("x"), 0u);
    ASSERT_TRUE(second.execute("x = 1").has_value());
    EXPECT_DOUBLE_EQ(second.state().scalars.at("x"), 1.0);
    EXPECT_DOUBLE_EQ(first.state().scalars.at("x"), 99.0);
}

namespace {

bool session_list_contains(const std::vector<std::pair<std::string, std::string>>& listed,
                           const std::string& handle,
                           const std::string& kind) {
    for (const auto& [entry_handle, entry_kind] : listed) {
        if (entry_handle == handle && entry_kind == kind) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(ReplSessionTest, list_session_objects_empty_initially) {
    Interpreter interp;
    EXPECT_TRUE(interp.list_session_objects().empty());
}

TEST(ReplSessionTest, list_session_objects_after_bloom_new) {
    ms::izaac::clear_session();
    Interpreter interp;
    ASSERT_TRUE(interp.execute("izaac seed 42").has_value());
    ASSERT_TRUE(interp.execute("bloom_new(bf, 100, 0.01)").has_value());
    const auto listed = interp.list_session_objects();
    ASSERT_EQ(listed.size(), 1u);
    EXPECT_EQ(listed[0].first, "bf");
    EXPECT_EQ(listed[0].second, "bloom");
}

TEST(ReplSessionTest, list_session_objects_multiple_kinds) {
    ms::izaac::clear_session();
    Interpreter interp;
    ASSERT_TRUE(interp.execute("izaac seed 42").has_value());
    ASSERT_TRUE(interp.execute("bloom_new(bf, 100, 0.01)").has_value());
    ASSERT_TRUE(interp.execute("tokenbucket_new(tb, 10, 1)").has_value());
    ASSERT_TRUE(interp.execute("cellmemory_new(cm, 2, 4, [0.1, 1, 10])").has_value());
    const auto listed = interp.list_session_objects();
    EXPECT_EQ(listed.size(), 3u);
    EXPECT_TRUE(session_list_contains(listed, "bf", "bloom"));
    EXPECT_TRUE(session_list_contains(listed, "tb", "tokenbucket"));
    EXPECT_TRUE(session_list_contains(listed, "cm", "cellmemory"));
}

TEST(ReplSessionTest, list_session_objects_matches_repl_command) {
    ms::izaac::clear_session();
    Interpreter interp;
    ASSERT_TRUE(interp.execute("izaac seed 42").has_value());
    ASSERT_TRUE(interp.execute("bloom_new(bf, 100, 0.01)").has_value());
    ASSERT_TRUE(interp.execute("tokenbucket_new(tb, 10, 1)").has_value());
    const auto listed = interp.list_session_objects();
    const auto repl_out = interp.execute("session_objects()");
    ASSERT_TRUE(repl_out.has_value());
    for (const auto& [handle, kind] : listed) {
        const std::string line = handle + " " + kind;
        EXPECT_NE(repl_out->find(line), std::string::npos) << line;
    }
}

TEST(ReplSessionTest, list_session_objects_after_clear) {
    ms::izaac::clear_session();
    Interpreter interp;
    ASSERT_TRUE(interp.execute("izaac seed 42").has_value());
    ASSERT_TRUE(interp.execute("tokenbucket_new(tb, 10, 1)").has_value());
    ASSERT_TRUE(interp.execute("session_object_clear(tb)").has_value());
    EXPECT_TRUE(interp.list_session_objects().empty());
}

TEST(ReplSessionTest, list_session_objects_reset_clears) {
    ms::izaac::clear_session();
    Interpreter interp;
    ASSERT_TRUE(interp.execute("izaac seed 42").has_value());
    ASSERT_TRUE(interp.execute("bloom_new(bf, 100, 0.01)").has_value());
    ASSERT_FALSE(interp.list_session_objects().empty());
    interp.reset();
    EXPECT_TRUE(interp.list_session_objects().empty());
}

TEST(ReplSessionTest, list_session_objects_empty_repl_message) {
    Interpreter interp;
    const auto repl_out = interp.execute("session_objects()");
    ASSERT_TRUE(repl_out.has_value());
    EXPECT_NE(repl_out->find("(no session objects)"), std::string::npos);
    EXPECT_TRUE(interp.list_session_objects().empty());
}
