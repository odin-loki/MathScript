#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <sstream>
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
