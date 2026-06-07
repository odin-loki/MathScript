#include <gtest/gtest.h>
#include <filesystem>
#include "ms/interp/repl_engine.hpp"
#include "ms/cuda/nccl.hpp"

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
