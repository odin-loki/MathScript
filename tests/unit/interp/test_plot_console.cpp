#include <gtest/gtest.h>
#include "ms/core/matrix.hpp"
#include "ms/interp/plot_console.hpp"

using namespace ms;
using namespace ms::interp;

TEST(PlotConsoleTest, heatmap_preview_shows_range) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Heatmap;
    plot.grid = Matrix<double>{{{1.0, 2.0}, {3.0, 4.0}}};
    plot.matrix_rows = 2;
    plot.matrix_cols = 2;

    const auto text = format_plot_preview(plot);
    EXPECT_NE(text.find("range [1.0000, 4.0000]"), std::string::npos);
    EXPECT_NE(text.find("1.0000"), std::string::npos);
}

TEST(PlotConsoleTest, spy_preview_uses_pattern_chars) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Spy;
    plot.grid = Matrix<double>{{{1.0, 0.0}, {0.0, 2.0}}};
    plot.matrix_rows = 2;
    plot.matrix_cols = 2;
    plot.nnz = 2;

    const auto text = format_plot_preview(plot);
    EXPECT_NE(text.find("#."), std::string::npos);
    EXPECT_NE(text.find(".#"), std::string::npos);
}

TEST(PlotConsoleTest, surface_preview_labels_z_grid) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Surface3D;
    plot.grid = Matrix<double>{{{0.0, 1.0}, {2.0, 3.0}}};
    plot.matrix_rows = 2;
    plot.matrix_cols = 2;

    const auto text = format_plot_preview(plot);
    EXPECT_NE(text.find("Z (2x2)"), std::string::npos);
}

TEST(PlotConsoleTest, invalid_plot_message) {
    PlotSeries plot;
    EXPECT_EQ(format_plot_preview(plot), "(no plot)\n");
}

TEST(PlotConsoleTest, line_preview_not_empty) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Line;
    plot.x = {0.0, 1.0, 2.0};
    plot.y = {1.0, 3.0, 2.0};

    const auto text = format_plot_preview(plot);
    EXPECT_FALSE(text.empty());
    EXPECT_NE(text.find("plot"), std::string::npos);
}

TEST(PlotConsoleTest, scatter_preview_not_empty) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Scatter;
    plot.x = {0.0, 1.0, 2.0};
    plot.y = {1.0, 2.0, 3.0};

    const auto text = format_plot_preview(plot);
    EXPECT_FALSE(text.empty());
    EXPECT_NE(text.find("scatter"), std::string::npos);
}

TEST(PlotConsoleTest, bar_preview_not_empty) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Bar;
    plot.x = {1.0, 2.0, 3.0};
    plot.y = {5.0, 3.0, 7.0};

    const auto text = format_plot_preview(plot);
    EXPECT_FALSE(text.empty());
    EXPECT_NE(text.find("histogram"), std::string::npos);
}

TEST(PlotConsoleTest, invalid_series_no_data) {
    PlotSeries plot;
    EXPECT_FALSE(plot.valid);
    const auto text = format_plot_preview(plot);
    EXPECT_NE(text.find("(no plot)"), std::string::npos);
}

TEST(PlotConsoleTest, heatmap_empty_grid_shows_declared_dims) {
    PlotSeries plot;
    plot.valid = true;
    plot.kind = PlotSeries::Kind::Heatmap;
    plot.matrix_rows = 4;
    plot.matrix_cols = 5;

    const auto text = format_plot_preview(plot);
    EXPECT_NE(text.find("matrix plot (4x5)"), std::string::npos);
}

TEST(PlotConsoleTest, oversize_heatmap_and_scatter_truncate) {
    PlotSeries heat;
    heat.valid = true;
    heat.kind = PlotSeries::Kind::Heatmap;
    heat.grid = Matrix<double>(12, 2, 0.0);
    heat.grid(0, 0) = -1.0;
    heat.grid(11, 1) = 8.0;
    heat.matrix_rows = 12;
    heat.matrix_cols = 2;

    const auto heat_text = format_plot_preview(heat);
    EXPECT_NE(heat_text.find("data (12x2)"), std::string::npos);
    EXPECT_NE(heat_text.find("range [-1.0000, 8.0000]"), std::string::npos);
    EXPECT_NE(heat_text.find("  ...\n"), std::string::npos);

    PlotSeries scatter;
    scatter.valid = true;
    scatter.kind = PlotSeries::Kind::Scatter;
    scatter.x = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    scatter.y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    const auto scatter_text = format_plot_preview(scatter);
    EXPECT_NE(scatter_text.find("scatter (8 points)"), std::string::npos);
    EXPECT_NE(scatter_text.find("5 -> 6"), std::string::npos);
    EXPECT_EQ(scatter_text.find("7 -> 8"), std::string::npos);
    EXPECT_NE(scatter_text.find("  ...\n"), std::string::npos);
}
