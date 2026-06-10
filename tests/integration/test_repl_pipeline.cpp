#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "ms/core/matrix.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;
using ms::Matrix;

namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void expect_matrix_near(const Matrix<double>& a, const Matrix<double>& b, double tol) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            EXPECT_NEAR(a(i, j), b(i, j), tol);
        }
    }
}

} // namespace

TEST(ReplPipeline, full_linalg_plot_save) {
    Interpreter interp;

    ASSERT_TRUE(interp.execute("A = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(interp.execute("d = det(A)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("d"), 5.0);

    ASSERT_TRUE(interp.execute("U, S, V = svd(A)").has_value());
    EXPECT_TRUE(interp.state().matrices.count("U") > 0);
    EXPECT_TRUE(interp.state().matrices.count("S") > 0);
    EXPECT_TRUE(interp.state().matrices.count("V") > 0);

    ASSERT_TRUE(interp.execute("scatter([1, 2, 3], [1, 4, 9])").has_value());
    EXPECT_TRUE(interp.has_plot());
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Scatter);

    const auto plot_path =
        std::filesystem::temp_directory_path() / "mathscript_integration_plot.txt";
    const std::string save_cmd = "saveplot " + plot_path.string();
    ASSERT_TRUE(interp.execute(save_cmd).has_value());

    const std::string contents = read_file(plot_path);
    EXPECT_NE(contents.find("scatter"), std::string::npos);
    EXPECT_FALSE(contents.empty());

    std::filesystem::remove(plot_path);
}

TEST(ReplPipeline, session_roundtrip_with_matrices) {
    const auto path =
        std::filesystem::temp_directory_path() / "mathscript_integration_session.ms";

    Interpreter first;
    ASSERT_TRUE(first.execute("x = 3.14").has_value());
    ASSERT_TRUE(first.execute("M = [1, 2; 3, 4]").has_value());
    ASSERT_TRUE(first.save_session(path.string()).has_value());

    Interpreter second;
    ASSERT_TRUE(second.load_session(path.string()).has_value());
    EXPECT_DOUBLE_EQ(second.state().scalars.at("x"), 3.14);
    expect_matrix_near(second.state().matrices.at("M"), Matrix<double>{{1, 2}, {3, 4}}, 1e-12);

    std::filesystem::remove(path);
}

TEST(ReplPipeline, chained_scalar_assignments) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("a = 3.0").has_value());
    ASSERT_TRUE(interp.execute("b = a * 2.0").has_value());
    ASSERT_TRUE(interp.execute("c = sqrt(b)").has_value());
    EXPECT_NEAR(interp.state().scalars.at("c"), std::sqrt(6.0), 1e-12);
}
