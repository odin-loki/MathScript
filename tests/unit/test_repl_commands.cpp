#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>

#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

void expect_error_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_FALSE(result.has_value()) << cmd;
    const std::string message = ms::format_error(result.error());
    EXPECT_NE(message.find(needle), std::string::npos) << cmd << " error: " << message;
}

} // namespace

TEST(ReplCommandsTest, version_command) {
    Interpreter interp;
    expect_contains(interp, "version", std::string(ms::VERSION_STRING));
    expect_contains(interp, "help", "version");
}

TEST(ReplCommandsTest, meta_commands) {
    Interpreter interp;
    expect_contains(interp, "help", "Commands");
    expect_contains(interp, "topology", "threads=");
    expect_contains(interp, "simd", "ISA:");
    expect_contains(interp, "gpu", "cuda=");
    expect_contains(interp, "dispatch", "backend=");
    expect_contains(interp, "balance", "threads=");
    expect_contains(interp, "mpi", "backend=");
    expect_contains(interp, "frameworks", "GRIA");
}

TEST(ReplCommandsTest, gpu_command_device_stats) {
    Interpreter interp;
    const auto result = interp.execute("gpu");
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("cuda="), std::string::npos);
    if (ms::has_cuda()) {
        EXPECT_NE(result->find("mem_free="), std::string::npos) << *result;
        EXPECT_NE(result->find("mem_total="), std::string::npos) << *result;
    }
}

TEST(ReplCommandsTest, session_lifecycle) {
    Interpreter interp;
    expect_ok(interp, "x = 2.5");
    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_contains(interp, "vars", "x =");
    expect_contains(interp, "vars", "A (2x2)");
    expect_contains(interp, "clear", "session cleared");
    expect_contains(interp, "vars", "(empty session)");
}

TEST(ReplCommandsTest, framework_commands) {
    Interpreter interp;
    expect_contains(interp, "izaac seed 42", "seeded");
    expect_contains(interp, "axiom evolve", "fitness=");
    expect_contains(interp, "gria([1, 2, 3, 4])", "alpha=");
}

TEST(ReplCommandsTest, matrix_analytics) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("M = [3, 1; 1, 2]").has_value());
    expect_contains(interp, "det(M)", "\n");
    expect_contains(interp, "trace(M)", "\n");
    expect_contains(interp, "norm(M)", "\n");
    expect_contains(interp, "rank(M)", "\n");
    expect_contains(interp, "cond(M)", "\n");
}

TEST(ReplCommandsTest, decompositions_and_spectrum) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("M = [3, 1; 1, 2]").has_value());
    expect_contains(interp, "lu(M)", "L =");
    expect_contains(interp, "lu(M)", "U =");
    expect_contains(interp, "L, U = lu(M)", "L =");
    EXPECT_TRUE(interp.state().matrices.count("L") > 0);
    expect_contains(interp, "Q, R = qr(M)", "Q =");
    EXPECT_TRUE(interp.state().matrices.count("Q") > 0);
    expect_contains(interp, "qr(M)", "Q =");
    expect_contains(interp, "qr(M)", "R =");
    ASSERT_TRUE(interp.execute("S = [4, 1; 1, 3]").has_value());
    expect_contains(interp, "chol(S)", "L =");
    expect_contains(interp, "eig_sym(S)", "eigenvalues:");
    expect_contains(interp, "D, V = eig_sym(S)", "D =");
    EXPECT_TRUE(interp.state().matrices.count("V") > 0);
    expect_contains(interp, "svd(M)", "singular values:");
    expect_contains(interp, "U, Sig = svd(M)", "U =");
    EXPECT_TRUE(interp.state().matrices.count("Sig") > 0);
}

TEST(ReplCommandsTest, solve_and_matmul) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(interp.execute("B = [1; 1]").has_value());
    expect_contains(interp, "solve(A, B)", "x =");
    expect_contains(interp, "x = solve(A, B)", "x =");
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 0.2, 1e-9);
    expect_contains(interp, "matmul(A, A)", "C =");
    expect_contains(interp, "C = matmul(A, A)", "C =");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("C")(0, 0), 10.0);
    ASSERT_TRUE(interp.execute("T = transpose(A)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 1), 1.0);
}

TEST(ReplCommandsTest, fft_and_special_unary) {
    Interpreter interp;
    expect_contains(interp, "fft([1, 2, 3, 4])", "fft magnitudes:");
    expect_contains(interp, "erf(0.5)", "\n");
    expect_contains(interp, "erfc(0.5)", "\n");
    expect_contains(interp, "gamma(5)", "\n");
    expect_contains(interp, "bessel_j0(1)", "\n");
    expect_contains(interp, "fresnel_c(1)", "\n");
    expect_contains(interp, "fresnel_s(1)", "\n");
}

TEST(ReplCommandsTest, special_binary) {
    Interpreter interp;
    expect_contains(interp, "beta(0.5, 0.5)", "\n");
    expect_contains(interp, "legendre_p(2, 0.5)", "\n");
    expect_contains(interp, "spherical_jn(0, 1)", "\n");
    expect_contains(interp, "kelvin_ber(0, 1)", "\n");
    expect_contains(interp, "struve_h(0, 1)", "\n");
    expect_contains(interp, "bessel_zero_jnu(0, 1)", "\n");
    expect_contains(interp, "jacobi_sn(1, 0.5)", "\n");
    expect_contains(interp, "theta3(0, 0.2)", "\n");
    expect_contains(interp, "polylog(2, 0.5)", "\n");
}

TEST(ReplCommandsTest, special_multivariate_from_help) {
    Interpreter interp;
    expect_contains(interp, "kummer_m(1, 2, 0.5)", "\n");
    expect_contains(interp, "hypergeo_2f1(1, 1, 2, 0.5)", "\n");
    expect_contains(interp, "whittaker_m(0, 0.5, 1)", "\n");
    expect_contains(interp, "jacobi_p(2, 0.5, 0.5, 0.3)", "\n");
    expect_contains(interp, "mathieu_ce(1, 0.1, 0.5)", "\n");
    expect_contains(interp, "heun_g(0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0)", "\n");
    expect_contains(interp, "painleve1(0.5, 0.0, 0.0)", "\n");
}

TEST(ReplCommandsTest, special_unary_from_help) {
    Interpreter interp;
    expect_contains(interp, "ellip_k(0.5)", "\n");
    expect_contains(interp, "zeta(2)", "\n");
}

TEST(ReplCommandsTest, plot_commands) {
    Interpreter interp;
    expect_ok(interp, "plot([1, 4, 2, 8])");
    EXPECT_TRUE(interp.has_plot());
    expect_ok(interp, "plot([0, 1], [5, 10])");
    expect_ok(interp, "hist([1, 2, 2, 10])");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Bar);
}

TEST(ReplCommandsTest, scatter_imshow_spy_commands) {
    Interpreter interp;
    expect_contains(interp, "help", "scatter([x...], [y...])");
    expect_contains(interp, "help", "imshow(matrix)");
    expect_contains(interp, "help", "spy(matrix)");

    expect_contains(interp, "scatter([0, 1], [5, 10])", "scatter updated");
    EXPECT_TRUE(interp.has_plot());
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Scatter);
    ASSERT_EQ(interp.plot().x.size(), 2u);
    EXPECT_DOUBLE_EQ(interp.plot().y[1], 10.0);

    expect_contains(interp, "imshow([1, 2; 3, 4])", "imshow updated (2x2)");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Heatmap);
    EXPECT_EQ(interp.plot().matrix_rows, 2u);
    EXPECT_EQ(interp.plot().matrix_cols, 2u);

    expect_contains(interp, "spy([1, 0; 0, 2])", "spy updated (2 nonzeros)");
    expect_contains(interp, "spy([1, 0; 0, 2])", "spy pattern");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Spy);
    EXPECT_EQ(interp.plot().nnz, 2u);
    EXPECT_EQ(interp.plot().matrix_rows, 2u);
    EXPECT_EQ(interp.plot().matrix_cols, 2u);

    ASSERT_TRUE(interp.execute("X = [0, 1, 2]").has_value());
    ASSERT_TRUE(interp.execute("Y = [3, 4, 5]").has_value());
    expect_contains(interp, "scatter(X, Y)", "scatter updated");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Scatter);

    ASSERT_TRUE(interp.execute("M = [1, 2; 3, 4]").has_value());
    expect_contains(interp, "imshow(M)", "imshow updated (2x2)");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Heatmap);

    ASSERT_TRUE(interp.execute("S = [1, 0, 0; 0, 0, 3]").has_value());
    expect_contains(interp, "spy(S)", "spy updated (2 nonzeros)");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Spy);
    EXPECT_EQ(interp.plot().nnz, 2u);

    expect_contains(interp, "surf([1, 2; 3, 4])", "surf updated (2x2");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Surface3D);
    EXPECT_EQ(interp.plot().matrix_rows, 2u);
    EXPECT_EQ(interp.plot().matrix_cols, 2u);
    expect_contains(interp, "show", "Z (2x2)");

    const auto preview_path =
        (std::filesystem::temp_directory_path() / "mathscript_plot_preview.txt").string();
    expect_contains(interp, "saveplot " + preview_path, "saved plot preview");
    EXPECT_TRUE(std::filesystem::exists(preview_path));
    std::filesystem::remove(preview_path);

    ASSERT_TRUE(interp.execute("Z = [1, 2; 3, 4]").has_value());
    ASSERT_TRUE(interp.execute("GX = [0, 1; 2, 3]").has_value());
    ASSERT_TRUE(interp.execute("GY = [4, 5; 6, 7]").has_value());
    expect_contains(interp, "surf(GX, GY, Z)", "surf updated (2x2");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Surface3D);

    expect_error_contains(interp, "surf([])", "empty matrix");
    expect_error_contains(interp, "surf(Missing, Y, Z)", "unknown matrix");
    expect_ok(interp, "clear");
    expect_error_contains(interp, "show", "no plot");
    expect_error_contains(interp, "saveplot " + preview_path, "no plot");
    EXPECT_FALSE(interp.execute("surf([1, 2], [3, 4], [5, 6, 7])").has_value());

    expect_error_contains(interp, "imshow([])", "empty matrix");
    expect_error_contains(interp, "spy(Missing)", "unknown matrix");
    EXPECT_FALSE(interp.execute("scatter([1, 2], [3])").has_value());
}

TEST(ReplCommandsTest, save_load_inline) {
    const auto path = (std::filesystem::temp_directory_path() / "mathscript_repl_cmd.ms").string();
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 7").has_value());
    expect_contains(interp, "save " + path, "saved session");
    Interpreter loaded;
    expect_contains(loaded, "load " + path, "loaded session");
    EXPECT_DOUBLE_EQ(loaded.state().scalars.at("x"), 7.0);
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, run_file_executes_script_lines) {
    const auto path = (std::filesystem::temp_directory_path() / "mathscript_run_file.ms").string();
    {
        std::ofstream out(path);
        out << "# setup\n\n";
        out << "y = 3\n";
        out << "z = y + 1\n";
    }

    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 1").has_value());
    expect_contains(interp, "run_file " + path, "ran script from");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 3.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("z"), 4.0);
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, source_alias_runs_script) {
    const auto path = (std::filesystem::temp_directory_path() / "mathscript_source.ms").string();
    {
        std::ofstream out(path);
        out << "k = 11\n";
    }

    Interpreter interp;
    expect_contains(interp, "source " + path, "ran script from");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("k"), 11.0);
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, run_file_help_and_errors) {
    Interpreter interp;
    expect_contains(interp, "help", "run_file");
    expect_contains(interp, "help", "source");
    expect_contains(interp, "help", "unlike load");
    expect_error_contains(interp, "run_file ", "missing path");
    expect_error_contains(interp, "run_file /no/such/script.ms", "cannot open");
}

TEST(ReplCommandsTest, run_file_unlike_load_keeps_session) {
    const auto script_path = (std::filesystem::temp_directory_path() / "mathscript_run_not_load.ms").string();
    const auto session_path = (std::filesystem::temp_directory_path() / "mathscript_run_session.ms").string();
    {
        std::ofstream script(script_path);
        script << "y = 2\n";
    }

    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 5").has_value());
    expect_ok(interp, "run_file " + script_path);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 5.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 2.0);

    ASSERT_TRUE(interp.execute("save " + session_path).has_value());
    ASSERT_TRUE(interp.execute("x = 99").has_value());
    expect_ok(interp, "load " + session_path);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 5.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 2.0);
    EXPECT_EQ(interp.state().scalars.count("x"), 1u);

    std::filesystem::remove(script_path);
    std::filesystem::remove(session_path);
}

TEST(ReplCommandsTest, is_script_skip_line) {
    EXPECT_TRUE(Interpreter::is_script_skip_line(""));
    EXPECT_TRUE(Interpreter::is_script_skip_line("   "));
    EXPECT_TRUE(Interpreter::is_script_skip_line("# comment"));
    EXPECT_TRUE(Interpreter::is_script_skip_line("  # comment"));
    EXPECT_FALSE(Interpreter::is_script_skip_line("x = 1"));
}

TEST(ReplCommandsTest, parse_errors) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("nosuch(1)").has_value());
    EXPECT_FALSE(interp.execute("= 3").has_value());
    EXPECT_FALSE(interp.execute("save ").has_value());
}

TEST(ReplCommandsTest, gria_class_branches) {
    Interpreter interp;
    expect_contains(interp, "gria([1,1,1,1])", "reversible");
    expect_contains(interp, "gria([1,2,3,4,5,6,7,8,9])", "class=");
}

TEST(ReplCommandsTest, fft_row_vector_and_hist_constant) {
    Interpreter interp;
    expect_contains(interp, "fft([1, 2, 3])", "fft magnitudes:");
    expect_ok(interp, "hist([5, 5, 5, 5])");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Bar);
}

TEST(ReplCommandsTest, matrix_literal_and_invalid_calls) {
    Interpreter interp;
    expect_contains(interp, "det([1, 2; 3, 4])", "\n");
    expect_contains(interp, "trace([1, 2; 3, 4])", "\n");
    EXPECT_FALSE(interp.execute("beta(a,b)").has_value());
    EXPECT_FALSE(interp.execute("legendre_p(x,0.5)").has_value());
    EXPECT_FALSE(interp.execute("fft([1, 2; 3, 4])").has_value());
    EXPECT_FALSE(interp.execute("unknown([1])").has_value());
}

TEST(ReplCommandsTest, plot_binary_invalid_vectors) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("plot([1,2], [3])").has_value());
}

TEST(ReplCommandsTest, assignments_and_empty_line) {
    Interpreter interp;
    expect_ok(interp, "");
    expect_ok(interp, "s = 2.718");
    expect_ok(interp, "A = [1, 0; 0, 1]");
    expect_ok(interp, "x = 10");
    expect_ok(interp, "y = x / 2");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 5.0);
    expect_contains(interp, "vars", "s =");
    expect_contains(interp, "vars", "A (2x2)");
}

TEST(ReplCommandsTest, load_missing_file) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("load /no/such/mathscript.ms").has_value());
}

TEST(ReplCommandsTest, load_corrupt_session_file) {
    const auto path = (std::filesystem::temp_directory_path() / "mathscript_repl_bad.ms").string();
    {
        std::ofstream out(path);
        out << "scalar x = not_a_number\n";
    }
    Interpreter interp;
    EXPECT_FALSE(interp.execute("load " + path).has_value());
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, invalid_matrix_assignment) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("M = [1, 2; 3]").has_value());
}

TEST(ReplCommandsTest, singular_solve_and_matmul_mismatch) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("S = [1, 1; 1, 1]").has_value());
    ASSERT_TRUE(interp.execute("b = [1; 1]").has_value());
    EXPECT_FALSE(interp.execute("solve(S, b)").has_value());
    ASSERT_TRUE(interp.execute("A = [1, 2; 3, 4]").has_value());
    ASSERT_TRUE(interp.execute("B = [1; 1; 1]").has_value());
    EXPECT_FALSE(interp.execute("matmul(A, B)").has_value());
}

TEST(ReplCommandsTest, solve_and_matmul_via_variables) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(interp.execute("B = [1; 1]").has_value());
    expect_contains(interp, "solve(A, B)", "x =");
    expect_contains(interp, "matmul(A, A)", "C =");
}

TEST(ReplCommandsTest, gria_via_matrix_variable) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("G = [1, 2, 3, 4]").has_value());
    expect_contains(interp, "gria(G)", "alpha=");
}

TEST(ReplCommandsTest, plot_row_vectors_and_special_numeric_errors) {
    Interpreter interp;
    expect_ok(interp, "plot([1, 2, 3, 4])");
    EXPECT_TRUE(interp.has_plot());
    expect_ok(interp, "plot([0, 1, 2], [3, 4, 5])");
    EXPECT_FALSE(interp.execute("erf(x)").has_value());
    EXPECT_FALSE(interp.execute("gamma(bad)").has_value());
}

TEST(ReplCommandsTest, izaac_invalid_seed_and_hist_empty) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("izaac seed abc").has_value());
    EXPECT_FALSE(interp.execute("hist([])").has_value());
}

TEST(ReplCommandsTest, matrix_analytics_error_propagation) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("det([1, 2, 3])").has_value());
    EXPECT_FALSE(interp.execute("lu([1, 2; 3, 4; 5, 6])").has_value());
    EXPECT_FALSE(interp.execute("chol([1, 2; 3, 4])").has_value());
    ASSERT_TRUE(interp.execute("S = [1, 1; 1, 1]").has_value());
    EXPECT_FALSE(interp.execute("lu(S)").has_value());
    EXPECT_FALSE(interp.execute("det(S)").has_value());
}

TEST(ReplCommandsTest, eig_sym_on_nonsymmetric_matrix) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("M = [1, 2; 3, 4]").has_value());
    EXPECT_FALSE(interp.execute("eig_sym(M)").has_value());
}

TEST(ReplCommandsTest, save_to_invalid_path) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 1").has_value());
    EXPECT_FALSE(interp.execute("save ").has_value());
#ifdef _WIN32
    EXPECT_FALSE(interp.execute("save C:\\no\\such\\mathscript_dir\\session.ms").has_value());
#else
    EXPECT_FALSE(interp.execute("save /nonexistent_mathscript_xyz/session.ms").has_value());
#endif
}

TEST(ReplCommandsTest, gria_critical_class) {
    Interpreter interp;
    expect_contains(
        interp,
        "gria([1.87, 2.13, 8.21, 4.65, 3.05, 4.77, 18.92, 4.04])",
        "critical");
}

TEST(ReplCommandsTest, special_multivariate_parse_errors) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("heun_g(a,1,2,3,4,5,6)").has_value());
    EXPECT_FALSE(interp.execute("hypergeo_2f1(1,b,2,0.5)").has_value());
    EXPECT_FALSE(interp.execute("jacobi_p(1,a,b,0.3)").has_value());
    EXPECT_FALSE(interp.execute("kummer_m(1,b,0.5)").has_value());
    EXPECT_FALSE(interp.execute("whittaker_m(a,0.5,1)").has_value());
    EXPECT_FALSE(interp.execute("mathieu_ce(1,q,x)").has_value());
    EXPECT_FALSE(interp.execute("painleve1(x,0,0)").has_value());
}

TEST(ReplCommandsTest, unrecognized_command) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("not_a_command").has_value());
    EXPECT_FALSE(interp.execute("rank").has_value());
}

TEST(ReplCommandsTest, load_invalid_matrix_line) {
    const auto path = (std::filesystem::temp_directory_path() / "mathscript_repl_bad_matrix.ms").string();
    {
        std::ofstream out(path);
        out << "matrix M = [1, 2; 3]\n";
    }
    Interpreter interp;
    EXPECT_FALSE(interp.execute("load " + path).has_value());
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, matrix_literal_svd_and_lu_error) {
    Interpreter interp;
    expect_contains(interp, "svd([3, 1; 1, 2])", "singular values:");
    EXPECT_FALSE(interp.execute("lu([1, 2; 3, 4; 5, 6])").has_value());
}

TEST(ReplCommandsTest, plot_with_row_vector_x) {
    Interpreter interp;
    expect_ok(interp, "plot([0, 1, 2], [3, 4, 5])");
    ASSERT_EQ(interp.plot().x.size(), 3u);
    EXPECT_DOUBLE_EQ(interp.plot().x[2], 2.0);
}

TEST(ReplCommandsTest, hist_via_matrix_variable) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("D = [1, 2, 2, 10]").has_value());
    expect_contains(interp, "hist(D)", "histogram updated");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Bar);
    EXPECT_EQ(interp.plot().y.size(), 10u);
}

TEST(ReplCommandsTest, hist_max_value_clamps_to_last_bin) {
    Interpreter interp;
    expect_ok(interp, "hist([0, 5, 10])");
    EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Bar);
    EXPECT_GT(interp.plot().y.back(), 0.0);
}

TEST(ReplCommandsTest, hist_unknown_matrix_error) {
    Interpreter interp;
    expect_error_contains(interp, "hist(Missing)", "unknown matrix");
}

TEST(ReplCommandsTest, empty_and_malformed_matrix_literals) {
    Interpreter interp;
    expect_ok(interp, "E = []");
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("E").cols(), 0u);
    expect_error_contains(interp, "M = [;]", "no rows found");
    expect_error_contains(interp, "M = [1, bad]", "invalid number");
    expect_error_contains(interp, "M = 1, 2", "expected [ ... ]");
}

TEST(ReplCommandsTest, bare_save_load_parse_errors) {
    Interpreter interp;
    expect_error_contains(interp, "load", "could not parse: load");
    expect_error_contains(interp, "save", "could not parse: save");
}

TEST(ReplCommandsTest, unknown_matrix_resolve_errors) {
    Interpreter interp;
    expect_error_contains(interp, "det(Unknown)", "unknown matrix");
    expect_error_contains(interp, "plot(Unknown)", "unknown matrix");
}

TEST(ReplCommandsTest, plot_empty_data_error) {
    Interpreter interp;
    expect_error_contains(interp, "plot([])", "empty data");
}

TEST(ReplCommandsTest, plot_via_matrix_variables) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("X = [0, 1, 2]").has_value());
    ASSERT_TRUE(interp.execute("Y = [3, 4, 5]").has_value());
    expect_contains(interp, "plot(X, Y)", "plot updated");
    ASSERT_EQ(interp.plot().x.size(), 3u);
    EXPECT_DOUBLE_EQ(interp.plot().y[1], 4.0);
}

TEST(ReplCommandsTest, special_binary_nonnumeric_parse_errors) {
    Interpreter interp;
    expect_error_contains(interp, "spherical_jn(a, 1)", "expected spherical_jn(n,x)");
    expect_error_contains(interp, "kelvin_ber(x, 1)", "expected kelvin_ber(nu,x)");
    expect_error_contains(interp, "theta3(z, q)", "expected theta3(z,q)");
    expect_error_contains(interp, "polylog(n, z)", "expected polylog(n,z)");
}

TEST(ReplCommandsTest, zeros_eye_ones_assign) {
    Interpreter interp;
    expect_ok(interp, "Z = zeros(3)");
    ASSERT_TRUE(interp.state().matrices.count("Z") > 0);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("Z")(1, 1), 0.0);

    expect_ok(interp, "I = eye(4)");
    ASSERT_TRUE(interp.state().matrices.count("I") > 0);
    EXPECT_EQ(interp.state().matrices.at("I").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("I")(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("I")(0, 1), 0.0);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_TRUE(interp.state().matrices.count("O") > 0);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("O")(0, 0), 1.0);
}

TEST(ReplCommandsTest, expm_via_repl) {
    Interpreter interp;
    expect_ok(interp, "A = [0,1;0,0]");
    expect_ok(interp, "E = expm(A)");
    ASSERT_TRUE(interp.state().matrices.count("E") > 0);
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, sqrtm_logm_tril_triu_via_repl) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    ASSERT_TRUE(interp.state().matrices.count("S") > 0);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 1), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    ASSERT_TRUE(interp.state().matrices.count("L") > 0);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("L")(1, 1), 0.0, 1e-6);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    ASSERT_TRUE(interp.state().matrices.count("Lwr") > 0);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(1, 1), 4.0, 1e-9);

    expect_ok(interp, "Upr = triu(A)");
    ASSERT_TRUE(interp.state().matrices.count("Upr") > 0);
    EXPECT_NEAR(interp.state().matrices.at("Upr")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Upr")(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 1), 4.0, 1e-9);

    expect_ok(interp, "U1 = triu(A, 1)");
    ASSERT_TRUE(interp.state().matrices.count("U1") > 0);
    EXPECT_NEAR(interp.state().matrices.at("U1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("U1")(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("U1")(1, 1), 0.0, 1e-9);

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_TRUE(interp.state().matrices.count("Cs") > 0);
    ASSERT_TRUE(interp.state().matrices.count("Sn") > 0);
    EXPECT_NEAR(interp.state().matrices.at("Cs")(0, 0), 1.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("Cs")(1, 1), 0.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("Sn")(0, 0), 0.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("Sn")(1, 1), 1.0, 1e-5);

    expect_contains(interp, "help", "sqrtm(A)");
    expect_contains(interp, "help", "tril(A[, k])");
}

TEST(ReplCommandsTest, rand_randn_assign) {
    Interpreter interp;
    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_TRUE(interp.state().matrices.count("R") > 0);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }
    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_TRUE(interp.state().matrices.count("N") > 0);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_linspace_repmat) {
    Interpreter interp;

    // pinv(A) on small square matrix
    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_TRUE(interp.state().matrices.count("P") > 0);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("P").cols(), 2u);
    expect_ok(interp, "AP = matmul(A, P)");
    const auto& ap = interp.state().matrices.at("AP");
    EXPECT_NEAR(ap(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(ap(1, 1), 1.0, 1e-9);
    EXPECT_NEAR(ap(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(ap(1, 0), 0.0, 1e-9);

    // null(A) on rank-deficient 2x4 matrix
    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_TRUE(interp.state().matrices.count("N") > 0);
    const auto& n = interp.state().matrices.at("N");
    EXPECT_EQ(n.rows(), 4u);
    EXPECT_GE(n.cols(), 1u);
    expect_ok(interp, "WN = matmul(W, N)");
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    // orth(A) on full-rank 5x3 matrix
    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_TRUE(interp.state().matrices.count("Q") > 0);
    const auto& q = interp.state().matrices.at("Q");
    EXPECT_EQ(q.rows(), 5u);
    ASSERT_EQ(q.cols(), 3u);
    for (size_t i = 0; i < q.cols(); ++i) {
        for (size_t j = 0; j < q.cols(); ++j) {
            double dot = 0.0;
            for (size_t r = 0; r < q.rows(); ++r) {
                dot += q(r, i) * q(r, j);
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-8);
        }
    }

    // kron(eye(2), eye(2)) -> 4x4 identity
    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_TRUE(interp.state().matrices.count("K") > 0);
    const auto& k = interp.state().matrices.at("K");
    EXPECT_EQ(k.rows(), 4u);
    EXPECT_EQ(k.cols(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(k(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
        }
    }

    // linspace(0, 1, 5) -> 5x1 column
    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_TRUE(interp.state().matrices.count("V") > 0);
    const auto& v = interp.state().matrices.at("V");
    EXPECT_EQ(v.rows(), 5u);
    EXPECT_EQ(v.cols(), 1u);
    EXPECT_NEAR(v(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(v(4, 0), 1.0, 1e-12);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(v(i, 0), static_cast<double>(i) * 0.25, 1e-12);
    }

    // repmat([1,2;3,4], 2, 2) -> 4x4 tiled
    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_TRUE(interp.state().matrices.count("R") > 0);
    const auto& r = interp.state().matrices.at("R");
    EXPECT_EQ(r.rows(), 4u);
    EXPECT_EQ(r.cols(), 4u);
    const double tile[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(r(i, j), tile[i % 2][j % 2]);
        }
    }
}

TEST(ReplCommandsTest, wave60_rgb2gray_and_rle_roundtrip) {
    Interpreter interp;
    expect_contains(interp, "help", "rgb2gray(M)");
    expect_contains(interp, "help", "rle_encode_vec(M)");
    expect_contains(interp, "help", "bigint(\"495\")");

    // Pure red and pure green pixels in [0,1]
    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_TRUE(interp.state().matrices.count("G") > 0);
    const auto& gray = interp.state().matrices.at("G");
    EXPECT_EQ(gray.rows(), 2u);
    EXPECT_EQ(gray.cols(), 1u);
    EXPECT_NEAR(gray(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(gray(1, 0), 0.587, 1e-6);

    // RLE roundtrip on a small byte matrix
    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_TRUE(interp.state().matrices.count("R") > 0);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 8u);
    EXPECT_EQ(restored.cols(), 1u);
    const double expected[] = {1, 1, 2, 2, 2, 2, 3, 3};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(restored(i, 0), expected[i]);
    }

    expect_ok(interp, "x = bigint(\"495\")");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 495.0);
    expect_error_contains(interp, "y = bigint(\"9007199254740993\")", "too large");
}

TEST(ReplCommandsTest, wave60_image_filters_and_resize) {
    Interpreter interp;
    expect_contains(interp, "help", "imgaussfilt(M,s)");
    expect_contains(interp, "help", "laplacian(M)");
    expect_contains(interp, "help", "imresize(M,r,c)");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    expect_ok(interp, "L = laplacian(G)");
    expect_ok(interp, "H = histeq(G)");
    expect_ok(interp, "S = sharpen(G)");
    expect_ok(interp, "T = threshold_otsu(G)");
    expect_ok(interp, "R = imresize(G, 2, 2)");

    ASSERT_TRUE(interp.state().matrices.count("B") > 0);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 2u);
}

TEST(ReplCommandsTest, wave60_delta_encode_roundtrip) {
    Interpreter interp;
    expect_contains(interp, "help", "delta_encode_vec(M)");

    expect_ok(interp, "B = [10, 12, 15, 20]");
    expect_ok(interp, "E = delta_encode_vec(B)");
    expect_ok(interp, "R = delta_decode_vec(E)");
    ASSERT_TRUE(interp.state().matrices.count("R") > 0);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 4u);
    const double expected[] = {10, 12, 15, 20};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(restored(i, 0), expected[i]);
    }
}

TEST(ReplCommandsTest, wave60_ml_metrics) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_accuracy(p,t)");
    expect_contains(interp, "help", "ml_rmse(p,t)");

    expect_ok(interp, "yp = [1; 0; 1; 1]");
    expect_ok(interp, "yt = [1; 0; 0; 1]");
    expect_ok(interp, "acc = ml_accuracy(yp, yt)");
    expect_ok(interp, "rmse = ml_rmse([1; 2; 3], [1; 2; 4])");
    expect_ok(interp, "mse = ml_mse([1; 2; 3], [1; 2; 4])");
    expect_ok(interp, "r2 = ml_r2([1; 2; 3], [1; 2; 4])");
    expect_ok(interp, "f1 = ml_f1([1; 0; 1], [1; 0; 0])");

    EXPECT_DOUBLE_EQ(interp.state().scalars.at("acc"), 0.75);
    EXPECT_NEAR(interp.state().scalars.at("rmse"), 1.0 / std::sqrt(3.0), 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("mse"), 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("f1"), 2.0 / 3.0, 1e-9);

    expect_contains(interp, "ml_accuracy([1,0,1,1], [1,0,0,1])", "0.75");
}

TEST(ReplCommandsTest, wave60_bignum_ops) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_factorial(n)");
    expect_contains(interp, "help", "bigint_gcd(\"a\",\"b\")");

    expect_ok(interp, "f = bigint_factorial(10)");
    expect_ok(interp, "fib = bigint_fib(10)");
    expect_ok(interp, "g = bigint_gcd(\"48\", \"18\")");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("f"), 3628800.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("fib"), 55.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("g"), 6.0);

    expect_contains(interp, "bigint_factorial(5)", "120");
    expect_contains(interp, "bigint_gcd(\"48\", \"18\")", "6");
}

TEST(ReplCommandsTest, wave57_59_graph_geo_combo_numthy) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_pagerank(A)");
    expect_contains(interp, "help", "geo_dist2d(x1,y1,x2,y2)");
    expect_contains(interp, "help", "combo_nchoosek(n,k)");
    expect_contains(interp, "help", "numthy_gcd(a,b)");

    expect_contains(interp, "geo_dist2d(0, 0, 3, 4)", "5");
    expect_contains(interp, "combo_nchoosek(5, 2)", "10");
    expect_contains(interp, "numthy_gcd(48, 18)", "6");

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_contains(interp, "graph_dijkstra_dist(A, 0, 4)", "7");
    expect_ok(interp, "pr = graph_pagerank(A)");
    ASSERT_TRUE(interp.state().matrices.count("pr") > 0);
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "area = geo_convex_hull_area(sq)");
    EXPECT_NEAR(interp.state().scalars.at("area"), 1.0, 1e-9);
    expect_contains(interp, "geo_convex_hull_area(sq)", "1");

    expect_contains(interp, "help", "geo_convex_hull(P)");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    ASSERT_TRUE(interp.state().matrices.count("hull") > 0);
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("hull").cols(), 2u);
    expect_ok(interp, "hull_area = geo_convex_hull_area(hull)");
    EXPECT_NEAR(interp.state().scalars.at("hull_area"), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave57_59_control_quantum) {
    Interpreter interp;
    expect_contains(interp, "help", "control_step_final(num,den)");
    expect_contains(interp, "help", "quantum_hadamard(psi)");

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "yfinal = control_step_final(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("yfinal"), 1.0, 0.05);
    expect_contains(interp, "control_step_final([1], [1, 1])", "\n");

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_TRUE(interp.state().matrices.count("hpsi") > 0);
    const auto& hpsi = interp.state().matrices.at("hpsi");
    EXPECT_EQ(hpsi.rows(), 2u);
    EXPECT_NEAR(hpsi(0, 0), 1.0 / std::sqrt(2.0), 1e-6);
    EXPECT_NEAR(hpsi(1, 0), 1.0 / std::sqrt(2.0), 1e-6);
    expect_contains(interp, "quantum_hadamard([1; 0])", "state =");
}

TEST(ReplCommandsTest, wave64_finance_info_bindings) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_call(S,K,T,r,sigma)");
    expect_contains(interp, "help", "finance_npv(rate,cf)");
    expect_contains(interp, "help", "finance_sharpe(r)");
    expect_contains(interp, "help", "info_entropy(p)");

    expect_contains(interp, "finance_npv(0.1, [-100, 50, 60])", "-4.958");

    expect_ok(interp, "cf = [100, 50, 60]");
    expect_ok(interp, "v = finance_npv(0.1, cf)");
    EXPECT_NEAR(interp.state().scalars.at("v"), 100.0 + 50.0 / 1.1 + 60.0 / 1.21, 1e-6);

    expect_ok(interp, "h = info_entropy([0.5; 0.5])");
    EXPECT_NEAR(interp.state().scalars.at("h"), 1.0, 1e-9);

    expect_ok(interp, "sh = finance_sharpe([0.1; 0.2; -0.05])");
    EXPECT_GT(interp.state().scalars.at("sh"), 0.0);

    expect_contains(interp, "info_entropy([0.5, 0.5])", "1");
}

TEST(ReplCommandsTest, wave64_cplx_bindings) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_joukowski(re,im)");
    expect_contains(interp, "help", "cplx_cross_ratio(z1re,z1im");

    expect_ok(interp, "j = cplx_joukowski(2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("j"), 2.5, 1e-9);
    expect_contains(interp, "cplx_joukowski(2, 0)", "2.5");

    expect_ok(interp, "cr = cplx_cross_ratio(0, 0, 1, 0, 2, 0, 3, 0)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), 4.0 / 3.0, 1e-9);
    expect_contains(interp, "cplx_cross_ratio(0, 0, 1, 0, 2, 0, 3, 0)", "1.33333");
}

TEST(ReplCommandsTest, wave64_tensorops_and_finance_bs) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_norm(T)");

    expect_ok(interp, "n = tensorops_norm([3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("n"), 5.0, 1e-9);
    expect_contains(interp, "tensorops_norm([3, 4])", "5");

    expect_ok(interp, "c = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    expect_contains(interp, "finance_bs_call(100, 100, 1, 0.05, 0.2)", "\n");
}

TEST(ReplCommandsTest, wave65_diffgeo_sphere_curvatures) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_gaussian_sphere()");
    expect_contains(interp, "help", "diffgeo_mean_sphere()");

    expect_ok(interp, "K = diffgeo_gaussian_sphere()");
    EXPECT_NEAR(interp.state().scalars.at("K"), 1.0, 0.05);
    expect_contains(interp, "diffgeo_gaussian_sphere()", "\n");

    expect_ok(interp, "H = diffgeo_mean_sphere()");
    EXPECT_NEAR(std::abs(interp.state().scalars.at("H")), 1.0, 0.08);
    expect_contains(interp, "diffgeo_mean_sphere()", "\n");
}

TEST(ReplCommandsTest, wave65_topo_euler_tetrahedron) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_euler_tetrahedron()");

    expect_ok(interp, "chi = topo_euler_tetrahedron()");
    EXPECT_NEAR(interp.state().scalars.at("chi"), 1.0, 1e-9);
    expect_contains(interp, "topo_euler_tetrahedron()", "1");
}

TEST(ReplCommandsTest, wave66_tensorops_matmul) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_matmul(A,B)");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = [5, 6; 7, 8]");
    expect_ok(interp, "C = tensorops_matmul(A, B)");
    const auto& C = interp.state().matrices.at("C");
    EXPECT_NEAR(C(0, 0), 19.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 22.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 43.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 50.0, 1e-9);
    expect_contains(interp, "tensorops_matmul([1, 2; 3, 4], [5, 6; 7, 8])", "19");
}

TEST(ReplCommandsTest, wave66_combo_numthy_finance_scalars) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_factorial(n)");
    expect_contains(interp, "help", "numthy_partition(n)");
    expect_contains(interp, "help", "finance_bond_price(c,y,n,fv)");

    expect_ok(interp, "f = combo_factorial(5)");
    EXPECT_NEAR(interp.state().scalars.at("f"), 120.0, 1e-9);
    expect_contains(interp, "combo_factorial(5)", "120");

    expect_ok(interp, "p = numthy_partition(5)");
    EXPECT_NEAR(interp.state().scalars.at("p"), 7.0, 1e-9);
    expect_contains(interp, "numthy_partition(5)", "7");

    expect_ok(interp, "bp = finance_bond_price(0.05, 0.05, 10)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 100.0, 1e-6);
    expect_contains(interp, "finance_bond_price(0.05, 0.05, 10)", "100");
}

TEST(ReplCommandsTest, wave67_tensorops_einsum) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_einsum(A,B)");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = [5, 6; 7, 8]");
    expect_ok(interp, "C = tensorops_einsum(A, B)");
    const auto& C = interp.state().matrices.at("C");
    EXPECT_NEAR(C(0, 0), 19.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 22.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 43.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 50.0, 1e-9);
    expect_contains(interp, "tensorops_einsum([1, 2; 3, 4], [5, 6; 7, 8])", "19");
}

TEST(ReplCommandsTest, wave67_geo_polygon_area) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_polygon_area(P)");

    expect_ok(interp, "P = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "a = geo_polygon_area(P)");
    EXPECT_NEAR(interp.state().scalars.at("a"), 1.0, 1e-9);
    expect_contains(interp, "geo_polygon_area(P)", "1");

    expect_ok(interp, "T = [0, 0; 2, 0; 1, 2]");
    expect_ok(interp, "ta = geo_polygon_area(T)");
    EXPECT_NEAR(interp.state().scalars.at("ta"), 2.0, 1e-9);
    expect_contains(interp, "geo_polygon_area([0, 0; 2, 0; 1, 2])", "2");
}

TEST(ReplCommandsTest, wave68_finance_irr) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_irr(cf)");

    expect_contains(interp, "finance_irr([-100, 110])", "0.1");

    expect_ok(interp, "cf = [-100; 110]");
    expect_ok(interp, "r = finance_irr(cf)");
    EXPECT_NEAR(interp.state().scalars.at("r"), 0.1, 1e-6);

    expect_ok(interp, "r2 = finance_irr([-100, 110])");
    EXPECT_NEAR(interp.state().scalars.at("r2"), 0.1, 1e-6);
}

TEST(ReplCommandsTest, wave68_info_kl_divergence) {
    Interpreter interp;
    expect_contains(interp, "help", "info_kl_divergence(p,q)");

    expect_ok(interp, "p = [0.3; 0.4; 0.3]");
    expect_ok(interp, "kl0 = info_kl_divergence(p, p)");
    EXPECT_NEAR(interp.state().scalars.at("kl0"), 0.0, 1e-9);

    expect_ok(interp, "kl = info_kl_divergence([0.4; 0.6], [0.5; 0.5])");
    EXPECT_GT(interp.state().scalars.at("kl"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("kl"), 0.029049, 1e-4);

    expect_contains(interp, "info_kl_divergence([0.4, 0.6], [0.5, 0.5])", "0.029");
}

TEST(ReplCommandsTest, wave69_control_dcgain) {
    Interpreter interp;
    expect_contains(interp, "help", "control_dcgain(num,den)");

    expect_contains(interp, "control_dcgain([1], [1, 1])", "1");

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "g = control_dcgain(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 1.0, 1e-6);
}

TEST(ReplCommandsTest, wave69_info_cross_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_cross_entropy(p,q)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "q = [0.25; 0.75]");
    expect_ok(interp, "ce = info_cross_entropy(p, q)");
    EXPECT_GT(interp.state().scalars.at("ce"), 0.0);

    expect_contains(interp, "info_cross_entropy([0.5, 0.5], [0.25, 0.75])", "1.");
}

TEST(ReplCommandsTest, wave69_matrix_literal_negative) {
    Interpreter interp;
    expect_ok(interp, "cf = [-100, 110]");
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("cf").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("cf")(0, 0), -100.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cf")(0, 1), 110.0, 1e-9);

    expect_ok(interp, "cf = [-100; 110]");
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cf").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("cf")(0, 0), -100.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cf")(1, 0), 110.0, 1e-9);
}

TEST(ReplCommandsTest, wave70_control_is_stable) {
    Interpreter interp;
    expect_contains(interp, "help", "control_is_stable(num,den)");

    expect_contains(interp, "control_is_stable([1], [1, 1])", "1");

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "s = control_is_stable(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 1.0, 1e-6);
}

TEST(ReplCommandsTest, wave70_finance_var) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_var(r)");

    expect_ok(interp, "r = [0.1; -0.05; 0.2]");
    expect_ok(interp, "v = finance_var(r)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("v")));

    expect_contains(interp, "finance_var([0.1; -0.05; 0.2])", "0.");
}

TEST(ReplCommandsTest, wave70_info_mutual_info) {
    Interpreter interp;
    expect_contains(interp, "help", "info_mutual_info(joint)");

    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "mi = info_mutual_info(J)");
    EXPECT_NEAR(interp.state().scalars.at("mi"), 0.0, 1e-9);

    expect_contains(interp, "info_mutual_info([0.25, 0; 0, 0.25])", "1");
}

TEST(ReplCommandsTest, wave70_combo_nchoosek_assignment) {
    Interpreter interp;
    expect_ok(interp, "c = combo_nchoosek(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 10.0, 1e-9);
    expect_contains(interp, "combo_nchoosek(5, 2)", "10");
}

TEST(ReplCommandsTest, wave70_numthy_gcd_assignment) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-9);
    expect_contains(interp, "numthy_gcd(48, 18)", "6");
}

TEST(ReplCommandsTest, wave71_finance_cvar) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_cvar(r)");

    expect_ok(interp, "r = [0.1; -0.05; 0.2]");
    expect_ok(interp, "cv = finance_cvar(r)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("cv")));

    expect_contains(interp, "finance_cvar([0.1; -0.05; 0.2])", "0.");
}

TEST(ReplCommandsTest, wave71_info_js_divergence) {
    Interpreter interp;
    expect_contains(interp, "help", "info_js_divergence(p,q)");

    expect_ok(interp, "p = [1; 0]");
    expect_ok(interp, "q = [0; 1]");
    expect_ok(interp, "js = info_js_divergence(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("js"), 1.0, 1e-9);

    expect_contains(interp, "info_js_divergence([1, 0], [0, 1])", "1");
}

TEST(ReplCommandsTest, wave71_quantum_von_neumann_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_von_neumann_entropy(rho)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "S = quantum_von_neumann_entropy(rho)");
    EXPECT_NEAR(interp.state().scalars.at("S"), 0.0, 1e-9);

    expect_contains(interp, "quantum_von_neumann_entropy([1, 0; 0, 0])", "0");
}

TEST(ReplCommandsTest, wave72_finance_sortino) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_sortino(r)");

    expect_ok(interp, "ret = [0.1; -0.05; 0.2]");
    expect_ok(interp, "sortino = finance_sortino(ret)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("sortino")));
    EXPECT_GT(interp.state().scalars.at("sortino"), 0.0);

    expect_contains(interp, "finance_sortino([0.1; -0.05; 0.2])", "1.666");
}

TEST(ReplCommandsTest, wave72_info_tv_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "info_tv_distance(p,q)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "q = [1; 0]");
    expect_ok(interp, "tv = info_tv_distance(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("tv"), 0.5, 1e-9);

    expect_contains(interp, "info_tv_distance([0.5; 0.5], [1; 0])", "0.5");
}

TEST(ReplCommandsTest, wave72_quantum_fidelity) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_fidelity(rho,sigma)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "F = quantum_fidelity(rho, rho)");
    EXPECT_NEAR(interp.state().scalars.at("F"), 1.0, 1e-6);

    expect_contains(interp, "quantum_fidelity([1, 0; 0, 0], [1, 0; 0, 0])", "1");
}

TEST(ReplCommandsTest, wave73_finance_max_drawdown) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_max_drawdown(equity)");

    expect_ok(interp, "equity = [100; 110; 105; 120; 90; 95]");
    expect_ok(interp, "mdd = finance_max_drawdown(equity)");
    EXPECT_NEAR(interp.state().scalars.at("mdd"), 0.25, 1e-6);

    expect_contains(interp, "finance_max_drawdown([100; 110; 105; 120; 90; 95])", "0.25");
}

TEST(ReplCommandsTest, wave73_info_hellinger_dist) {
    Interpreter interp;
    expect_contains(interp, "help", "info_hellinger_dist(p,q)");

    expect_ok(interp, "p = [1; 0]");
    expect_ok(interp, "q = [0; 1]");
    expect_ok(interp, "hd = info_hellinger_dist(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("hd"), 1.0, 1e-9);

    expect_contains(interp, "info_hellinger_dist([1; 0], [0; 1])", "1");
}

TEST(ReplCommandsTest, wave73_quantum_trace_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_trace_distance(rho,sigma)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "T = quantum_trace_distance(rho, rho)");
    EXPECT_NEAR(interp.state().scalars.at("T"), 0.0, 1e-9);

    expect_contains(interp, "quantum_trace_distance([1, 0; 0, 0], [1, 0; 0, 0])", "0");
}

TEST(ReplCommandsTest, wave74_finance_kelly_fraction) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_kelly_fraction(p,b)");

    expect_ok(interp, "kelly = finance_kelly_fraction(0.6, 2.0)");
    EXPECT_NEAR(interp.state().scalars.at("kelly"), 0.4, 1e-9);

    expect_contains(interp, "finance_kelly_fraction(0.6, 2.0)", "0.4");
}

TEST(ReplCommandsTest, wave74_info_renyi_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_renyi_entropy(alpha,p)");

    expect_ok(interp, "p = [0.25; 0.25; 0.25; 0.25]");
    expect_ok(interp, "r2 = info_renyi_entropy(2.0, p)");
    EXPECT_NEAR(interp.state().scalars.at("r2"), 2.0, 1e-9);

    expect_contains(interp, "info_renyi_entropy(2.0, [0.25; 0.25; 0.25; 0.25])", "2");
}

TEST(ReplCommandsTest, wave74_quantum_concurrence) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_concurrence(rho)");

    expect_ok(interp, "rho4 = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "Cq = quantum_concurrence(rho4)");
    EXPECT_NEAR(interp.state().scalars.at("Cq"), 0.0, 1e-9);

    expect_contains(interp,
                    "quantum_concurrence([1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0])",
                    "0");
}

TEST(ReplCommandsTest, wave75_finance_compound) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_compound(principal,rate,n_periods,compounds_per_period)");

    expect_ok(interp, "fv = finance_compound(100, 0.1, 3)");
    EXPECT_NEAR(interp.state().scalars.at("fv"), 133.1, 1e-6);

    expect_contains(interp, "finance_compound(100, 0.1, 3)", "133.1");
}

TEST(ReplCommandsTest, wave75_info_redundancy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_redundancy(p)");

    expect_ok(interp, "p = [0.9; 0.1]");
    expect_ok(interp, "red = info_redundancy(p)");
    EXPECT_GT(interp.state().scalars.at("red"), 0.0);

    expect_contains(interp, "info_redundancy([0.9; 0.1])", "0.5");
}

TEST(ReplCommandsTest, wave75_quantum_entanglement_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_entanglement_entropy(psi,dim_a,dim_b)");

    expect_ok(interp, "psi = [1; 0; 0; 0]");
    expect_ok(interp, "Ee = quantum_entanglement_entropy(psi, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("Ee"), 0.0, 1e-9);

    expect_contains(interp, "quantum_entanglement_entropy([1; 0; 0; 0], 2, 2)", "0");
}

TEST(ReplCommandsTest, wave76_finance_continuous_compound) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_continuous_compound(principal,rate,t)");

    expect_ok(interp, "cc = finance_continuous_compound(100, 0.1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), 100.0 * std::exp(0.1), 1e-6);

    expect_contains(interp, "finance_continuous_compound(100, 0.1, 1)", "110.517");
}

TEST(ReplCommandsTest, wave76_info_efficiency) {
    Interpreter interp;
    expect_contains(interp, "help", "info_efficiency(p)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "eff = info_efficiency(p)");
    EXPECT_NEAR(interp.state().scalars.at("eff"), 1.0, 1e-9);

    expect_contains(interp, "info_efficiency([0.5; 0.5])", "1");
}

TEST(ReplCommandsTest, wave76_quantum_expectation) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_expectation(psi,A)");

    expect_ok(interp, "ex = quantum_expectation([1; 0], [1, 0; 0, -1])");
    EXPECT_NEAR(interp.state().scalars.at("ex"), 1.0, 1e-9);

    expect_contains(interp, "quantum_expectation([1; 0], [1, 0; 0, -1])", "1");
}

TEST(ReplCommandsTest, wave77_finance_pv) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_pv(rate,n,pmt,fv)");

    expect_ok(interp, "pv0 = finance_pv(0, 5, -10, 0)");
    EXPECT_NEAR(interp.state().scalars.at("pv0"), 50.0, 1e-9);

    expect_contains(interp, "finance_pv(0, 5, -10, 0)", "50");
}

TEST(ReplCommandsTest, wave77_info_channel_capacity_bsc) {
    Interpreter interp;
    expect_contains(interp, "help", "info_channel_capacity_bsc(p_error)");

    expect_ok(interp, "cap = info_channel_capacity_bsc(0)");
    EXPECT_NEAR(interp.state().scalars.at("cap"), 1.0, 1e-9);

    expect_contains(interp, "info_channel_capacity_bsc(0)", "1");
}

TEST(ReplCommandsTest, wave77_quantum_expectation_dm) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_expectation_dm(rho,op)");

    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "exdm = quantum_expectation_dm(rho, [1, 0; 0, -1])");
    EXPECT_NEAR(interp.state().scalars.at("exdm"), 1.0, 1e-9);

    expect_contains(interp, "quantum_expectation_dm([1, 0; 0, 0], [1, 0; 0, -1])", "1");
}

TEST(ReplCommandsTest, wave78_finance_fv_annuity) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_fv_annuity(rate,n,pmt,pv0)");

    expect_ok(interp, "fv0 = finance_fv_annuity(0, 5, -10, 0)");
    EXPECT_NEAR(interp.state().scalars.at("fv0"), 50.0, 1e-9);

    expect_contains(interp, "finance_fv_annuity(0, 5, -10, 0)", "50");
}

TEST(ReplCommandsTest, wave78_info_channel_capacity_bec) {
    Interpreter interp;
    expect_contains(interp, "help", "info_channel_capacity_bec(epsilon)");

    expect_ok(interp, "capbec = info_channel_capacity_bec(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("capbec"), 0.5, 1e-9);

    expect_contains(interp, "info_channel_capacity_bec(0.5)", "0.5");
}

TEST(ReplCommandsTest, wave78_quantum_inner) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_inner(bra,ket)");

    expect_ok(interp, "inn = quantum_inner([1; 0], [1; 0])");
    EXPECT_NEAR(interp.state().scalars.at("inn"), 1.0, 1e-9);

    expect_contains(interp, "quantum_inner([1; 0], [1; 0])", "1");
}

TEST(ReplCommandsTest, wave79_finance_pmt_annuity) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_pmt_annuity(rate,n,pv0,fv)");

    expect_ok(interp, "pmt = finance_pmt_annuity(0, 5, -50, 0)");
    EXPECT_NEAR(interp.state().scalars.at("pmt"), 10.0, 1e-9);

    expect_contains(interp, "finance_pmt_annuity(0, 5, -50, 0)", "10");
}

TEST(ReplCommandsTest, wave79_info_shannon_hartley) {
    Interpreter interp;
    expect_contains(interp, "help", "info_shannon_hartley(bandwidth_hz,snr_linear)");

    expect_ok(interp, "sh = info_shannon_hartley(1000000, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), 1e6, 1.0);

    expect_contains(interp, "info_shannon_hartley(1000000, 1)", "1000000");
}

TEST(ReplCommandsTest, wave79_quantum_ket_normalise) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_normalise(psi)");

    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_TRUE(interp.state().matrices.count("psi_n") > 0);
    const auto& psi_n = interp.state().matrices.at("psi_n");
    EXPECT_EQ(psi_n.rows(), 2u);
    EXPECT_EQ(psi_n.cols(), 1u);
    EXPECT_NEAR(psi_n(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(psi_n(1, 0), 0.0, 1e-9);
    double psi_norm_sq = 0.0;
    for (size_t i = 0; i < psi_n.rows(); ++i) {
        psi_norm_sq += psi_n(i, 0) * psi_n(i, 0);
    }
    EXPECT_NEAR(std::sqrt(psi_norm_sq), 1.0, 1e-9);

    expect_contains(interp, "quantum_ket_normalise([2; 0])", "state =");
}

TEST(ReplCommandsTest, wave80_finance_binomial_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_binomial_call(S,K,T,r,sigma,steps)");

    expect_ok(interp, "bin_c = finance_binomial_call(100, 100, 1, 0.05, 0.2, 200)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bin_c")));
    EXPECT_GT(interp.state().scalars.at("bin_c"), 0.0);

    expect_contains(interp, "finance_binomial_call(100, 100, 1, 0.05, 0.2, 200)", "10.");
}

TEST(ReplCommandsTest, wave80_info_differential_entropy_gaussian) {
    Interpreter interp;
    expect_contains(interp, "help", "info_differential_entropy_gaussian(sigma)");

    expect_ok(interp, "hgauss = info_differential_entropy_gaussian(1)");
    EXPECT_NEAR(interp.state().scalars.at("hgauss"), 1.4189385332046727, 1e-3);

    expect_contains(interp, "info_differential_entropy_gaussian(1)", "1.418");
}

TEST(ReplCommandsTest, wave80_quantum_partial_trace) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_partial_trace(rho,d1,d2,subsystem)");

    expect_ok(interp, "rho4 = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "rhoA = quantum_partial_trace(rho4, 2, 2, 0)");
    ASSERT_TRUE(interp.state().matrices.count("rhoA") > 0);
    const auto& rhoA = interp.state().matrices.at("rhoA");
    EXPECT_EQ(rhoA.rows(), 2u);
    EXPECT_EQ(rhoA.cols(), 2u);
    EXPECT_NEAR(rhoA(0, 0), 1.0, 1e-9);

    expect_contains(interp,
                    "quantum_partial_trace([1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0], 2, 2, 0)",
                    "rho =");
}

TEST(ReplCommandsTest, wave81_finance_binomial_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_binomial_put(S,K,T,r,sigma,steps)");

    expect_ok(interp, "bin_p = finance_binomial_put(100, 100, 1, 0.05, 0.2, 200)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bin_p")));
    EXPECT_GT(interp.state().scalars.at("bin_p"), 0.0);

    expect_contains(interp, "finance_binomial_put(100, 100, 1, 0.05, 0.2, 200)", "5.56");
}

TEST(ReplCommandsTest, wave81_info_differential_entropy_uniform) {
    Interpreter interp;
    expect_contains(interp, "help", "info_differential_entropy_uniform(a,b)");

    expect_ok(interp, "hu = info_differential_entropy_uniform(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hu"), 0.0, 1e-9);

    expect_contains(interp, "info_differential_entropy_uniform(0, 1)", "0");
}

TEST(ReplCommandsTest, wave81_finance_bs_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_put(S,K,T,r,sigma)");

    expect_ok(interp, "bs_p = finance_bs_put(100, 100, 1, 0.05, 0.2)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bs_p")));
    EXPECT_GT(interp.state().scalars.at("bs_p"), 0.0);

    expect_contains(interp, "finance_bs_put(100, 100, 1, 0.05, 0.2)", "5.57");
}

TEST(ReplCommandsTest, wave82_finance_bs_gamma) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_gamma(S,K,T,r,sigma)");

    expect_ok(interp, "g = finance_bs_gamma(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("g"), 0.0);

    expect_contains(interp, "finance_bs_gamma(100, 100, 1, 0.05, 0.2)", "0.018762");
}

TEST(ReplCommandsTest, wave82_finance_bond_duration) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_duration(c,y,n)");

    expect_ok(interp, "dur = finance_bond_duration(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("dur"), 5.0, 1e-6);

    expect_contains(interp, "finance_bond_duration(0, 0.05, 5)", "5");
}

TEST(ReplCommandsTest, wave82_info_rate_distortion_gaussian) {
    Interpreter interp;
    expect_contains(interp, "help", "info_rate_distortion_gaussian(variance,distortion)");

    expect_ok(interp, "rd = info_rate_distortion_gaussian(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("rd"), 1.0, 1e-9);

    expect_contains(interp, "info_rate_distortion_gaussian(1, 0.25)", "1");
}

TEST(ReplCommandsTest, wave83_finance_bs_delta) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_delta(S,K,T,r,sigma,call)");

    expect_ok(interp, "d = finance_bs_delta(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_GT(interp.state().scalars.at("d"), 0.0);
    EXPECT_LT(interp.state().scalars.at("d"), 1.0);

    expect_contains(interp, "finance_bs_delta(100, 100, 1, 0.05, 0.2, 1)", "0.636831");
}

TEST(ReplCommandsTest, wave83_finance_bs_vega) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_vega(S,K,T,r,sigma)");

    expect_ok(interp, "v = finance_bs_vega(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("v"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("v")));

    expect_contains(interp, "finance_bs_vega(100, 100, 1, 0.05, 0.2)", "37.524035");
}

TEST(ReplCommandsTest, wave83_finance_bond_modified_duration) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_modified_duration(c,y,n)");

    expect_ok(interp, "mdur = finance_bond_modified_duration(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("mdur"), 5.0 / 1.05, 1e-6);

    expect_contains(interp, "finance_bond_modified_duration(0, 0.05, 5)", "4.761905");
}

TEST(ReplCommandsTest, wave84_finance_bs_theta) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_theta(S,K,T,r,sigma,call)");

    expect_ok(interp, "th = finance_bs_theta(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_LT(interp.state().scalars.at("th"), 0.0);

    expect_contains(interp, "finance_bs_theta(100, 100, 1, 0.05, 0.2, 1)", "-6.414028");
}

TEST(ReplCommandsTest, wave84_finance_bs_rho) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_rho(S,K,T,r,sigma,call)");

    expect_ok(interp, "rh = finance_bs_rho(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_GT(interp.state().scalars.at("rh"), 0.0);

    expect_contains(interp, "finance_bs_rho(100, 100, 1, 0.05, 0.2, 1)", "53.232482");
}

TEST(ReplCommandsTest, wave84_finance_bond_convexity) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_convexity(c,y,n)");

    expect_ok(interp, "conv = finance_bond_convexity(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("conv"), 30.0 / (1.05 * 1.05), 1e-6);

    expect_contains(interp, "finance_bond_convexity(0, 0.05, 5)", "27.210884");
}

TEST(ReplCommandsTest, wave85_finance_bond_ytm) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_ytm(price,c,n)");

    expect_ok(interp, "bp = finance_bond_price(0.05, 0.07, 10)");
    expect_ok(interp, "ytm = finance_bond_ytm(bp, 0.05, 10)");
    EXPECT_NEAR(interp.state().scalars.at("ytm"), 0.07, 1e-5);

    expect_contains(interp, "finance_bond_ytm(85.952837, 0.05, 10)", "0.07");
}

TEST(ReplCommandsTest, wave85_info_source_coding_rate) {
    Interpreter interp;
    expect_contains(interp, "help", "info_source_coding_rate(p)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "scr = info_source_coding_rate(p)");
    EXPECT_NEAR(interp.state().scalars.at("scr"), 1.0, 1e-9);

    expect_contains(interp, "info_source_coding_rate([0.5; 0.5])", "1");
}

TEST(ReplCommandsTest, wave85_info_tsallis_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_tsallis_entropy(q,p)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "ts = info_tsallis_entropy(2.0, p)");
    EXPECT_NEAR(interp.state().scalars.at("ts"), 0.5, 1e-9);

    expect_contains(interp, "info_tsallis_entropy(2.0, [0.5; 0.5])", "0.5");
}

TEST(ReplCommandsTest, wave86_finance_bs_implied_vol) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_implied_vol(price,S,K,T,r,call)");

    expect_ok(interp, "mp = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    expect_ok(interp, "iv = finance_bs_implied_vol(mp, 100, 100, 1, 0.05, 1)");
    EXPECT_NEAR(interp.state().scalars.at("iv"), 0.2, 1e-5);

    expect_contains(interp, "finance_bs_implied_vol(10.450583572185565, 100, 100, 1, 0.05, 1)",
                    "0.2");
}

TEST(ReplCommandsTest, wave86_finance_portfolio_return) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_portfolio_return(weights,returns)");

    expect_ok(interp, "w = [0.6; 0.4]");
    expect_ok(interp, "ret = [0.1; 0.05]");
    expect_ok(interp, "pr = finance_portfolio_return(w, ret)");
    EXPECT_NEAR(interp.state().scalars.at("pr"), 0.08, 1e-9);

    expect_contains(interp, "finance_portfolio_return([0.6; 0.4], [0.1; 0.05])", "0.08");
}

TEST(ReplCommandsTest, wave86_finance_portfolio_variance) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_portfolio_variance(weights,cov)");

    expect_ok(interp, "w = [0.5; 0.5]");
    expect_ok(interp, "cov = [0.04, 0.01; 0.01, 0.09]");
    expect_ok(interp, "pvar = finance_portfolio_variance(w, cov)");
    EXPECT_NEAR(interp.state().scalars.at("pvar"), 0.0375, 1e-9);

    expect_contains(interp, "finance_portfolio_variance([0.5; 0.5], [0.04, 0.01; 0.01, 0.09])",
                    "0.0375");
}

TEST(ReplCommandsTest, wave87_info_joint_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_joint_entropy(joint,rows,cols)");

    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "hj = info_joint_entropy(J, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("hj"), 2.0, 1e-9);

    expect_contains(interp, "info_joint_entropy([0.25, 0.25; 0.25, 0.25], 2, 2)", "2");
}

TEST(ReplCommandsTest, wave87_info_conditional_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_conditional_entropy(joint,rows,cols)");

    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "hc = info_conditional_entropy(J, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("hc"), 1.0, 1e-9);

    expect_contains(interp, "info_conditional_entropy([0.25, 0.25; 0.25, 0.25], 2, 2)", "1");
}

TEST(ReplCommandsTest, wave87_info_sample_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_sample_entropy(x,m,r)");

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "se = info_sample_entropy(x, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("se"), 0.0, 1e-9);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("se")));

    expect_contains(interp, "info_sample_entropy([1; 2; 3; 4; 5], 2, 0.5)", "0");
}

TEST(ReplCommandsTest, wave88_quantum_pauli_x) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_x()");

    expect_ok(interp, "X = quantum_pauli_x()");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    const auto& X = interp.state().matrices.at("X");
    EXPECT_EQ(X.rows(), 2u);
    EXPECT_EQ(X.cols(), 2u);
    EXPECT_NEAR(X(0, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_x()", "op =");
}

TEST(ReplCommandsTest, wave88_quantum_pauli_z) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_z()");

    expect_ok(interp, "Z = quantum_pauli_z()");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 2u);
    EXPECT_EQ(Z.cols(), 2u);
    EXPECT_NEAR(Z(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(Z(1, 1), -1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_z()", "op =");
}

TEST(ReplCommandsTest, wave88_quantum_cnot_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_cnot_gate()");

    expect_ok(interp, "CNOT = quantum_cnot_gate()");
    ASSERT_GT(interp.state().matrices.count("CNOT"), 0u);
    const auto& CNOT = interp.state().matrices.at("CNOT");
    EXPECT_EQ(CNOT.rows(), 4u);
    EXPECT_EQ(CNOT.cols(), 4u);

    expect_contains(interp, "quantum_cnot_gate()", "op =");
}

TEST(ReplCommandsTest, wave91_quantum_pauli_plus) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_plus()");

    expect_ok(interp, "Pp = quantum_pauli_plus()");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    const auto& Pp = interp.state().matrices.at("Pp");
    EXPECT_EQ(Pp.rows(), 2u);
    EXPECT_EQ(Pp.cols(), 2u);
    EXPECT_NEAR(Pp(0, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_plus()", "op =");
}

TEST(ReplCommandsTest, wave91_quantum_pauli_minus) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_minus()");

    expect_ok(interp, "Pm = quantum_pauli_minus()");
    ASSERT_GT(interp.state().matrices.count("Pm"), 0u);
    const auto& Pm = interp.state().matrices.at("Pm");
    EXPECT_EQ(Pm.rows(), 2u);
    EXPECT_EQ(Pm.cols(), 2u);
    EXPECT_NEAR(Pm(1, 0), 1.0, 1e-9);

    expect_contains(interp, "quantum_pauli_minus()", "op =");
}

TEST(ReplCommandsTest, wave91_quantum_toffoli_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_toffoli_gate()");

    expect_ok(interp, "T = quantum_toffoli_gate()");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    const auto& T = interp.state().matrices.at("T");
    EXPECT_EQ(T.rows(), 8u);
    EXPECT_EQ(T.cols(), 8u);
    EXPECT_NEAR(T(6, 7), 1.0, 1e-9);
    EXPECT_NEAR(T(7, 6), 1.0, 1e-9);

    expect_contains(interp, "quantum_toffoli_gate()", "op =");
}

TEST(ReplCommandsTest, wave89_quantum_pauli_y) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_pauli_y()");

    expect_ok(interp, "Y = quantum_pauli_y()");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    const auto& Y = interp.state().matrices.at("Y");
    EXPECT_EQ(Y.rows(), 2u);
    EXPECT_EQ(Y.cols(), 2u);

    expect_contains(interp, "quantum_pauli_y()", "op =");
}

TEST(ReplCommandsTest, wave89_quantum_swap_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_swap_gate()");

    expect_ok(interp, "SWAP = quantum_swap_gate()");
    ASSERT_GT(interp.state().matrices.count("SWAP"), 0u);
    const auto& SWAP = interp.state().matrices.at("SWAP");
    EXPECT_EQ(SWAP.rows(), 4u);
    EXPECT_EQ(SWAP.cols(), 4u);
    EXPECT_NEAR(SWAP(1, 2), 1.0, 1e-9);
    EXPECT_NEAR(SWAP(2, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_swap_gate()", "op =");
}

TEST(ReplCommandsTest, wave89_quantum_identity) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_identity()");

    expect_ok(interp, "I2 = quantum_identity()");
    ASSERT_GT(interp.state().matrices.count("I2"), 0u);
    const auto& I2 = interp.state().matrices.at("I2");
    EXPECT_EQ(I2.rows(), 2u);
    EXPECT_EQ(I2.cols(), 2u);
    EXPECT_NEAR(I2(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(I2(1, 1), 1.0, 1e-9);

    expect_contains(interp, "quantum_identity()", "op =");
}

TEST(ReplCommandsTest, wave90_quantum_hadamard_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_hadamard_gate()");

    expect_ok(interp, "H = quantum_hadamard_gate()");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    const auto& H = interp.state().matrices.at("H");
    EXPECT_EQ(H.rows(), 2u);
    EXPECT_EQ(H.cols(), 2u);
    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(H(0, 0), h, 1e-9);

    expect_contains(interp, "quantum_hadamard_gate()", "op =");
}

TEST(ReplCommandsTest, wave90_cplx_hyperbolic_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_hyperbolic_distance(z1re,z1im,z2re,z2im)");

    expect_ok(interp, "hd = cplx_hyperbolic_distance(0, 0, 0.5, 0)");
    EXPECT_GT(interp.state().scalars.at("hd"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("hd")));

    expect_contains(interp, "cplx_hyperbolic_distance(0, 0, 0.5, 0)", "1.");
}

TEST(ReplCommandsTest, wave90_info_lz_complexity) {
    Interpreter interp;
    expect_contains(interp, "help", "info_lz_complexity(seq)");

    expect_ok(interp, "seq = [0; 1; 0; 1; 1; 0]");
    expect_ok(interp, "lz = info_lz_complexity(seq)");
    EXPECT_GE(interp.state().scalars.at("lz"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("lz")));

    expect_contains(interp, "info_lz_complexity([0; 1; 0; 1; 1; 0])", "1.");
}

TEST(ReplCommandsTest, wave92_quantum_rotation_z) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_rotation_z(theta)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Rz = quantum_rotation_z(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Rz"), 0u);
    const auto& Rz = interp.state().matrices.at("Rz");
    EXPECT_EQ(Rz.rows(), 2u);
    EXPECT_EQ(Rz.cols(), 2u);

    expect_contains(interp, "quantum_rotation_z(pi/2)", "op =");
}

TEST(ReplCommandsTest, wave92_quantum_rotation_x) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_rotation_x(theta)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Rx = quantum_rotation_x(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Rx"), 0u);
    const auto& Rx = interp.state().matrices.at("Rx");
    EXPECT_EQ(Rx.rows(), 2u);
    EXPECT_EQ(Rx.cols(), 2u);

    expect_contains(interp, "quantum_rotation_x(pi/2)", "op =");
}

TEST(ReplCommandsTest, wave92_quantum_rotation_y) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_rotation_y(theta)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Ry = quantum_rotation_y(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Ry"), 0u);
    const auto& Ry = interp.state().matrices.at("Ry");
    EXPECT_EQ(Ry.rows(), 2u);
    EXPECT_EQ(Ry.cols(), 2u);

    expect_contains(interp, "quantum_rotation_y(pi/2)", "op =");
}

TEST(ReplCommandsTest, wave93_quantum_phase_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_phase_gate(theta)");

    expect_ok(interp, "P = quantum_phase_gate(1.57)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    const auto& P = interp.state().matrices.at("P");
    EXPECT_EQ(P.rows(), 2u);
    EXPECT_EQ(P.cols(), 2u);
    for (size_t i = 0; i < 2u; ++i) {
        for (size_t j = 0; j < 2u; ++j) {
            EXPECT_TRUE(std::isfinite(P(i, j)));
        }
    }

    expect_contains(interp, "quantum_phase_gate(1.57)", "op =");
}

TEST(ReplCommandsTest, wave93_quantum_qft_gate) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_qft_gate(n_qubits)");

    expect_ok(interp, "Q = quantum_qft_gate(2)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    const auto& Q = interp.state().matrices.at("Q");
    EXPECT_EQ(Q.rows(), 4u);
    EXPECT_EQ(Q.cols(), 4u);

    expect_contains(interp, "quantum_qft_gate(2)", "op =");
}

TEST(ReplCommandsTest, wave93_cplx_poisson_kernel) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_poisson_kernel(theta,phi,r)");

    expect_ok(interp, "pk = cplx_poisson_kernel(0, 0, 0.5)");
    EXPECT_GT(interp.state().scalars.at("pk"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("pk")));

    expect_contains(interp, "cplx_poisson_kernel(0, 0, 0.5)", "3");
}

TEST(ReplCommandsTest, wave94_tensorops_inner) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_inner(A,B)");

    expect_ok(interp, "V1 = [1; 2; 3]");
    expect_ok(interp, "V2 = [4; 5; 6]");
    expect_ok(interp, "ti = tensorops_inner(V1, V2)");
    EXPECT_NEAR(interp.state().scalars.at("ti"), 32.0, 1e-9);

    expect_contains(interp, "tensorops_inner([1; 2; 3], [4; 5; 6])", "32");
}

TEST(ReplCommandsTest, wave94_geo_dist3d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist3d(x1,y1,z1,x2,y2,z2)");

    expect_ok(interp, "d3 = geo_dist3d(0, 0, 0, 3, 4, 12)");
    EXPECT_NEAR(interp.state().scalars.at("d3"), 13.0, 1e-9);

    expect_contains(interp, "geo_dist3d(0, 0, 0, 3, 4, 12)", "13");
}

TEST(ReplCommandsTest, wave94_numthy_isprime) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_isprime(n)");

    expect_ok(interp, "ip = numthy_isprime(17)");
    EXPECT_NEAR(interp.state().scalars.at("ip"), 1.0, 1e-9);

    expect_contains(interp, "numthy_isprime(17)", "1");
    expect_contains(interp, "numthy_isprime(18)", "0");
}

TEST(ReplCommandsTest, wave95_combo_stirling2) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_stirling2(n,k)");

    expect_ok(interp, "s2 = combo_stirling2(4, 2)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), 7.0, 1e-9);

    expect_contains(interp, "combo_stirling2(4, 2)", "7");
}

TEST(ReplCommandsTest, wave95_numthy_euler_phi) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_euler_phi(n)");

    expect_ok(interp, "phi = numthy_euler_phi(12)");
    EXPECT_NEAR(interp.state().scalars.at("phi"), 4.0, 1e-9);

    expect_contains(interp, "numthy_euler_phi(12)", "4");
}

TEST(ReplCommandsTest, wave95_numthy_mobius) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_mobius(n)");

    expect_ok(interp, "mu = numthy_mobius(6)");
    EXPECT_NEAR(interp.state().scalars.at("mu"), 1.0, 1e-9);

    expect_contains(interp, "numthy_mobius(6)", "1");
    expect_contains(interp, "numthy_mobius(4)", "0");
}

TEST(ReplCommandsTest, wave96_combo_catalan) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_catalan(n)");

    expect_ok(interp, "cat = combo_catalan(4)");
    EXPECT_NEAR(interp.state().scalars.at("cat"), 14.0, 1e-9);

    expect_contains(interp, "combo_catalan(4)", "14");
}

TEST(ReplCommandsTest, wave96_combo_bell) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_bell(n)");

    expect_ok(interp, "bell = combo_bell(4)");
    EXPECT_NEAR(interp.state().scalars.at("bell"), 15.0, 1e-9);

    expect_contains(interp, "combo_bell(4)", "15");
}

TEST(ReplCommandsTest, wave96_numthy_num_divisors) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_num_divisors(n)");

    expect_ok(interp, "tau = numthy_num_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("tau"), 6.0, 1e-9);

    expect_contains(interp, "numthy_num_divisors(12)", "6");
}

TEST(ReplCommandsTest, wave97_combo_stirling1) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_stirling1(n,k)");

    expect_ok(interp, "s1 = combo_stirling1(4, 2)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), 11.0, 1e-9);

    expect_contains(interp, "combo_stirling1(4, 2)", "11");
}

TEST(ReplCommandsTest, wave97_numthy_lcm) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_lcm(a,b)");

    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-9);

    expect_contains(interp, "numthy_lcm(4, 6)", "12");
}

TEST(ReplCommandsTest, wave97_numthy_mod_pow) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_mod_pow(base,exp,mod)");

    expect_ok(interp, "mp = numthy_mod_pow(2, 10, 1000)");
    EXPECT_NEAR(interp.state().scalars.at("mp"), 24.0, 1e-9);

    expect_contains(interp, "numthy_mod_pow(2, 10, 1000)", "24");
    expect_contains(interp, "numthy_mod_pow(3, 12, 13)", "1");
}

TEST(ReplCommandsTest, wave98_combo_motzkin) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_motzkin(n)");

    expect_ok(interp, "motz = combo_motzkin(4)");
    EXPECT_NEAR(interp.state().scalars.at("motz"), 9.0, 1e-9);

    expect_contains(interp, "combo_motzkin(4)", "9");
}

TEST(ReplCommandsTest, wave98_combo_permutations) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_permutations(n,k)");

    expect_ok(interp, "perm = combo_permutations(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("perm"), 20.0, 1e-9);

    expect_contains(interp, "combo_permutations(5, 2)", "20");
}

TEST(ReplCommandsTest, wave98_numthy_sum_divisors) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_sum_divisors(n)");

    expect_ok(interp, "sigma = numthy_sum_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("sigma"), 28.0, 1e-9);

    expect_contains(interp, "numthy_sum_divisors(12)", "28");
}

TEST(ReplCommandsTest, wave99_numthy_nextprime) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_nextprime(n)");

    expect_ok(interp, "np = numthy_nextprime(10)");
    EXPECT_NEAR(interp.state().scalars.at("np"), 11.0, 1e-9);

    expect_contains(interp, "numthy_nextprime(10)", "11");
    expect_contains(interp, "numthy_nextprime(11)", "13");
}

TEST(ReplCommandsTest, wave99_numthy_liouville) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_liouville(n)");

    expect_ok(interp, "lam = numthy_liouville(12)");
    EXPECT_NEAR(interp.state().scalars.at("lam"), -1.0, 1e-9);

    expect_contains(interp, "numthy_liouville(12)", "-1");
    expect_contains(interp, "numthy_liouville(6)", "1");
}

TEST(ReplCommandsTest, wave99_combo_subfactorial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_subfactorial(n)");

    expect_ok(interp, "subf = combo_subfactorial(4)");
    EXPECT_NEAR(interp.state().scalars.at("subf"), 9.0, 1e-9);

    expect_contains(interp, "combo_subfactorial(4)", "9");
}

TEST(ReplCommandsTest, wave100_numthy_prime_pi) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_prime_pi(n)");

    expect_ok(interp, "pi = numthy_prime_pi(100)");
    EXPECT_NEAR(interp.state().scalars.at("pi"), 25.0, 1e-9);

    expect_contains(interp, "numthy_prime_pi(10)", "4");
    expect_contains(interp, "numthy_prime_pi(100)", "25");
}

TEST(ReplCommandsTest, wave100_numthy_legendre_symbol) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_legendre_symbol(a,p)");

    expect_ok(interp, "ls = numthy_legendre_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ls"), 1.0, 1e-9);

    expect_contains(interp, "numthy_legendre_symbol(2, 7)", "1");
    expect_contains(interp, "numthy_legendre_symbol(3, 7)", "-1");
}

TEST(ReplCommandsTest, wave100_combo_combinations_with_rep) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_combinations_with_rep(n,k)");

    expect_ok(interp, "cwr = combo_combinations_with_rep(3, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cwr"), 6.0, 1e-9);

    expect_contains(interp, "combo_combinations_with_rep(3, 2)", "6");
    expect_contains(interp, "combo_combinations_with_rep(5, 3)", "35");
}

TEST(ReplCommandsTest, wave101_numthy_prevprime) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_prevprime(n)");

    expect_ok(interp, "pp = numthy_prevprime(10)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), 7.0, 1e-9);

    expect_contains(interp, "numthy_prevprime(10)", "7");
    expect_contains(interp, "numthy_prevprime(3)", "2");
}

TEST(ReplCommandsTest, wave101_combo_double_factorial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_double_factorial(n)");

    expect_ok(interp, "df = combo_double_factorial(5)");
    EXPECT_NEAR(interp.state().scalars.at("df"), 15.0, 1e-9);

    expect_contains(interp, "combo_double_factorial(5)", "15");
    expect_contains(interp, "combo_double_factorial(6)", "48");
}

TEST(ReplCommandsTest, wave101_numthy_jacobi_symbol) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_jacobi_symbol(a,n)");

    expect_ok(interp, "js = numthy_jacobi_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("js"), 1.0, 1e-9);

    expect_contains(interp, "numthy_jacobi_symbol(2, 7)", "1");
    expect_contains(interp, "numthy_jacobi_symbol(3, 7)", "-1");
}

TEST(ReplCommandsTest, wave102_numthy_prime_nth) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_prime_nth(n)");

    expect_ok(interp, "pn = numthy_prime_nth(6)");
    EXPECT_NEAR(interp.state().scalars.at("pn"), 13.0, 1e-9);

    expect_contains(interp, "numthy_prime_nth(6)", "13");
    expect_contains(interp, "numthy_prime_nth(1)", "2");
}

TEST(ReplCommandsTest, wave102_numthy_kronecker_symbol) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_kronecker_symbol(a,n)");

    expect_ok(interp, "ks = numthy_kronecker_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ks"), 1.0, 1e-9);

    expect_contains(interp, "numthy_kronecker_symbol(2, 7)", "1");
    expect_contains(interp, "numthy_kronecker_symbol(3, 7)", "-1");
}

TEST(ReplCommandsTest, wave102_numthy_tonelli_shanks) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_tonelli_shanks(n,p)");

    expect_ok(interp, "x = numthy_tonelli_shanks(2, 7)");
    const int64_t root = static_cast<int64_t>(interp.state().scalars.at("x"));
    EXPECT_EQ((root * root) % 7, 2);

    expect_contains(interp, "numthy_tonelli_shanks(2, 7)", "4");
}

TEST(ReplCommandsTest, wave103_ml_precision) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_precision(p,t)");

    expect_ok(interp, "prec = ml_precision([1; 1; 0; 0], [1; 0; 0; 1])");
    EXPECT_NEAR(interp.state().scalars.at("prec"), 0.5, 1e-9);

    expect_contains(interp, "ml_precision([1; 1; 0; 0], [1; 0; 0; 1])", "0.5");
}

TEST(ReplCommandsTest, wave103_ml_recall) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_recall(p,t)");

    expect_ok(interp, "rec = ml_recall([1; 1; 0; 0], [1; 0; 0; 1])");
    EXPECT_NEAR(interp.state().scalars.at("rec"), 0.5, 1e-9);

    expect_contains(interp, "ml_recall([1; 1; 0; 0], [1; 0; 0; 1])", "0.5");
}

TEST(ReplCommandsTest, wave103_ml_mae) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mae(p,t)");

    expect_ok(interp, "mae = ml_mae([1; 2; 3], [1; 3; 3])");
    EXPECT_NEAR(interp.state().scalars.at("mae"), 1.0 / 3.0, 1e-9);

    expect_contains(interp, "ml_mae([1; 2; 3], [1; 3; 3])", "0.333333");
}

TEST(ReplCommandsTest, wave104_ml_hinge) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_hinge(p,t)");

    expect_ok(interp, "hinge = ml_hinge([1.5; -0.5; 0.8], [1; -1; 1])");
    EXPECT_NEAR(interp.state().scalars.at("hinge"), 0.7 / 3.0, 1e-9);

    expect_contains(interp, "ml_hinge([1.5; -0.5; 0.8], [1; -1; 1])", "0.233333");
}

TEST(ReplCommandsTest, wave104_ml_huber) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_huber(p,t)");

    expect_ok(interp, "huber = ml_huber([0; 1; 5], [0; 0; 0])");
    EXPECT_GT(interp.state().scalars.at("huber"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("huber")));

    expect_contains(interp, "ml_huber([0; 1; 5], [0; 0; 0])", "1.666667");
}

TEST(ReplCommandsTest, wave104_numthy_mod_inv) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_mod_inv(a,m)");

    expect_ok(interp, "inv = numthy_mod_inv(3, 7)");
    EXPECT_NEAR(interp.state().scalars.at("inv"), 5.0, 1e-9);

    expect_contains(interp, "numthy_mod_inv(3, 7)", "5");
}

TEST(ReplCommandsTest, wave106_ml_vec_norm) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_vec_norm(v)");

    expect_ok(interp, "vn = ml_vec_norm([3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("vn"), 5.0, 1e-9);

    expect_contains(interp, "ml_vec_norm([3; 4])", "5");
}

TEST(ReplCommandsTest, wave106_numthy_factor_count) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor_count(n)");

    expect_ok(interp, "fc = numthy_factor_count(12)");
    EXPECT_NEAR(interp.state().scalars.at("fc"), 3.0, 1e-9);

    expect_contains(interp, "numthy_factor_count(12)", "3");
}

TEST(ReplCommandsTest, wave106_geo_polygon_perimeter) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_polygon_perimeter(P)");

    expect_ok(interp, "per = geo_polygon_perimeter([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("per"), 16.0, 1e-9);

    expect_contains(interp, "geo_polygon_perimeter([0, 0; 4, 0; 4, 4; 0, 4])", "16");
}

TEST(ReplCommandsTest, wave107_ml_vec_dot) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_vec_dot(a,b)");

    expect_ok(interp, "vd = ml_vec_dot([1; 2], [3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("vd"), 11.0, 1e-9);

    expect_contains(interp, "ml_vec_dot([1; 2], [3; 4])", "11");
}

TEST(ReplCommandsTest, wave107_numthy_primitive_root) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_primitive_root(p)");

    expect_ok(interp, "proot = numthy_primitive_root(7)");
    EXPECT_NEAR(interp.state().scalars.at("proot"), 3.0, 1e-9);

    expect_contains(interp, "numthy_primitive_root(7)", "3");
}

TEST(ReplCommandsTest, wave107_geo_triangle_area) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_triangle_area(x1,y1,x2,y2,x3,y3)");

    expect_ok(interp, "tri = geo_triangle_area(0, 0, 4, 0, 0, 3)");
    EXPECT_NEAR(interp.state().scalars.at("tri"), 6.0, 1e-9);

    expect_contains(interp, "geo_triangle_area(0, 0, 4, 0, 0, 3)", "6");
}

TEST(ReplCommandsTest, wave108_geo_dist_sq2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_sq2d(x1,y1,x2,y2)");

    expect_ok(interp, "dsq = geo_dist_sq2d(0, 0, 3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("dsq"), 25.0, 1e-9);

    expect_contains(interp, "geo_dist_sq2d(0, 0, 3, 4)", "25");
}

TEST(ReplCommandsTest, wave108_geo_vec2d_length) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_vec2d_length(x,y)");

    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-9);

    expect_contains(interp, "geo_vec2d_length(3, 4)", "5");
}

TEST(ReplCommandsTest, wave108_geo_cross2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_cross2d(x1,y1,x2,y2)");

    expect_ok(interp, "cr = geo_cross2d(1, 2, 3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), -2.0, 1e-9);

    expect_contains(interp, "geo_cross2d(1, 2, 3, 4)", "-2");
}

TEST(ReplCommandsTest, wave112_quantum_ket_basis) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_basis(dim,index)");

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    ASSERT_GT(interp.state().matrices.count("kb"), 0u);
    const auto& kb = interp.state().matrices.at("kb");
    EXPECT_NEAR(kb(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(kb(1, 0), 0.0, 1e-9);

    expect_contains(interp, "quantum_ket_basis(2, 0)", "state =");
}

TEST(ReplCommandsTest, wave112_quantum_fock_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_fock_state(n,n_max)");

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    ASSERT_GT(interp.state().matrices.count("fs"), 0u);
    const auto& fs = interp.state().matrices.at("fs");
    EXPECT_NEAR(fs(1, 0), 1.0, 1e-9);

    expect_contains(interp, "quantum_fock_state(1, 3)", "state =");
}

TEST(ReplCommandsTest, wave112_quantum_identity_n) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_identity_n(dim)");

    expect_ok(interp, "I3 = quantum_identity_n(3)");
    ASSERT_GT(interp.state().matrices.count("I3"), 0u);
    const auto& I3 = interp.state().matrices.at("I3");
    EXPECT_EQ(I3.rows(), 3u);
    EXPECT_EQ(I3.cols(), 3u);
    EXPECT_NEAR(I3(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(I3(2, 2), 1.0, 1e-9);

    expect_contains(interp, "quantum_identity_n(3)", "op =");
}

TEST(ReplCommandsTest, wave113_control_is_controllable) {
    Interpreter interp;
    expect_contains(interp, "help", "control_is_controllable(A,B)");

    expect_ok(interp, "A = [0, 1; 0, 0]");
    expect_ok(interp, "B = [0; 1]");
    expect_ok(interp, "ctrl = control_is_controllable(A, B)");
    EXPECT_NEAR(interp.state().scalars.at("ctrl"), 1.0, 1e-9);

    expect_contains(interp, "control_is_controllable([0, 1; 0, 0], [0; 1])", "1");
}

TEST(ReplCommandsTest, wave113_quantum_ket_superposition) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_superposition(amps)");

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
    const auto& sup = interp.state().matrices.at("sup");
    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(sup(0, 0), h, 1e-9);
    EXPECT_NEAR(sup(1, 0), h, 1e-9);

    expect_contains(interp, "quantum_ket_superposition([1; 1])", "state =");
}

TEST(ReplCommandsTest, wave113_numthy_extended_gcd) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_extended_gcd(a,b)");

    expect_ok(interp, "g = numthy_extended_gcd(35, 15)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 5.0, 1e-9);

    expect_contains(interp, "numthy_extended_gcd(35, 15)", "5");
}

TEST(ReplCommandsTest, wave114_mtf_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "mtf_encode_vec(M)");

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, wave114_geo_centroid_x) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_centroid_x(P)");

    expect_ok(interp, "cx = geo_centroid_x([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("cx"), 2.0, 1e-9);

    expect_contains(interp, "geo_centroid_x([0, 0; 4, 0; 4, 4; 0, 4])", "2");
}

TEST(ReplCommandsTest, wave114_quantum_ghz_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ghz_state(n)");

    expect_ok(interp, "ghz = quantum_ghz_state(3)");
    ASSERT_GT(interp.state().matrices.count("ghz"), 0u);
    const auto& ghz = interp.state().matrices.at("ghz");
    EXPECT_EQ(ghz.rows(), 8u);
    EXPECT_EQ(ghz.cols(), 1u);
    const double amp = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(ghz(0, 0), amp, 1e-9);
    EXPECT_NEAR(ghz(7, 0), amp, 1e-9);

    expect_contains(interp, "quantum_ghz_state(3)", "state =");
}

TEST(ReplCommandsTest, wave115_control_is_observable) {
    Interpreter interp;
    expect_contains(interp, "help", "control_is_observable(A,C)");

    expect_ok(interp, "A = [0, 1; 0, 0]");
    expect_ok(interp, "C = [1, 0]");
    expect_ok(interp, "obs = control_is_observable(A, C)");
    EXPECT_NEAR(interp.state().scalars.at("obs"), 1.0, 1e-9);

    expect_contains(interp, "control_is_observable([0, 1; 0, 0], [1, 0])", "1");
}

TEST(ReplCommandsTest, wave115_mtf_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "mtf_decode_vec(M)");

    expect_ok(interp, "B = [10, 12, 15, 20]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
}

TEST(ReplCommandsTest, wave115_numthy_crt) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_crt(r,m)");

    expect_ok(interp, "r = [2; 3; 2]");
    expect_ok(interp, "m = [3; 5; 7]");
    expect_ok(interp, "x = numthy_crt(r, m)");
    EXPECT_NEAR(interp.state().scalars.at("x"), 23.0, 1e-9);

    expect_contains(interp, "numthy_crt([2; 3; 2], [3; 5; 7])", "23");
}

TEST(ReplCommandsTest, wave116_geo_centroid_y) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_centroid_y(P)");

    expect_ok(interp, "cy = geo_centroid_y([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("cy"), 2.0, 1e-9);

    expect_contains(interp, "geo_centroid_y([0, 0; 4, 0; 4, 4; 0, 4])", "2");
}

TEST(ReplCommandsTest, wave116_quantum_w_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_w_state(n)");

    expect_ok(interp, "w = quantum_w_state(3)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 8u);
    EXPECT_EQ(w.cols(), 1u);
    const double amp = 1.0 / std::sqrt(3.0);
    EXPECT_NEAR(w(1, 0), amp, 1e-9);
    EXPECT_NEAR(w(2, 0), amp, 1e-9);
    EXPECT_NEAR(w(4, 0), amp, 1e-9);

    expect_contains(interp, "quantum_w_state(3)", "state =");
}

TEST(ReplCommandsTest, wave116_numthy_divisors_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_divisors_vec(n)");

    expect_ok(interp, "d = numthy_divisors_vec(12)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    const auto& divs = interp.state().matrices.at("d");
    EXPECT_EQ(divs.rows(), 6u);
    EXPECT_EQ(divs.cols(), 1u);
    const double expected[] = {1, 2, 3, 4, 6, 12};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(divs(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, wave117_bwt_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "bwt_encode_vec(M)");

    expect_ok(interp, "B = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "E = bwt_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 7u);
}

TEST(ReplCommandsTest, wave117_bwt_primary_index) {
    Interpreter interp;
    expect_contains(interp, "help", "bwt_primary_index(M)");

    expect_ok(interp, "B = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "pi = bwt_primary_index(B)");
    const double pi = interp.state().scalars.at("pi");
    EXPECT_GE(pi, 0.0);
    EXPECT_LT(pi, 7.0);
}

TEST(ReplCommandsTest, wave117_bwt_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "bwt_decode_vec(L,primary_index)");

    expect_ok(interp, "B = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "E = bwt_encode_vec(B)");
    expect_ok(interp, "pi = bwt_primary_index(B)");
    expect_ok(interp, "R = bwt_decode_vec(E, pi)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 6u);
    const double banana[] = {98, 97, 110, 97, 110, 97};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(restored(i, 0), banana[i], 1e-9);
    }
}

TEST(ReplCommandsTest, wave118_control_impulse_final) {
    Interpreter interp;
    expect_contains(interp, "help", "control_impulse_final(num,den)");

    expect_ok(interp, "imp = control_impulse_final([1], [1, 1])");
    EXPECT_NEAR(interp.state().scalars.at("imp"), std::exp(-10.0), 1e-4);

    expect_contains(interp, "control_impulse_final([1], [1, 1])", "0.00004");
}

TEST(ReplCommandsTest, wave118_combo_multinomial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_multinomial(n,ks)");

    expect_ok(interp, "m = combo_multinomial(6, [2; 2; 2])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 90.0, 1e-9);

    expect_contains(interp, "combo_multinomial(6, [2; 2; 2])", "90");
}

TEST(ReplCommandsTest, wave118_numthy_factor_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor_vec(n)");

    expect_ok(interp, "f = numthy_factor_vec(12)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& factors = interp.state().matrices.at("f");
    EXPECT_EQ(factors.rows(), 3u);
    EXPECT_EQ(factors.cols(), 1u);
    EXPECT_NEAR(factors(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(2, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, wave119_lzw_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "lzw_encode_vec(M)");

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "E = lzw_encode_vec(M)");
    expect_ok(interp, "R = lzw_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 6u);
    const double expected[] = {97, 98, 99, 97, 98, 99};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(restored(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, wave119_lzw_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "lzw_decode_vec(C)");

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "E = lzw_encode_vec(M)");
    expect_ok(interp, "R = lzw_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 6u);
}

TEST(ReplCommandsTest, wave119_quantum_coherent_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_coherent_state(alpha_re,alpha_im,n_max)");

    expect_ok(interp, "s = quantum_coherent_state(1, 0, 20)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& state = interp.state().matrices.at("s");
    EXPECT_EQ(state.rows(), 21u);
    EXPECT_EQ(state.cols(), 1u);
    double norm_sq = 0.0;
    for (size_t i = 0; i < state.rows(); ++i) {
        norm_sq += state(i, 0) * state(i, 0);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-5);

    expect_contains(interp, "quantum_coherent_state(1, 0, 20)", "state =");
}

TEST(ReplCommandsTest, wave119_geo_dist_point_line2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_point_line2d(px,py,a,b,c)");

    expect_ok(interp, "d = geo_dist_point_line2d(0, 0, 1, 1, -1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), std::sqrt(2.0) / 2.0, 1e-5);

    expect_contains(interp, "geo_dist_point_line2d(0, 0, 1, 1, -1)", "0.707");
}

TEST(ReplCommandsTest, wave120_huffman_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "huffman_encode_vec(M)");

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "E = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_GE(interp.state().matrices.at("E").rows(), 1u);
}

TEST(ReplCommandsTest, wave120_huffman_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "huffman_decode_vec(orig_M,E)");

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "E = huffman_encode_vec(M)");
    expect_ok(interp, "R = huffman_decode_vec(M, E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 6u);
    const double expected[] = {97, 98, 99, 97, 97, 98};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(restored(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, wave120_geo_volume_tetrahedron) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_volume_tetrahedron");

    expect_ok(interp, "v = geo_volume_tetrahedron(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("v"), 1.0 / 6.0, 1e-5);

    expect_contains(interp, "geo_volume_tetrahedron(0,0,0,1,0,0,0,1,0,0,0,1)", "0.166667");
}

TEST(ReplCommandsTest, wave121_control_lyap) {
    Interpreter interp;
    expect_contains(interp, "help", "control_lyap(A,Q)");

    expect_ok(interp, "X = control_lyap([-1], [1])");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    const auto& X = interp.state().matrices.at("X");
    EXPECT_EQ(X.rows(), 1u);
    EXPECT_EQ(X.cols(), 1u);
    EXPECT_NEAR(X(0, 0), 0.5, 1e-4);
}

TEST(ReplCommandsTest, wave121_combo_rank_permutation) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_rank_permutation(v)");

    expect_ok(interp, "r0 = combo_rank_permutation([0; 1; 2])");
    EXPECT_NEAR(interp.state().scalars.at("r0"), 0.0, 1e-9);

    expect_ok(interp, "r5 = combo_rank_permutation([2; 1; 0])");
    EXPECT_NEAR(interp.state().scalars.at("r5"), 5.0, 1e-9);

    expect_contains(interp, "combo_rank_permutation([2; 1; 0])", "5");
}

TEST(ReplCommandsTest, wave121_combo_unrank_permutation) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_unrank_permutation(n,rank)");

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);

    expect_contains(interp, "combo_unrank_permutation(3, 5)", "perm =");
}

TEST(ReplCommandsTest, wave122_control_lqr) {
    Interpreter interp;
    expect_contains(interp, "help", "control_lqr(A,B,Q,R)");

    expect_ok(interp, "Q = eye(2)");
    expect_ok(interp, "K = control_lqr([-2, 1; 0, -3], [1; 1], Q, [1])");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    const auto& Klqr = interp.state().matrices.at("K");
    EXPECT_EQ(Klqr.rows(), 1u);
    EXPECT_EQ(Klqr.cols(), 2u);
}

TEST(ReplCommandsTest, wave122_combo_rank_combination) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_rank_combination(v,n)");

    expect_ok(interp, "r0 = combo_rank_combination([0; 1], 4)");
    EXPECT_NEAR(interp.state().scalars.at("r0"), 0.0, 1e-9);

    expect_ok(interp, "r1 = combo_rank_combination([0; 2], 4)");
    EXPECT_NEAR(interp.state().scalars.at("r1"), 1.0, 1e-9);

    expect_contains(interp, "combo_rank_combination([0; 2], 4)", "1");
}

TEST(ReplCommandsTest, wave122_lz77_encode_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "lz77_encode_vec(M)");
    expect_contains(interp, "help", "lz77_decode_vec(T)");

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);
}

TEST(ReplCommandsTest, wave123_control_pidtune_kp) {
    Interpreter interp;
    expect_contains(interp, "help", "control_pidtune_kp(num,den)");

    expect_ok(interp, "kp = control_pidtune_kp([1], [1, 1])");
    EXPECT_GT(interp.state().scalars.at("kp"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("kp"), 1.414, 0.1);

    expect_contains(interp, "control_pidtune_kp([1], [1, 1])", "1.");
}

TEST(ReplCommandsTest, wave123_quantum_bell_state) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_bell_state(index)");

    expect_ok(interp, "s = quantum_bell_state(0)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    const auto& bell = interp.state().matrices.at("s");
    EXPECT_EQ(bell.rows(), 4u);
    EXPECT_NEAR(bell(0, 0), 0.707107, 1e-4);
    EXPECT_NEAR(bell(3, 0), 0.707107, 1e-4);

    expect_contains(interp, "quantum_bell_state(0)", "state =");
}

TEST(ReplCommandsTest, wave123_bzip2_compress_decompress_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "bzip2_compress_vec(M)");
    expect_contains(interp, "help", "bzip2_decompress_vec(C)");

    expect_ok(interp, "M = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(M)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);

    expect_ok(interp, "R = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 11u);
}

TEST(ReplCommandsTest, wave124_control_place) {
    Interpreter interp;
    expect_contains(interp, "help", "control_place(A,B,poles)");

    expect_ok(interp, "K = control_place([0, 1; 0, 0], [0; 1], [-2; -3])");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 2u);
}

TEST(ReplCommandsTest, wave124_diffgeo_principal_curvature_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_principal_curvature_sphere()");

    expect_ok(interp, "k1 = diffgeo_principal_curvature_sphere()");
    EXPECT_NEAR(interp.state().scalars.at("k1"), 1.0, 0.15);

    expect_contains(interp, "diffgeo_principal_curvature_sphere()", "0.999");
}

TEST(ReplCommandsTest, wave124_topo_euler_sphere_surface) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_euler_sphere_surface()");

    expect_ok(interp, "chi = topo_euler_sphere_surface()");
    EXPECT_NEAR(interp.state().scalars.at("chi"), 2.0, 1e-9);

    expect_contains(interp, "topo_euler_sphere_surface()", "2");
}

TEST(ReplCommandsTest, wave125_combo_unrank_combination) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_unrank_combination(n,k,rank)");

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave125_control_pidtune_ki) {
    Interpreter interp;
    expect_contains(interp, "help", "control_pidtune_ki(num,den)");

    expect_ok(interp, "ki = control_pidtune_ki([1], [1, 1])");
    EXPECT_GT(interp.state().scalars.at("ki"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("ki"), 0.141, 0.05);

    expect_contains(interp, "control_pidtune_ki([1], [1, 1])", "0.");
}

TEST(ReplCommandsTest, wave125_control_pidtune_kd) {
    Interpreter interp;
    expect_contains(interp, "help", "control_pidtune_kd(num,den)");

    expect_ok(interp, "kd = control_pidtune_kd([1], [1, 1])");
    EXPECT_GT(interp.state().scalars.at("kd"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("kd"), 0.141, 0.05);

    expect_contains(interp, "control_pidtune_kd([1], [1, 1])", "0.");
}

TEST(ReplCommandsTest, wave126_cplx_power_series_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_power_series_eval(coeffs,zre,zim)");

    expect_ok(interp, "expv = cplx_power_series_eval([1; 1; 0.5; 0.166667], 0.5, 0)");
    EXPECT_NEAR(interp.state().scalars.at("expv"), 1.6487, 0.01);

    expect_contains(interp, "cplx_power_series_eval([1; 1; 0.5; 0.166667], 0.5, 0)", "1.64");
}

TEST(ReplCommandsTest, wave126_cplx_winding_number) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_winding_number(G,z0re,z0im)");

    expect_ok(interp, "wn1 = cplx_winding_number([1, 0; 0, 1; -1, 0; 0, -1], 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wn1"), 1.0, 1e-9);

    expect_ok(interp, "wn0 = cplx_winding_number([1, 0; 0, 1; -1, 0; 0, -1], 2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wn0"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave126_quantum_schrodinger) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schrodinger(H,psi0,t0,t1,n_steps)");

    expect_ok(interp, "traj = quantum_schrodinger([0.5, 0; 0, -0.5], [1; 0], 0, 3.141592653589793, 100)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 101u);
    EXPECT_EQ(interp.state().matrices.at("traj").cols(), 2u);
}

TEST(ReplCommandsTest, wave127_topo_vietoris_rips_betti0) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_vietoris_rips_betti0(D,r,max_dim)");

    expect_ok(interp, "Dtri = [0, 1, 1; 1, 0, 1; 1, 1, 0]");
    expect_ok(interp, "b0t = topo_vietoris_rips_betti0(Dtri, 1.1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("b0t"), 1.0, 1e-9);

    expect_ok(interp, "Dfar = [0, 10; 10, 0]");
    expect_ok(interp, "b0d = topo_vietoris_rips_betti0(Dfar, 1.0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("b0d"), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave127_diffgeo_gaussian_curvature_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_gaussian_curvature_sphere(u,v)");

    expect_ok(interp, "Kg = diffgeo_gaussian_curvature_sphere(0.3, 0.7)");
    EXPECT_NEAR(interp.state().scalars.at("Kg"), 1.0, 0.15);

    expect_contains(interp, "diffgeo_gaussian_curvature_sphere(0.3, 0.7)", "0.999");
}

TEST(ReplCommandsTest, wave127_control_dlyap) {
    Interpreter interp;
    expect_contains(interp, "help", "control_dlyap(A,Q)");

    expect_ok(interp, "Xd = control_dlyap([0.5], [1])");
    ASSERT_GT(interp.state().matrices.count("Xd"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Xd")(0, 0), 4.0 / 3.0, 1e-3);

    expect_contains(interp, "control_dlyap([0.5], [1])", "1.333");
}

TEST(ReplCommandsTest, wave128_lz77_encode_vec_custom) {
    Interpreter interp;
    expect_contains(interp, "help", "lz77_encode_vec(M,window,lookahead)");

    expect_ok(interp, "lzM2 = [120; 121; 122; 120; 121; 122; 120; 121]");
    expect_ok(interp, "T2 = lz77_encode_vec(lzM2, 64, 8)");
    ASSERT_GT(interp.state().matrices.count("T2"), 0u);
    expect_ok(interp, "R2 = lz77_decode_vec(T2)");
    ASSERT_GT(interp.state().matrices.count("R2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R2").rows(), 8u);
}

TEST(ReplCommandsTest, wave128_control_riccati) {
    Interpreter interp;
    expect_contains(interp, "help", "control_riccati(A,B,Q,R)");

    expect_ok(interp, "Q = eye(2)");
    expect_ok(interp, "Xr = control_riccati([-2, 1; 0, -3], [1; 1], Q, [1])");
    ASSERT_GT(interp.state().matrices.count("Xr"), 0u);
    EXPECT_GT(interp.state().matrices.at("Xr")(0, 0), 0.0);
}

TEST(ReplCommandsTest, wave128_control_dare) {
    Interpreter interp;
    expect_contains(interp, "help", "control_dare(A,B,Q,R)");

    expect_ok(interp, "Xdare = control_dare([0.5], [1], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("Xdare"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Xdare")(0, 0), 4.0 / 3.0, 0.25);
}

TEST(ReplCommandsTest, wave129_control_bode_mag_db) {
    Interpreter interp;
    expect_contains(interp, "help", "control_bode_mag_db(num,den,w)");

    expect_ok(interp, "bodeDb = control_bode_mag_db([1], [1, 1], 1)");
    EXPECT_NEAR(interp.state().scalars.at("bodeDb"), -3.01, 0.15);
}

TEST(ReplCommandsTest, wave129_cplx_residue_inv) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_residue_inv(pole_re,pole_im)");

    expect_ok(interp, "res1 = cplx_residue_inv(1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("res1"), 1.0, 0.05);
}

TEST(ReplCommandsTest, wave129_cplx_contour_integral_oneoverz_im) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_contour_integral_oneoverz_im()");

    expect_ok(interp, "imz = cplx_contour_integral_oneoverz_im()");
    EXPECT_NEAR(interp.state().scalars.at("imz"), 2.0 * 3.141592653589793, 0.2);
}

TEST(ReplCommandsTest, wave130_quantum_time_evolution) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_time_evolution(H,t)");

    expect_ok(interp, "U0 = quantum_time_evolution([0, 0.5; 0.5, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("U0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("U0")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("U0")(1, 1), 1.0, 1e-6);
}

TEST(ReplCommandsTest, wave130_topo_betti_curve) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_betti_curve(D,thresholds,max_dim)");

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave130_diffgeo_mean_curvature_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_mean_curvature_sphere(u,v)");

    expect_ok(interp, "Hm = diffgeo_mean_curvature_sphere(0.3, 0.7)");
    EXPECT_NEAR(interp.state().scalars.at("Hm"), 1.0, 0.15);
}

TEST(ReplCommandsTest, wave131_control_bode_phase) {
    Interpreter interp;
    expect_contains(interp, "help", "control_bode_phase(num,den,w)");

    expect_ok(interp, "bodePh = control_bode_phase([1], [1, 1], 1)");
    EXPECT_NEAR(interp.state().scalars.at("bodePh"), -45.0, 2.0);
}

TEST(ReplCommandsTest, wave131_control_phase_margin) {
    Interpreter interp;
    expect_contains(interp, "help", "control_phase_margin(num,den)");

    expect_ok(interp, "pm = control_phase_margin([1], [1, 1])");
    EXPECT_NEAR(interp.state().scalars.at("pm"), 180.0, 5.0);
}

TEST(ReplCommandsTest, wave131_combo_derangements) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_derangements(n)");

    expect_ok(interp, "d3 = combo_derangements(3)");
    ASSERT_GT(interp.state().matrices.count("d3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("d3").cols(), 3u);
}

TEST(ReplCommandsTest, wave132_cplx_line_integral_one) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_line_integral_one()");

    expect_ok(interp, "li1 = cplx_line_integral_one()");
    EXPECT_NEAR(interp.state().scalars.at("li1"), 1.0, 0.05);
}

TEST(ReplCommandsTest, wave132_quantum_density_matrix) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_density_matrix(psi)");

    expect_ok(interp, "rho = quantum_density_matrix([1; 0])");
    ASSERT_GT(interp.state().matrices.count("rho"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rho")(0, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave132_topo_bottleneck_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_bottleneck_distance(dgm1,dgm2,dim)");

    expect_ok(interp, "dgm1 = [0, 0, 2; 1, 1, 3]");
    expect_ok(interp, "dgm2 = [0, 0.2, 2.2; 1, 1.2, 3.2]");
    expect_ok(interp, "bn = topo_bottleneck_distance(dgm1, dgm2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("bn"), 0.2, 0.05);
}

TEST(ReplCommandsTest, wave133_diffgeo_christoffel_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_christoffel_sphere(k,i,j,u,v)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "G011 = diffgeo_christoffel_sphere(0, 1, 1, pi/4, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("G011"), -0.5, 0.05);
}

TEST(ReplCommandsTest, wave133_finance_bond_price_fv) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_price(c,y,n,fv)");

    expect_ok(interp, "bp100 = finance_bond_price(0.05, 0.05, 10, 100)");
    EXPECT_NEAR(interp.state().scalars.at("bp100"), 100.0, 0.5);
}

TEST(ReplCommandsTest, wave133_control_gain_margin) {
    Interpreter interp;
    expect_contains(interp, "help", "control_gain_margin(num,den)");

    expect_ok(interp, "gm = control_gain_margin([1], [1, 1])");
    expect_contains(interp, "control_gain_margin([1], [1, 1])", "inf");
}

TEST(ReplCommandsTest, wave134_finance_compound_cpp) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_compound(principal,rate,n_periods,compounds_per_period)");

    expect_ok(interp, "fv4 = finance_compound(100, 0.1, 1, 4)");
    EXPECT_NEAR(interp.state().scalars.at("fv4"), 110.381, 0.05);
}

TEST(ReplCommandsTest, wave134_combo_all_permutations) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_permutations(n)");

    expect_ok(interp, "p3 = combo_all_permutations(3)");
    ASSERT_GT(interp.state().matrices.count("p3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p3").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("p3").cols(), 3u);
}

TEST(ReplCommandsTest, wave134_control_bode) {
    Interpreter interp;
    expect_contains(interp, "help", "control_bode(num,den,w)");

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    ASSERT_GT(interp.state().matrices.count("bode"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bode")(0, 0), -3.01, 0.1);
    EXPECT_NEAR(interp.state().matrices.at("bode")(0, 1), -45.0, 2.0);
}

TEST(ReplCommandsTest, wave260_control_step_response) {
    Interpreter interp;
    expect_contains(interp, "help", "control_step_response(num,den[,t_end[,n_pts]])");

    // 1/(s+1): y(t) = 1 - e^{-t}; at t=5 should be near 1
    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_NEAR(step(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(step(step.rows() - 1, 0), 5.0, 1e-9);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);

    expect_ok(interp, "step_def = control_step_response([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("step_def"), 0u);
    EXPECT_EQ(interp.state().matrices.at("step_def").cols(), 2u);
    EXPECT_GT(interp.state().matrices.at("step_def").rows(), 0u);
}

TEST(ReplCommandsTest, wave260_control_impulse_response) {
    Interpreter interp;
    expect_contains(interp, "help", "control_impulse_response(num,den[,t_end[,n_pts]])");

    // 1/(s+1): impulse y(t) = e^{-t}; at t=5 should be near e^{-5}
    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    const auto& imp = interp.state().matrices.at("imp");
    EXPECT_EQ(imp.cols(), 2u);
    EXPECT_EQ(imp.rows(), 200u);
    EXPECT_NEAR(imp(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(imp(imp.rows() - 1, 0), 5.0, 1e-9);
    EXPECT_NEAR(imp(imp.rows() - 1, 1), std::exp(-5.0), 0.05);
}

TEST(ReplCommandsTest, wave263_control_kalman) {
    Interpreter interp;
    expect_contains(interp, "help", "control_kalman_predict(x,P,A,Q)");
    expect_contains(interp, "help", "control_kalman_update(x,P,z,H,R)");

    // Scalar closed-form predict+update from test_control.cpp (q=0.05, r=0.5, z=2).
    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-6);

    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    const double K = 1.05 / 1.55;
    EXPECT_NEAR(interp.state().matrices.at("Pu")(0, 0), (1.0 - K) * 1.05, 1e-6);
}

TEST(ReplCommandsTest, wave135_quantum_op_apply) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_op_apply(op,psi)");

    expect_ok(interp, "psi_h = quantum_op_apply(quantum_hadamard_gate(), [1; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_h"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("psi_h")(0, 0), 0.707, 0.05);
    EXPECT_NEAR(interp.state().matrices.at("psi_h")(1, 0), 0.707, 0.05);
}

TEST(ReplCommandsTest, wave135_topo_persistence_diagram) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_persistence_diagram(S,births)");

    expect_ok(interp, "S = [0, -1; 1, -1; 0, 1]");
    expect_ok(interp, "births = [0; 0; 1]");
    expect_ok(interp, "pd = topo_persistence_diagram(S, births)");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
    EXPECT_GT(interp.state().matrices.at("pd").rows(), 0u);
}

TEST(ReplCommandsTest, wave135_diffgeo_geodesic_euclidean) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end)");

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, wave136_compress_bits_to_bytes) {
    Interpreter interp;
    expect_contains(interp, "help", "compress_bits_to_bytes(bits_vec)");

    expect_ok(interp, "bits = [1;0;1;0;1;0;1;1;1;1;0;0;1;0;1;1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);
}

TEST(ReplCommandsTest, wave136_cplx_blaschke_product) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_blaschke_product(zre,zim,zeros)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "zeros = [0.3, 0; 0, 0.4]");
    expect_ok(interp, "bp = cplx_blaschke_product(cos(pi/7), sin(pi/7), zeros)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 1.0, 0.05);
}

TEST(ReplCommandsTest, wave136_graph_diameter) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_diameter(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "diam = graph_diameter(Achain)");
    EXPECT_NEAR(interp.state().scalars.at("diam"), 3.0, 1e-9);
}

TEST(ReplCommandsTest, wave255_graph_spectral) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_katz_centrality(A)");
    expect_contains(interp, "help", "graph_algebraic_connectivity(A)");

    expect_ok(interp, "A = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "k = graph_katz_centrality(A)");
    ASSERT_GT(interp.state().matrices.count("k"), 0u);
    const auto& k = interp.state().matrices.at("k");
    EXPECT_EQ(k.rows(), 3u);
    EXPECT_EQ(k.cols(), 1u);
    EXPECT_TRUE(std::isfinite(k(0, 0)));
    EXPECT_TRUE(std::isfinite(k(1, 0)));
    EXPECT_TRUE(std::isfinite(k(2, 0)));

    expect_ok(interp, "ac = graph_algebraic_connectivity(A)");
    EXPECT_GT(interp.state().scalars.at("ac"), 0.0);

    expect_ok(interp, "L = graph_laplacian(A)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    // C++ adjacency_spectrum is power-iteration spectral radius (1×1), not full eig.
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    ASSERT_GT(interp.state().matrices.count("spec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("spec").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("spec")(0, 0)));
    EXPECT_GT(interp.state().matrices.at("spec")(0, 0), 0.0);
}

TEST(ReplCommandsTest, wave258_graph_dijkstra_bellman_ford) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_dijkstra(A,source)");
    expect_contains(interp, "help", "graph_bellman_ford(A,source)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "D = graph_dijkstra(A, 0)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& D = interp.state().matrices.at("D");
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 2u);
    EXPECT_NEAR(D(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(D(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(D(2, 0), 3.0, 1e-9);
    EXPECT_NEAR(D(0, 1), -1.0, 1e-9);
    EXPECT_NEAR(D(1, 1), 0.0, 1e-9);
    EXPECT_NEAR(D(2, 1), 1.0, 1e-9);

    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    const auto& B = interp.state().matrices.at("B");
    EXPECT_EQ(B.rows(), 3u);
    EXPECT_EQ(B.cols(), 2u);
    EXPECT_NEAR(B(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(B(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(B(2, 0), 3.0, 1e-9);
    EXPECT_NEAR(B(0, 1), -1.0, 1e-9);
    EXPECT_NEAR(B(1, 1), 0.0, 1e-9);
    EXPECT_NEAR(B(2, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave256_graph_structure) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_normalised_laplacian(A)");
    expect_contains(interp, "help", "graph_modularity(A,C)");
    expect_contains(interp, "help", "graph_eccentricity(A)");
    expect_contains(interp, "help", "graph_is_strongly_connected(A)");

    expect_ok(interp, "A = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    ASSERT_GT(interp.state().matrices.count("Ln"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Ln").cols(), 3u);

    expect_ok(interp, "C = [0, 1, -1; 2, -1, -1]");
    expect_ok(interp, "Q = graph_modularity(A, C)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("Q")));

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    ASSERT_GT(interp.state().matrices.count("ecc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("ecc").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("ecc")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ecc")(1, 0), 2.0, 1e-9);

    expect_ok(interp, "Adisc = [0, 1, 0, 0; 1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc2 = graph_eccentricity(Adisc)");
    EXPECT_NEAR(interp.state().matrices.at("ecc2")(0, 0), -1.0, 1e-9);

    expect_ok(interp, "Asc = [0, 1, 0; 0, 0, 1; 1, 0, 0]");
    expect_ok(interp, "sc = graph_is_strongly_connected(Asc)");
    EXPECT_NEAR(interp.state().scalars.at("sc"), 1.0, 1e-9);
    expect_ok(interp, "Aweak = [0, 1, 0; 0, 0, 1; 0, 0, 0]");
    expect_ok(interp, "nsc = graph_is_strongly_connected(Aweak)");
    EXPECT_NEAR(interp.state().scalars.at("nsc"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave137_compress_bytes_to_bits) {
    Interpreter interp;
    expect_contains(interp, "help", "compress_bytes_to_bits(bytes_vec)");

    expect_ok(interp, "bytes = [171; 205]");
    expect_ok(interp, "bits = compress_bytes_to_bits(bytes)");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits").rows(), 16u);
    expect_ok(interp, "bytes2 = compress_bits_to_bytes(bits)");
    EXPECT_EQ(interp.state().matrices.at("bytes2").rows(), 2u);
}

TEST(ReplCommandsTest, wave137_graph_radius) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_radius(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "rad = graph_radius(Achain)");
    EXPECT_NEAR(interp.state().scalars.at("rad"), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave137_combo_all_subsets) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_subsets(n)");

    expect_ok(interp, "subs = combo_all_subsets(3)");
    ASSERT_GT(interp.state().matrices.count("subs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("subs").rows(), 8u);
    EXPECT_EQ(interp.state().matrices.at("subs").cols(), 3u);
}

TEST(ReplCommandsTest, wave138_control_margins) {
    Interpreter interp;
    expect_contains(interp, "help", "control_margins(num,den)");

    expect_ok(interp, "marg = control_margins([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("marg"), 0u);
    expect_contains(interp, "control_margins([1], [1, 1])", "inf");
}

TEST(ReplCommandsTest, wave138_topo_wasserstein_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_wasserstein_distance(dgm1,dgm2,dim)");

    expect_ok(interp, "dgm = [0, 0, 2; 1, 1, 3]");
    expect_ok(interp, "ws = topo_wasserstein_distance(dgm, dgm, 0)");
    EXPECT_NEAR(interp.state().scalars.at("ws"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave138_diffgeo_ricci_scalar_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_ricci_scalar_sphere(u,v)");

    expect_ok(interp, "R = diffgeo_ricci_scalar_sphere(0.3, 1.2)");
    EXPECT_NEAR(interp.state().scalars.at("R"), 2.0, 0.05);
}

TEST(ReplCommandsTest, wave139_quantum_schrodinger_final) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schrodinger_final(H,psi0,t0,t1,n_steps)");

    expect_ok(interp,
              "psi = quantum_schrodinger_final([0.5, 0; 0, -0.5], [1; 0], 0, 3.141592653589793, 100)");
    ASSERT_GT(interp.state().matrices.count("psi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psi").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("psi").cols(), 1u);
}

TEST(ReplCommandsTest, wave139_graph_betweenness) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_betweenness(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
}

TEST(ReplCommandsTest, wave139_imcrop) {
    Interpreter interp;
    expect_contains(interp, "help", "imcrop(M,r0,c0,r1,c1)");

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    ASSERT_GT(interp.state().matrices.count("crop"), 0u);
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("crop").cols(), 4u);
}

TEST(ReplCommandsTest, wave140_medfilt2) {
    Interpreter interp;
    expect_contains(interp, "help", "medfilt2(M,k)");

    expect_ok(interp,
              "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave140_bilateral) {
    Interpreter interp;
    expect_contains(interp, "help", "bilateral(M,sigma_s,sigma_r)");

    expect_ok(interp, "M = ones(5, 5)");
    expect_ok(interp, "B = bilateral(M, 1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 5u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("B")(2, 2)));
}

TEST(ReplCommandsTest, wave140_canny) {
    Interpreter interp;
    expect_contains(interp, "help", "canny(M,low,high)");

    expect_ok(interp,
              "M = [0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; "
              "0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; "
              "0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; "
              "0,0,0,0,0,1,1,1,1,1]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 10u);
    EXPECT_EQ(interp.state().matrices.at("E").cols(), 10u);
}

TEST(ReplCommandsTest, wave141_combo_all_compositions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_compositions(n)");

    expect_ok(interp, "comps = combo_all_compositions(3)");
    ASSERT_GT(interp.state().matrices.count("comps"), 0u);
    EXPECT_EQ(interp.state().matrices.at("comps").rows(), 4u);
}

TEST(ReplCommandsTest, wave141_combo_all_partitions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_partitions(n)");

    expect_ok(interp, "parts = combo_all_partitions(4)");
    ASSERT_GT(interp.state().matrices.count("parts"), 0u);
    EXPECT_EQ(interp.state().matrices.at("parts").rows(), 5u);
}

TEST(ReplCommandsTest, wave141_graph_closeness) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_closeness(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    EXPECT_GT(interp.state().matrices.at("cc")(2, 0), interp.state().matrices.at("cc")(0, 0));
}

TEST(ReplCommandsTest, wave142_graph_degree_centrality) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_degree_centrality(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("dc")(0, 0), 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("dc")(3, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave142_diffgeo_einstein_scalar_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_einstein_scalar_sphere(u,v)");

    expect_ok(interp, "E = diffgeo_einstein_scalar_sphere(1.047197551196598, 0.523598775598299)");
    EXPECT_NEAR(interp.state().scalars.at("E"), 0.0, 0.05);
}

TEST(ReplCommandsTest, wave142_cplx_joukowski_inv) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_joukowski_inv(re,im)");

    expect_ok(interp, "zmag = cplx_joukowski_inv(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("zmag"), std::sqrt(5.0), 1e-9);
}

TEST(ReplCommandsTest, wave143_graph_max_flow) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_max_flow(A,source,sink)");

    expect_ok(interp, "Aflow = [0, 3, 2, 0; 0, 0, 0, 2; 0, 0, 0, 3; 0, 0, 0, 0]");
    expect_ok(interp, "mf = graph_max_flow(Aflow, 0, 3)");
    EXPECT_NEAR(interp.state().scalars.at("mf"), 4.0, 1e-9);
    expect_contains(interp, "graph_max_flow(Aflow, 0, 3)", "4");
}

TEST(ReplCommandsTest, wave143_diffgeo_surface_normal_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_surface_normal_sphere(u,v)");

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    const auto& N = interp.state().matrices.at("N");
    double n_norm = 0.0;
    for (size_t i = 0; i < N.rows(); ++i) {
        n_norm += N(i, 0) * N(i, 0);
    }
    EXPECT_NEAR(std::sqrt(n_norm), 1.0, 0.05);
}

TEST(ReplCommandsTest, wave143_quantum_commutator) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_commutator(A,B)");

    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "Z = quantum_pauli_z()");
    expect_ok(interp, "C = quantum_commutator(X, Z)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    double comm_norm = 0.0;
    const auto& C = interp.state().matrices.at("C");
    for (size_t i = 0; i < C.rows(); ++i) {
        for (size_t j = 0; j < C.cols(); ++j) {
            comm_norm += C(i, j) * C(i, j);
        }
    }
    EXPECT_GT(std::sqrt(comm_norm), 0.0);
}

TEST(ReplCommandsTest, wave144_stats_correlation) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_correlation(x,y)");

    expect_ok(interp, "r = stats_correlation([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])");
    EXPECT_NEAR(interp.state().scalars.at("r"), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave144_signal_moving_average) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_moving_average(x,window)");

    expect_ok(interp, "ma = signal_moving_average([5; 5; 5; 5], 3)");
    ASSERT_GT(interp.state().matrices.count("ma"), 0u);
    for (size_t i = 0; i < interp.state().matrices.at("ma").rows(); ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ma")(i, 0), 5.0, 1e-9);
    }
}

TEST(ReplCommandsTest, wave248_signal_upsample_downsample) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_upsample(x,n)");
    expect_contains(interp, "help", "signal_downsample(x,n)");

    expect_ok(interp, "up = signal_upsample([1; 2], 3)");
    ASSERT_GT(interp.state().matrices.count("up"), 0u);
    EXPECT_EQ(interp.state().matrices.at("up").rows(), 6u);
    EXPECT_NEAR(interp.state().matrices.at("up")(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("up")(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("up")(3, 0), 2.0, 1e-12);

    expect_ok(interp, "dn = signal_downsample([1; 2; 3; 4; 5; 6], 2)");
    ASSERT_GT(interp.state().matrices.count("dn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dn").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("dn")(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("dn")(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("dn")(2, 0), 5.0, 1e-12);
}

TEST(ReplCommandsTest, wave249_signal_resample_decimate_interpolate) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_decimate(x,q)");
    expect_contains(interp, "help", "signal_interpolate(x,p)");
    expect_contains(interp, "help", "signal_resample(x,p,q)");

    expect_ok(interp, "dec = signal_decimate([1; 2; 3; 4; 5; 6], 2)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 3u);

    expect_ok(interp, "itp = signal_interpolate([1; 2], 2)");
    ASSERT_GT(interp.state().matrices.count("itp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("itp").rows(), 4u);

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);
}

TEST(ReplCommandsTest, wave250_signal_coherence) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_coherence(x,y,fs,nperseg)");

    expect_ok(interp,
              "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    const auto& c = interp.state().matrices.at("c");
    EXPECT_EQ(c.cols(), 2u);
    EXPECT_GT(c.rows(), 0u);
    EXPECT_NEAR(c(0, 0), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(c(0, 1)));
}

TEST(ReplCommandsTest, wave251_signal_filter) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_filter(b,a,x)");

    expect_ok(interp, "b = [1, -1]");
    expect_ok(interp, "a = [1, -0.5]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_filter(b, a, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    const auto& y = interp.state().matrices.at("y");
    EXPECT_EQ(y.cols(), 1u);
    EXPECT_EQ(y.rows(), 5u);
    EXPECT_NEAR(y(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(y(1, 0), 1.5, 1e-12);
    EXPECT_NEAR(y(2, 0), 1.75, 1e-12);
    EXPECT_NEAR(y(3, 0), 1.875, 1e-12);
    EXPECT_NEAR(y(4, 0), 1.9375, 1e-12);
}

TEST(ReplCommandsTest, wave252_signal_sosfilt) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_sosfilt(sos,x)");

    // Same DF1 section as SignalSosfiltTest.normalizes_a0_per_section (scaled a0=2).
    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    const auto& y = interp.state().matrices.at("y");
    EXPECT_EQ(y.cols(), 1u);
    EXPECT_EQ(y.rows(), 5u);
    EXPECT_NEAR(y(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(y(1, 0), 1.5, 1e-12);
    EXPECT_NEAR(y(2, 0), 1.75, 1e-12);
    EXPECT_NEAR(y(3, 0), 1.875, 1e-12);
    EXPECT_NEAR(y(4, 0), 1.9375, 1e-12);
}

TEST(ReplCommandsTest, wave253_signal_conv2) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_conv2(A,K)");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(C(2, 2), 4.0, 1e-12);
}

TEST(ReplCommandsTest, wave254_signal_deconv) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_deconv(y,b)");

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    ASSERT_EQ(x.rows(), 3u);
    ASSERT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);
}

TEST(ReplCommandsTest, wave252_signal_savgol) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_savgol(x,window_length,polyorder)");

    expect_ok(interp, "x = [2; -1; 3; 0.5; 4; -2; 1.5]");
    expect_ok(interp, "sg = signal_savgol(x, 5, 2)");
    ASSERT_GT(interp.state().matrices.count("sg"), 0u);
    const auto& sg = interp.state().matrices.at("sg");
    EXPECT_EQ(sg.cols(), 1u);
    EXPECT_EQ(sg.rows(), 7u);
    EXPECT_NEAR(sg(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(sg(1, 0), -1.0, 1e-12);
    EXPECT_NEAR(sg(2, 0), 27.0 / 35.0, 1e-12);
}

TEST(ReplCommandsTest, wave253_signal_median_filter) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_median_filter(x,window_length)");

    // Hand-computed: x={5,1,3,2,4}, w=3 -> {5,3,2,3,4} (edges unfiltered).
    expect_ok(interp, "x = [5; 1; 3; 2; 4]");
    expect_ok(interp, "mf = signal_median_filter(x, 3)");
    ASSERT_GT(interp.state().matrices.count("mf"), 0u);
    const auto& mf = interp.state().matrices.at("mf");
    EXPECT_EQ(mf.cols(), 1u);
    EXPECT_EQ(mf.rows(), 5u);
    EXPECT_NEAR(mf(0, 0), 5.0, 1e-12);
    EXPECT_NEAR(mf(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(mf(2, 0), 2.0, 1e-12);
    EXPECT_NEAR(mf(3, 0), 3.0, 1e-12);
    EXPECT_NEAR(mf(4, 0), 4.0, 1e-12);
}

TEST(ReplCommandsTest, wave251_signal_cheby1) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_cheby1(order,rp_db,cutoff,fs[,type])");

    // scipy.signal.cheby1(2, 1.0, 0.25, fs=2.0, btype='low')
    expect_ok(interp, "ba = signal_cheby1(2, 1.0, 0.25, 2.0)");
    ASSERT_GT(interp.state().matrices.count("ba"), 0u);
    const auto& ba = interp.state().matrices.at("ba");
    EXPECT_EQ(ba.rows(), 2u);
    EXPECT_EQ(ba.cols(), 3u);
    EXPECT_NEAR(ba(0, 0), 0.10255744, 1e-6);
    EXPECT_NEAR(ba(1, 0), 1.0, 1e-12);

    expect_ok(interp, "bah = signal_cheby1(2, 1.0, 0.25, 2.0, 1)");
    ASSERT_GT(interp.state().matrices.count("bah"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bah").rows(), 2u);
}

TEST(ReplCommandsTest, wave252_signal_firwin) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_firwin(n_taps,cutoff[,window])");

    expect_ok(interp, "b = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    const auto& b = interp.state().matrices.at("b");
    EXPECT_EQ(b.rows(), 11u);
    EXPECT_EQ(b.cols(), 1u);
    double sum = 0.0;
    for (size_t i = 0; i < b.rows(); ++i) {
        sum += b(i, 0);
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("bh").cols(), 1u);
}

TEST(ReplCommandsTest, wave253_signal_xcorr_xcov_autocorr) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_xcorr(a,b,max_lag)");
    expect_contains(interp, "help", "signal_xcov(a,b,max_lag)");
    expect_contains(interp, "help", "signal_autocorr(x,max_lag)");

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    const auto& cv = interp.state().matrices.at("cv");
    EXPECT_EQ(cv.rows(), 5u);
    EXPECT_EQ(cv.cols(), 1u);
    EXPECT_TRUE(std::isfinite(cv(2, 0)));
}

TEST(ReplCommandsTest, wave254_signal_lms) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_lms(x,d,filter_length,mu)");
    expect_contains(interp, "help", "signal_lms_weights(x,d,filter_length,mu)");

    // mu=0 keeps zero weights; y[n]=0 and e[n]=d[n].
    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);
    EXPECT_NEAR(ye(2, 1), 1.0, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);
}

TEST(ReplCommandsTest, wave254_signal_czt_and_zoom) {
    Interpreter interp;
    // Distinct needles avoid signal_czt / signal_czt_zoom substring collisions.
    expect_contains(interp, "help", "signal_czt(x,m,w_re,w_im,a_re,a_im)");
    expect_contains(interp, "help", "signal_czt_zoom(x,f_start,f_stop,m,fs)");

    // DFT contour: m=N, w=exp(-2*pi*i/N)=(0,-1), a=1 -> matches length-4 DFT DC bin.
    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "Z = signal_czt(x, 4, 0, -1, 1, 0)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 4u);
    EXPECT_EQ(Z.cols(), 2u);
    EXPECT_NEAR(Z(0, 0), 10.0, 1e-9);
    EXPECT_NEAR(Z(0, 1), 0.0, 1e-9);

    expect_ok(interp, "zoom = signal_czt_zoom(x, 0.0, 0.5, 8, 4.0)");
    ASSERT_GT(interp.state().matrices.count("zoom"), 0u);
    const auto& zoom = interp.state().matrices.at("zoom");
    EXPECT_EQ(zoom.rows(), 8u);
    EXPECT_EQ(zoom.cols(), 2u);
    EXPECT_TRUE(std::isfinite(zoom(0, 0)));
    EXPECT_TRUE(std::isfinite(zoom(0, 1)));
}

TEST(ReplCommandsTest, wave144_geo_delaunay_2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_delaunay_2d(P)");

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T3").rows(), 1u);

    expect_ok(interp, "Tsq = geo_delaunay_2d([0, 0; 1, 0; 0, 1; 1, 1])");
    ASSERT_GT(interp.state().matrices.count("Tsq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Tsq").rows(), 2u);
}

TEST(ReplCommandsTest, wave145_fft_rfft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_rfft(x)");

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-6);
}

TEST(ReplCommandsTest, wave145_graph_is_bipartite) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_bipartite(A)");

    expect_ok(interp, "bp = graph_is_bipartite([0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 1.0, 1e-9);

    expect_ok(interp, "bt = graph_is_bipartite([0, 1, 1; 1, 0, 1; 1, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("bt"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave145_poly_deriv) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_deriv(coeffs)");

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pd").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pd")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pd")(1, 0), 6.0, 1e-9);
}

TEST(ReplCommandsTest, wave146_poly_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_eval(coeffs,x)");

    expect_ok(interp, "pe = poly_eval([1; -1; 1], 2)");
    EXPECT_NEAR(interp.state().scalars.at("pe"), 3.0, 1e-9);

    expect_contains(interp, "poly_eval([1; -1; 1], 2)", "3");
}

TEST(ReplCommandsTest, wave146_graph_is_dag) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_dag(A)");

    expect_ok(interp, "dag = graph_is_dag([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().scalars.at("dag"), 1.0, 1e-9);

    expect_ok(interp, "cyc = graph_is_dag([0, 1, 0; 0, 0, 1; 1, 0, 0])");
    EXPECT_NEAR(interp.state().scalars.at("cyc"), 0.0, 1e-9);

    expect_contains(interp, "graph_is_dag([0, 1, 0; 0, 0, 1; 0, 0, 0])", "1");
}

TEST(ReplCommandsTest, wave146_stats_mean) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mean(x)");

    expect_ok(interp, "m = stats_mean([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 3.0, 1e-9);

    expect_contains(interp, "stats_mean([1; 2; 3; 4; 5])", "3");
}

TEST(ReplCommandsTest, wave147_fft_irfft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_irfft(spectrum,n)");

    expect_ok(interp, "S = fft_rfft([1; 2; 3; 4])");
    expect_ok(interp, "x = fft_irfft(S, 4)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(3, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, wave147_signal_convolve) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_convolve(a,b)");

    expect_ok(interp, "c = signal_convolve([1; 2; 3], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    const auto& conv = interp.state().matrices.at("c");
    EXPECT_EQ(conv.rows(), 4u);
    EXPECT_NEAR(conv(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(conv(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(conv(2, 0), 5.0, 1e-9);
    EXPECT_NEAR(conv(3, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, wave147_graph_floyd_warshall) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_floyd_warshall(A)");

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 2), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave148_poly_integ) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_integ(coeffs,c)");

    expect_ok(interp, "pi = poly_integ([3], 0)");
    ASSERT_GT(interp.state().matrices.count("pi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pi").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pi")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pi")(1, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, wave148_stats_spearman) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_spearman(x,y)");

    expect_ok(interp, "sp = stats_spearman([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])");
    EXPECT_NEAR(interp.state().scalars.at("sp"), 1.0, 1e-9);

    expect_contains(interp, "stats_spearman([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])", "1");
}

TEST(ReplCommandsTest, wave148_signal_hamming) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_hamming(n)");

    expect_ok(interp, "w = signal_hamming(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("w")(3, 0), 1.0, 0.1);
}

TEST(ReplCommandsTest, wave149_stats_median) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_median(x)");

    expect_ok(interp, "med = stats_median([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("med"), 3.0, 1e-9);

    expect_contains(interp, "stats_median([1; 2; 3; 4; 5])", "3");
}

TEST(ReplCommandsTest, wave149_graph_is_connected) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_connected(A)");

    expect_ok(interp, "cn = graph_is_connected([0, 1, 0; 1, 0, 1; 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("cn"), 1.0, 1e-9);

    expect_ok(interp, "dc = graph_is_connected([0, 1, 0; 1, 0, 0; 0, 0, 0])");
    EXPECT_NEAR(interp.state().scalars.at("dc"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave149_signal_hanning) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_hanning(n)");

    expect_ok(interp, "w = signal_hanning(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("w")(3, 0), 1.0, 0.1);
}

TEST(ReplCommandsTest, wave150_fft_dct2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dct2(x)");

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d").rows(), 4u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("d")(0, 0)));
}

TEST(ReplCommandsTest, wave150_poly_add) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_add(a,b)");

    expect_ok(interp, "s = poly_add([1; 2], [3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("s")(0, 0), 4.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s")(1, 0), 6.0, 1e-9);
}

TEST(ReplCommandsTest, wave150_quantum_tensor_product) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_tensor_product(A,B)");

    expect_ok(interp, "I2 = quantum_identity_n(2)");
    expect_ok(interp, "X = quantum_pauli_x()");
    expect_ok(interp, "tp = quantum_tensor_product(I2, X)");
    ASSERT_GT(interp.state().matrices.count("tp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("tp").cols(), 4u);
}

TEST(ReplCommandsTest, wave151_stats_kendall) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_kendall(x,y)");

    expect_ok(interp, "kt = stats_kendall([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])");
    EXPECT_NEAR(interp.state().scalars.at("kt"), 1.0, 1e-9);

    expect_contains(interp, "stats_kendall([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])", "1");
}

TEST(ReplCommandsTest, wave151_graph_mst_kruskal) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_mst_kruskal(A)");

    expect_ok(interp, "mst = graph_mst_kruskal([0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0])");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mst").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("mst").cols(), 3u);
    double total_w = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        total_w += interp.state().matrices.at("mst")(i, 2);
    }
    EXPECT_NEAR(total_w, 6.0, 1e-9);
}

TEST(ReplCommandsTest, wave151_signal_correlate) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_correlate(a,b)");

    expect_ok(interp, "c = signal_correlate([1; 2; 3], [1; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_EQ(interp.state().matrices.at("c").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("c")(2, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave152_stats_stddev) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_stddev(x)");

    expect_ok(interp, "sd = stats_stddev([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("sd"), std::sqrt(2.5), 1e-9);

    expect_contains(interp, "stats_stddev([1; 2; 3; 4; 5])", "1.58114");
}

TEST(ReplCommandsTest, wave152_fft_idct2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_idct2(x)");

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "x = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("x")(3, 0), 4.0, 1e-5);
}

TEST(ReplCommandsTest, wave152_poly_mul) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_mul(a,b)");

    expect_ok(interp, "p = poly_mul([1; 2], [3; 4])");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("p")(1, 0), 10.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("p")(2, 0), 8.0, 1e-9);
}

TEST(ReplCommandsTest, wave153_graph_mst_prim) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_mst_prim(A)");

    expect_ok(interp, "mst = graph_mst_prim([0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0])");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mst").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("mst").cols(), 3u);
    double total_w = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        total_w += interp.state().matrices.at("mst")(i, 2);
    }
    EXPECT_NEAR(total_w, 6.0, 1e-9);
}

TEST(ReplCommandsTest, wave153_signal_blackman) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_blackman(n)");

    expect_ok(interp, "w = signal_blackman(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("w")(3, 0)));
}

TEST(ReplCommandsTest, wave153_stats_skewness) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_skewness(x)");

    expect_ok(interp, "sk = stats_skewness([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("sk"), 0.0, 1e-9);

    expect_contains(interp, "stats_skewness([1; 2; 3; 4; 5])", "0");
}

TEST(ReplCommandsTest, wave154_poly_sub) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_sub(a,b)");

    expect_ok(interp, "d = poly_sub([5; 3], [2; 1])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("d")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave154_fft_dst2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dst2(x)");

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s").rows(), 4u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("s")(0, 0)));
}

TEST(ReplCommandsTest, wave154_prob_norm_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_norm_cdf(x,mu,sigma)");

    expect_ok(interp, "p = prob_norm_cdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("p"), 0.5, 1e-9);

    expect_contains(interp, "prob_norm_cdf(0, 0, 1)", "0.5");
}

TEST(ReplCommandsTest, wave155_stats_kurtosis) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_kurtosis(x)");

    expect_ok(interp, "ku = stats_kurtosis([1; 2; 3; 4; 5])");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ku")));
}

TEST(ReplCommandsTest, wave155_prob_norm_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_norm_pdf(x,mu,sigma)");

    expect_ok(interp, "d = prob_norm_pdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 0.3989422804014327, 1e-6);

    expect_contains(interp, "prob_norm_pdf(0, 0, 1)", "0.398942");
}

TEST(ReplCommandsTest, wave155_signal_parzen) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_parzen(n)");

    expect_ok(interp, "w = signal_parzen(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
}

TEST(ReplCommandsTest, wave156_graph_bfs) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_bfs(A,source)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "ord = graph_bfs(A, 0)");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ord").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("ord")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ord")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ord")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave156_poly_compose) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_compose(p,q)");

    expect_ok(interp, "c = poly_compose([1; 1], [0; 2])");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave156_stats_var) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_var(x)");

    expect_ok(interp, "v = stats_var([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("v"), 2.5, 1e-9);

    expect_contains(interp, "stats_var([1; 2; 3; 4; 5])", "2.5");
}

TEST(ReplCommandsTest, wave157_prob_norm_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_norm_ppf(p,mu,sigma)");

    expect_ok(interp, "q = prob_norm_ppf(0.5, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("q"), 0.0, 1e-9);

    expect_contains(interp, "prob_norm_ppf(0.5, 0, 1)", "0");
}

TEST(ReplCommandsTest, wave157_signal_triangular) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_triangular(n)");

    expect_ok(interp, "w = signal_triangular(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
}

TEST(ReplCommandsTest, wave157_graph_is_tree) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_tree(A)");

    expect_ok(interp, "tr = graph_is_tree([0, 1, 0; 1, 0, 1; 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("tr"), 1.0, 1e-9);

    expect_ok(interp, "nt = graph_is_tree([0, 1, 1; 1, 0, 1; 1, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("nt"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave158_graph_dfs) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_dfs(A,source)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "ord = graph_dfs(A, 0)");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ord").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("ord")(0, 0), 0.0, 1e-9);

    expect_contains(interp, "graph_dfs(A, 0)", "0");
}

TEST(ReplCommandsTest, wave158_stats_percentile) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_percentile(x,p)");

    expect_ok(interp, "p50 = stats_percentile([1; 2; 3; 4; 5], 50)");
    EXPECT_NEAR(interp.state().scalars.at("p50"), 3.0, 1e-9);

    expect_contains(interp, "stats_percentile([1; 2; 3; 4; 5], 50)", "3");
}

TEST(ReplCommandsTest, wave158_signal_lowpass) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_lowpass(x,cutoff,fs)");

    expect_ok(interp, "y = signal_lowpass([1; 0; -1; 0], 0.25, 1.0)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 4u);

    expect_ok(interp, "signal_lowpass([1; 0; -1; 0], 0.25, 1.0)");
}

TEST(ReplCommandsTest, wave159_prob_binom_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_binom_pdf(k,n,p)");

    expect_ok(interp, "pk = prob_binom_pdf(2, 4, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("pk"), 0.375, 1e-9);

    expect_contains(interp, "prob_binom_pdf(2, 4, 0.5)", "0.375");
}

TEST(ReplCommandsTest, wave159_graph_topological_sort) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_topological_sort(A)");

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ord").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("ord")(0, 0), 0.0, 1e-9);

    expect_contains(interp, "graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])", "0");
}

TEST(ReplCommandsTest, wave159_stats_mode) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mode(x)");

    expect_ok(interp, "m = stats_mode([1; 2; 2; 3])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 2.0, 1e-9);

    expect_contains(interp, "stats_mode([1; 2; 2; 3])", "2");
}

TEST(ReplCommandsTest, wave160_prob_exp_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_exp_cdf(x,lambda)");

    expect_ok(interp, "c = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 1.0 - std::exp(-1.0), 1e-9);

    expect_contains(interp, "prob_exp_cdf(1, 1)", "0.632");
}

TEST(ReplCommandsTest, wave160_fft_dft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dft(x)");

    expect_ok(interp, "S = fft_dft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 1.0, 1e-9);

    expect_ok(interp, "fft_dft([1; 0; 0; 0])");
}

TEST(ReplCommandsTest, wave260_fft_goertzel) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_goertzel(x,f,fs)");

    // DC bin of a constant length-4 signal: sum of samples = 4.
    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("G").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 4.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 1), 0.0, 1e-9);

    expect_ok(interp, "fft_goertzel([1; 1; 1; 1], 0, 4)");
}

TEST(ReplCommandsTest, wave160_graph_greedy_colour) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_greedy_colour(A)");

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);
    EXPECT_EQ(interp.state().matrices.at("col").rows(), 4u);

    expect_ok(interp, "graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
}

TEST(ReplCommandsTest, wave161_prob_binom_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_binom_cdf(k,n,p)");

    expect_ok(interp, "cdf = prob_binom_cdf(2, 4, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cdf"), 0.6875, 1e-9);

    expect_contains(interp, "prob_binom_cdf(2, 4, 0.5)", "0.6875");
}

TEST(ReplCommandsTest, wave161_signal_butterworth) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_butterworth(x,cutoff,fs)");

    expect_ok(interp, "y = signal_butterworth([1; 0; -1; 0], 0.25, 1.0)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 4u);

    expect_ok(interp, "signal_butterworth([1; 0; -1; 0], 0.25, 1.0)");
}

TEST(ReplCommandsTest, wave161_graph_euler_circuit) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_euler_circuit(A)");

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);
    EXPECT_GT(interp.state().matrices.at("circ").rows(), 1u);

    expect_ok(interp, "graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
}

TEST(ReplCommandsTest, wave162_prob_pois_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_pois_pdf(k,lambda)");

    expect_ok(interp, "pk = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pk"), 2.0 * std::exp(-2.0), 1e-9);

    expect_ok(interp, "prob_pois_pdf(2, 2)");
}

TEST(ReplCommandsTest, wave162_stats_geometric_mean) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_geometric_mean(x)");

    expect_ok(interp, "gm = stats_geometric_mean([2; 8])");
    EXPECT_NEAR(interp.state().scalars.at("gm"), 4.0, 1e-9);

    expect_contains(interp, "stats_geometric_mean([2; 8])", "4");
}

TEST(ReplCommandsTest, wave162_signal_highpass) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_highpass(x,cutoff,fs)");

    expect_ok(interp, "hp = signal_highpass([1; 0; -1; 0], 0.25, 1.0)");
    ASSERT_GT(interp.state().matrices.count("hp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "signal_highpass([1; 0; -1; 0], 0.25, 1.0)");
}

TEST(ReplCommandsTest, wave163_prob_uniform_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_uniform_cdf(x,a,b)");

    expect_ok(interp, "ucdf = prob_uniform_cdf(3, 0, 10)");
    EXPECT_NEAR(interp.state().scalars.at("ucdf"), 0.3, 1e-9);

    expect_contains(interp, "prob_uniform_cdf(3, 0, 10)", "0.3");
}

TEST(ReplCommandsTest, wave163_stats_ttest) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ttest(x,mu)");

    expect_ok(interp, "t = stats_ttest([6; 7; 8; 9; 10], 5)");
    EXPECT_GT(interp.state().scalars.at("t"), 0.0);

    expect_ok(interp, "stats_ttest([6; 7; 8; 9; 10], 5)");
}

TEST(ReplCommandsTest, wave163_graph_bellman_ford_dist) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_bellman_ford_dist");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_contains(interp, "graph_bellman_ford_dist(A, 0, 2)", "3");
}

TEST(ReplCommandsTest, wave164_prob_pois_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_pois_cdf(k,lambda)");

    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_GT(interp.state().scalars.at("pc"), 0.0);
    EXPECT_LT(interp.state().scalars.at("pc"), 1.0);

    expect_ok(interp, "prob_pois_cdf(2, 2)");
}

TEST(ReplCommandsTest, wave164_stats_harmonic_mean) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_harmonic_mean(x)");

    expect_ok(interp, "hm = stats_harmonic_mean([1; 2; 3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("hm"), 48.0 / 25.0, 1e-9);

    expect_contains(interp, "stats_harmonic_mean([1; 2; 3; 4])", "1.92");
}

TEST(ReplCommandsTest, wave164_signal_bandpass) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_bandpass(x,low,high,fs)");

    expect_ok(interp, "bp = signal_bandpass([1; 0; -1; 0], 0.1, 0.3, 1.0)");
    ASSERT_GT(interp.state().matrices.count("bp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bp").rows(), 4u);

    expect_ok(interp, "signal_bandpass([1; 0; -1; 0], 0.1, 0.3, 1.0)");
}

TEST(ReplCommandsTest, wave165_prob_exp_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_exp_pdf(x,lambda)");

    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), std::exp(-1.0), 1e-9);

    expect_ok(interp, "prob_exp_pdf(1, 1)");
}

TEST(ReplCommandsTest, wave165_stats_rms) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_rms(x)");

    expect_ok(interp, "r = stats_rms([3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("r"), std::sqrt(12.5), 1e-9);

    expect_contains(interp, "stats_rms([3; 4])", "3.5355");
}

TEST(ReplCommandsTest, wave165_fft_ifft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_ifft(spectrum)");

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);

    expect_ok(interp, "fft_ifft(S)");
}

TEST(ReplCommandsTest, wave166_prob_chi2_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_chi2_cdf(x,df)");

    expect_ok(interp, "cc = prob_chi2_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), 0.682689492, 1e-3);

    expect_ok(interp, "prob_chi2_cdf(1, 1)");
}

TEST(ReplCommandsTest, wave166_stats_mad) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mad(x)");

    expect_ok(interp, "m = stats_mad([1; 1; 2; 2; 4; 6; 9])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 1.4826, 1e-9);

    expect_contains(interp, "stats_mad([1; 1; 2; 2; 4; 6; 9])", "1.4826");
}

TEST(ReplCommandsTest, wave166_graph_astar) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_astar");

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "h = [3; 2; 1; 0]");
    expect_ok(interp, "path = graph_astar(A, 0, 3, h)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("path")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("path")(3, 0), 3.0, 1e-9);

    expect_ok(interp, "graph_astar(A, 0, 3, h)");
}

TEST(ReplCommandsTest, wave167_prob_gamma_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_gamma_pdf(x,shape,scale)");

    expect_ok(interp, "gp = prob_gamma_pdf(2, 3, 1)");
    EXPECT_GT(interp.state().scalars.at("gp"), 0.0);

    expect_ok(interp, "prob_gamma_pdf(2, 3, 1)");
}

TEST(ReplCommandsTest, wave167_stats_ztest) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ztest(x,mu,sigma)");

    expect_ok(interp, "z = stats_ztest([6; 7; 8; 9; 10], 5, 1)");
    EXPECT_GT(interp.state().scalars.at("z"), 0.0);

    expect_ok(interp, "stats_ztest([6; 7; 8; 9; 10], 5, 1)");
}

TEST(ReplCommandsTest, wave167_stats_acf) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_acf(x,max_lag)");

    expect_ok(interp, "a = stats_acf([1; 2; 3; 4; 5], 2)");
    ASSERT_GT(interp.state().matrices.count("a"), 0u);
    EXPECT_EQ(interp.state().matrices.at("a").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("a")(0, 0), 1.0, 1e-9);

    expect_ok(interp, "stats_acf([1; 2; 3; 4; 5], 2)");
}

TEST(ReplCommandsTest, wave168_prob_t_cdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_t_cdf(x,df)");

    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), 0.5, 1e-6);

    expect_ok(interp, "prob_t_cdf(0, 5)");
}

TEST(ReplCommandsTest, wave168_stats_iqr) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_iqr(x)");

    expect_ok(interp, "iq = stats_iqr([1; 2; 3; 4; 5; 6; 7; 8; 9])");
    EXPECT_GT(interp.state().scalars.at("iq"), 0.0);

    expect_contains(interp, "stats_iqr([1; 2; 3; 4; 5; 6; 7; 8; 9])", "4");
}

TEST(ReplCommandsTest, wave168_fft_fft2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_fft2(S)");

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "F = fft_fft2(S)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("F").cols(), 2u);

    expect_ok(interp, "fft_fft2(S)");
}

TEST(ReplCommandsTest, wave169_prob_chi2_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_chi2_pdf(x,df)");

    expect_ok(interp, "cp = prob_chi2_pdf(1, 3)");
    EXPECT_GT(interp.state().scalars.at("cp"), 0.0);

    expect_ok(interp, "prob_chi2_pdf(1, 3)");
}

TEST(ReplCommandsTest, wave169_stats_two_sample_ttest) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_two_sample_ttest(a,b)");

    expect_ok(interp, "t2 = stats_two_sample_ttest([1; 2; 3], [4; 5; 6])");
    EXPECT_LT(interp.state().scalars.at("t2"), 0.0);

    expect_ok(interp, "stats_two_sample_ttest([1; 2; 3], [4; 5; 6])");
}

TEST(ReplCommandsTest, wave169_stats_chi2_gof) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_chi2_gof(observed,expected)");

    expect_ok(interp, "g = stats_chi2_gof([10; 20; 30], [10; 20; 30])");
    EXPECT_NEAR(interp.state().scalars.at("g"), 0.0, 1e-9);

    expect_ok(interp, "stats_chi2_gof([10; 20; 30], [10; 20; 30])");
}

TEST(ReplCommandsTest, wave170_geo_kdtree_nearest) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_nearest(P,x,y)");

    expect_ok(interp, "idx = geo_kdtree_nearest([0, 0; 1, 0; 2, 0; 3, 0; 4, 0], 1.1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("idx"), 1.0, 1e-9);

    expect_contains(interp, "geo_kdtree_nearest([0, 0; 1, 0; 2, 0; 3, 0; 4, 0], 1.1, 0)", "1");
}

TEST(ReplCommandsTest, wave255_geo_kdtree_query) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_knn(P,x,y,k)");
    expect_contains(interp, "help", "geo_kdtree_range(P,x,y,r)");

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("n").cols(), 1u);

    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("r").cols(), 1u);
}

TEST(ReplCommandsTest, wave170_topo_pairwise_distances) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_pairwise_distances(P)");

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
}

TEST(ReplCommandsTest, wave170_numthy_continued_fraction) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_continued_fraction(x,n)");

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);

    expect_ok(interp, "numthy_continued_fraction(3.14159, 5)");
}

TEST(ReplCommandsTest, wave171_combo_next_perm) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_next_perm(v)");

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);

    expect_ok(interp, "combo_next_perm([1; 2; 3])");
}

TEST(ReplCommandsTest, wave171_cplx_mobius_re) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_mobius_re(a,b,c,d,zre,zim)");

    expect_ok(interp, "mr = cplx_mobius_re(1, 1, 1, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("mr"), 2.0, 1e-9);

    expect_contains(interp, "cplx_mobius_re(1, 1, 1, 0, 1, 0)", "2");
}

TEST(ReplCommandsTest, wave171_boxfilter) {
    Interpreter interp;
    expect_contains(interp, "help", "boxfilter(M,k)");

    expect_ok(interp, "M = ones(5, 5)");
    expect_ok(interp, "B = boxfilter(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "boxfilter(M, 3)");
}

TEST(ReplCommandsTest, wave172_geo_voronoi) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_voronoi(P)");

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
}

TEST(ReplCommandsTest, wave172_numthy_convergents) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_convergents(cf)");

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "numthy_convergents(cf)");
}

TEST(ReplCommandsTest, wave172_ml_mat_transpose) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mat_transpose(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_EQ(interp.state().matrices.at("At").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 4.0, 1e-9);

    expect_ok(interp, "ml_mat_transpose(A)");
}

TEST(ReplCommandsTest, wave173_combo_next_comb) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_next_comb(v,n)");

    expect_ok(interp, "v2 = combo_next_comb([0; 1], 4)");
    ASSERT_GT(interp.state().matrices.count("v2"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("v2")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("v2")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave173_numthy_primes) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_primes(lo,hi)");

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave173_graph_scc) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_scc(A)");

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, wave174_geo_hermite_x) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_hermite_x(");

    expect_ok(interp, "hx = geo_hermite_x(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hx"), 0.5, 1e-9);
}

TEST(ReplCommandsTest, wave174_ml_mat_mul) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mat_mul(A,B)");

    expect_ok(interp, "I = eye(3)");
    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "C = ml_mat_mul(I, A)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("C")(2, 2), 9.0, 1e-9);
}

TEST(ReplCommandsTest, wave174_stats_min_value) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_min_value(x)");

    expect_ok(interp, "m = stats_min_value([3; 1; 4; 1; 5; 9; 2])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 1.0, 1e-9);
    expect_contains(interp, "stats_min_value([3; 1; 4; 1; 5; 9; 2])", "1");
}

TEST(ReplCommandsTest, wave175_count_components) {
    Interpreter interp;
    expect_contains(interp, "help", "count_components(B)");

    expect_ok(interp, "B = [1, 1, 0, 0, 0, 0; 1, 1, 0, 0, 0, 0; 0, 0, 0, 0, 0, 0; 0, 0, 0, 0, 0, 1; 0, 0, 0, 0, 1, 1; 0, 0, 0, 0, 0, 1]");
    expect_ok(interp, "n = count_components(B)");
    EXPECT_NEAR(interp.state().scalars.at("n"), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave175_prewitt) {
    Interpreter interp;
    expect_contains(interp, "help", "prewitt(M)");

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
}

TEST(ReplCommandsTest, wave175_fftshift) {
    Interpreter interp;
    expect_contains(interp, "help", "fftshift(S)");

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave261_fftfreq) {
    Interpreter interp;
    expect_contains(interp, "help", "fftfreq(n[,d])");

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    ASSERT_EQ(freqs.rows(), 8u);
    ASSERT_EQ(freqs.cols(), 1u);
    const std::vector<double> expected{0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    const auto& scaled = interp.state().matrices.at("f2");
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(scaled(i, 0), expected[i] / 0.5, 1e-12);
    }
}

TEST(ReplCommandsTest, wave261_rfftfreq) {
    Interpreter interp;
    expect_contains(interp, "help", "rfftfreq(n[,d])");

    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
    const auto& freqs = interp.state().matrices.at("rf");
    ASSERT_EQ(freqs.rows(), 5u);
    ASSERT_EQ(freqs.cols(), 1u);
    const std::vector<double> expected{0.0, 0.125, 0.25, 0.375, 0.5};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "rf2 = rfftfreq(5)");
    const auto& odd = interp.state().matrices.at("rf2");
    ASSERT_EQ(odd.rows(), 3u);
    EXPECT_NEAR(odd(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(odd(1, 0), 0.2, 1e-12);
    EXPECT_NEAR(odd(2, 0), 0.4, 1e-12);
}

TEST(ReplCommandsTest, wave261_ifftshift) {
    Interpreter interp;
    expect_contains(interp, "help", "ifftshift(S)");

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
    const auto& restored = interp.state().matrices.at("Sr");
    EXPECT_NEAR(restored(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(restored(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(restored(2, 0), 2.0, 1e-9);
    EXPECT_NEAR(restored(3, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, wave109_geo_signed_area) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_signed_area(P)");

    expect_ok(interp, "sa = geo_signed_area([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("sa"), 16.0, 1e-9);

    expect_contains(interp, "geo_signed_area([0, 0; 4, 0; 4, 4; 0, 4])", "16");
}

TEST(ReplCommandsTest, wave109_geo_moment_of_inertia) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_moment_of_inertia(P)");

    expect_ok(interp, "mi = geo_moment_of_inertia([0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("mi"), 128.0 / 3.0, 1e-5);

    expect_contains(interp, "geo_moment_of_inertia([0, 0; 4, 0; 4, 4; 0, 4])", "42.6667");
}

TEST(ReplCommandsTest, wave109_geo_dist_point_seg2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist_point_seg2d(px,py,x1,y1,x2,y2)");

    expect_ok(interp, "dps = geo_dist_point_seg2d(2, 3, 0, 0, 4, 0)");
    EXPECT_NEAR(interp.state().scalars.at("dps"), 3.0, 1e-9);

    expect_contains(interp, "geo_dist_point_seg2d(2, 3, 0, 0, 4, 0)", "3");
}

TEST(ReplCommandsTest, wave110_geo_point_in_polygon) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_point_in_polygon(px,py,P)");

    expect_ok(interp, "pip = geo_point_in_polygon(2, 2, [0, 0; 4, 0; 4, 4; 0, 4])");
    EXPECT_NEAR(interp.state().scalars.at("pip"), 1.0, 1e-9);

    expect_contains(interp, "geo_point_in_polygon(2, 2, [0, 0; 4, 0; 4, 4; 0, 4])", "1");
}

TEST(ReplCommandsTest, wave110_ml_categorical_crossentropy) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_categorical_crossentropy(p,t)");

    expect_ok(interp, "cce = ml_categorical_crossentropy([0.7, 0.2, 0.1; 0.1, 0.8, 0.1], [1, 0, 0; 0, 1, 0])");
    EXPECT_GT(interp.state().scalars.at("cce"), 0.0);
    EXPECT_LT(interp.state().scalars.at("cce"), 1.0);

    expect_contains(interp,
                    "ml_categorical_crossentropy([0.7, 0.2, 0.1; 0.1, 0.8, 0.1], [1, 0, 0; 0, 1, 0])",
                    "0.289909");
}

TEST(ReplCommandsTest, wave110_geo_overlap_circles) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_overlap_circles(x1,y1,r1,x2,y2,r2)");

    expect_ok(interp, "ov = geo_overlap_circles(0, 0, 1, 0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ov"), 1.0, 1e-9);

    expect_contains(interp, "geo_overlap_circles(0, 0, 1, 0, 0, 1)", "1");
}

TEST(ReplCommandsTest, wave254_geo_aabb) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_point_in_aabb(px,py,minx,miny,maxx,maxy)");
    expect_contains(interp, "help",
                    "geo_overlap_aabb(aminx,aminy,aminz,amaxx,amaxy,amaxz,bminx,bminy,bminz,bmaxx,bmaxy,bmaxz)");

    expect_ok(interp, "inside = geo_point_in_aabb(1, 1, 0, 0, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("inside"), 1.0, 1e-9);
    expect_ok(interp, "outside = geo_point_in_aabb(3, 1, 0, 0, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("outside"), 0.0, 1e-9);
    expect_contains(interp, "geo_point_in_aabb(1, 1, 0, 0, 2, 2)", "1");

    expect_ok(interp, "hit = geo_overlap_aabb(0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_overlap_aabb(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(interp, "geo_overlap_aabb(0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5)", "1");
}

TEST(ReplCommandsTest, wave255_geo_ray_intersect) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_intersect_seg_seg(x1,y1,x2,y2,x3,y3,x4,y4)");
    expect_contains(interp, "help", "geo_intersect_ray_sphere(ox,oy,oz,dx,dy,dz,cx,cy,cz,r)");
    expect_contains(interp, "help",
                    "geo_intersect_ray_aabb(ox,oy,oz,dx,dy,dz,minx,miny,minz,maxx,maxy,maxz)");

    expect_ok(interp, "hit = geo_intersect_seg_seg(0, 0, 2, 2, 0, 2, 2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = geo_intersect_seg_seg(0, 0, 1, 0, 0, 1, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(interp, "geo_intersect_seg_seg(0, 0, 2, 2, 0, 2, 2, 0)", "1");
    expect_contains(interp, "geo_intersect_seg_seg(0, 0, 1, 0, 0, 1, 1, 1)", "0");

    expect_ok(interp, "rs = geo_intersect_ray_sphere(0, 0, 0, 1, 0, 0, 2, 0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rs"), 1.0, 1e-9);

    expect_ok(interp, "ra = geo_intersect_ray_aabb(-5, 0, 0, 1, 0, 0, -1, -1, -1, 1, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ra"), 1.0, 1e-9);
    expect_contains(interp, "geo_intersect_ray_aabb(-5, 0, 0, 1, 0, 0, -1, -1, -1, 1, 1, 1)", "1");
}

TEST(ReplCommandsTest, wave255_geo_triangulate_hull3d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_triangulate_polygon(P)");
    expect_contains(interp, "help", "geo_convex_hull_3d(P)");

    expect_ok(interp, "P = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "T = geo_triangulate_polygon(P)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
}

TEST(ReplCommandsTest, wave111_geo_bezier_eval_x) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_eval_x(P,t)");

    expect_ok(interp, "bx = geo_bezier_eval_x([0, 0; 1, 2; 2, 0], 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bx"), 1.0, 1e-9);

    expect_contains(interp, "geo_bezier_eval_x([0, 0; 1, 2; 2, 0], 0.5)", "1");
}

TEST(ReplCommandsTest, wave111_geo_bezier_eval_y) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_eval_y(P,t)");

    expect_ok(interp, "by = geo_bezier_eval_y([0, 0; 1, 2; 2, 0], 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by"), 1.0, 1e-9);

    expect_contains(interp, "geo_bezier_eval_y([0, 0; 1, 2; 2, 0], 0.5)", "1");
}

TEST(ReplCommandsTest, wave259_geo_bezier_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_eval(P,t)");

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, wave260_geo_bezier_deriv) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bezier_deriv(P,t)");

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("d").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave260_geo_hermite_curve) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_hermite_curve(p0x,p0y,m0x,m0y,p1x,p1y,m1x,m1y,t)");

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 0.25, 1e-9);
}

TEST(ReplCommandsTest, wave259_geo_catmull_rom) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_catmull_rom(P,t)");

    expect_ok(interp, "ctrl = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "pt = geo_catmull_rom(ctrl, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.5, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave259_geo_bspline_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_bspline_eval(P,knots,degree,t)");

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "pt = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, wave105_ml_binary_crossentropy) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_binary_crossentropy(p,t)");

    expect_ok(interp, "bce = ml_binary_crossentropy([0.9; 0.1], [1; 0])");
    EXPECT_LT(interp.state().scalars.at("bce"), 0.15);

    expect_contains(interp, "ml_binary_crossentropy([0.9; 0.1], [1; 0])", "0.10536");
}

TEST(ReplCommandsTest, wave105_numthy_is_primitive_root) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_is_primitive_root(g,p)");

    expect_ok(interp, "pr = numthy_is_primitive_root(2, 11)");
    EXPECT_NEAR(interp.state().scalars.at("pr"), 1.0, 1e-9);

    expect_contains(interp, "numthy_is_primitive_root(2, 11)", "1");
}

TEST(ReplCommandsTest, wave105_numthy_discrete_log) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_discrete_log(g,h,p)");

    expect_ok(interp, "dl = numthy_discrete_log(2, 8, 11)");
    EXPECT_NEAR(interp.state().scalars.at("dl"), 3.0, 1e-9);

    expect_contains(interp, "numthy_discrete_log(2, 8, 11)", "3");
}

TEST(ReplCommandsTest, wave194_numthy_von_mangoldt) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_von_mangoldt(n)");

    // 8 = 2^3, so von Mangoldt(8) = ln(2).
    expect_ok(interp, "vm = numthy_von_mangoldt(8)");
    EXPECT_NEAR(interp.state().scalars.at("vm"), std::log(2.0), 1e-9);

    expect_contains(interp, "numthy_von_mangoldt(8)", "0.693");
}

TEST(ReplCommandsTest, wave194_numthy_jordan_totient) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_jordan_totient(k,n)");

    // J_2(6) = 36*(1-1/2)*(1-1/3) = 12.
    expect_ok(interp, "jt = numthy_jordan_totient(2, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 12.0, 1e-9);

    expect_contains(interp, "numthy_jordan_totient(2, 6)", "12");
}

TEST(ReplCommandsTest, wave194_poly_bernstein) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_bernstein(n,i,x)");

    // B_{2,1}(0.5) = C(2,1)*0.5*0.5 = 0.5.
    expect_ok(interp, "bn = poly_bernstein(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bn"), 0.5, 1e-9);

    expect_contains(interp, "poly_bernstein(2, 1, 0.5)", "0.5");
}

TEST(ReplCommandsTest, wave194_finance_capm) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_capm(risk_free,beta,market_return)");

    // 0.05 + 1.2 * (0.10 - 0.05) = 0.11.
    expect_ok(interp, "capm = finance_capm(0.05, 1.2, 0.10)");
    EXPECT_NEAR(interp.state().scalars.at("capm"), 0.11, 1e-9);

    expect_contains(interp, "finance_capm(0.05, 1.2, 0.10)", "0.11");
}

TEST(ReplCommandsTest, wave194_finance_forward_rate) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_forward_rate(r1,t1,r2,t2)");

    // (0.06*2 - 0.05*1) / (2 - 1) = 0.07.
    expect_ok(interp, "fr = finance_forward_rate(0.05, 1.0, 0.06, 2.0)");
    EXPECT_NEAR(interp.state().scalars.at("fr"), 0.07, 1e-9);

    expect_contains(interp, "finance_forward_rate(0.05, 1.0, 0.06, 2.0)", "0.07");
}

TEST(ReplCommandsTest, wave194_finance_black76) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_black76(F,K,T,r,sigma,call)");

    // With F = S*exp(r*T) (S=100,r=0.05,T=1), black76 call == bs_call(100,100,1,0.05,0.2)
    // == 10.450583572185565 (reused from the finance_bs_implied_vol precedent).
    expect_ok(interp, "b76 = finance_black76(105.12710963760241, 100, 1, 0.05, 0.2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("b76"), 10.450583572185565, 1e-6);

    expect_contains(interp, "finance_black76(105.12710963760241, 100, 1, 0.05, 0.2, 1)", "10.45");
}

TEST(ReplCommandsTest, wave194_finance_digital_option) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_digital_option(S,K,T,r,sigma,call,payout)");

    // Deep ITM call: price ~ payout * exp(-r*T) = exp(-0.05).
    expect_ok(interp, "d = finance_digital_option(200, 100, 1, 0.05, 0.2, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), std::exp(-0.05), 0.01);

    expect_contains(interp, "finance_digital_option(200, 100, 1, 0.05, 0.2, 1, 1)", "0.9");
}

TEST(ReplCommandsTest, wave194_finance_american_option) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_american_option(S,K,T,r,sigma,call,steps)");

    // American call without dividends should be close to the European (BS) call price.
    expect_ok(interp, "am = finance_american_option(100, 100, 1, 0.05, 0.2, 1, 200)");
    EXPECT_NEAR(interp.state().scalars.at("am"), 10.450583572185565, 1.0);

    expect_contains(interp, "finance_american_option(100, 100, 1, 0.05, 0.2, 1, 200)", "\n");
}

TEST(ReplCommandsTest, wave194_finance_mc_european_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_european_call(S,K,T,r,sigma,n_paths,seed)");

    // Loose tolerance vs. the analytic BS call price: this checks REPL plumbing
    // (right function/args/result), not Monte Carlo statistical accuracy.
    expect_ok(interp, "mc = finance_mc_european_call(100, 100, 1, 0.05, 0.2, 20000, 7)");
    EXPECT_GT(interp.state().scalars.at("mc"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("mc"), 10.450583572185565, 3.0);

    expect_contains(interp, "finance_mc_european_call(100, 100, 1, 0.05, 0.2, 20000, 7)", "\n");
}

TEST(ReplCommandsTest, wave194_finance_mc_european_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_european_put(S,K,T,r,sigma,n_paths,seed)");

    // bs_put(100,100,1,0.05,0.2) via put-call parity: bs_call - S + K*exp(-r*T).
    const double bs_put_ref = 10.450583572185565 - 100.0 + 100.0 * std::exp(-0.05);
    expect_ok(interp, "mc = finance_mc_european_put(100, 100, 1, 0.05, 0.2, 20000, 7)");
    EXPECT_GT(interp.state().scalars.at("mc"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("mc"), bs_put_ref, 3.0);

    expect_contains(interp, "finance_mc_european_put(100, 100, 1, 0.05, 0.2, 20000, 7)", "\n");
}

TEST(ReplCommandsTest, wave194_finance_mc_asian_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_asian_call(S,K,T,r,sigma,n_paths,n_steps,seed)");

    // Arithmetic-average Asian call should be cheaper than the corresponding European call.
    expect_ok(interp, "ac = finance_mc_asian_call(100, 100, 1, 0.05, 0.2, 20000, 50, 3)");
    EXPECT_GT(interp.state().scalars.at("ac"), 0.0);
    EXPECT_LT(interp.state().scalars.at("ac"), 10.450583572185565);

    expect_contains(interp, "finance_mc_asian_call(100, 100, 1, 0.05, 0.2, 20000, 50, 3)", "\n");
}

TEST(ReplCommandsTest, wave194_finance_mc_asian_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_asian_put(S,K,T,r,sigma,n_paths,n_steps,seed)");

    expect_ok(interp, "ap = finance_mc_asian_put(90, 110, 1, 0.05, 0.25, 20000, 50, 3)");
    EXPECT_GT(interp.state().scalars.at("ap"), 0.0);

    expect_contains(interp, "finance_mc_asian_put(90, 110, 1, 0.05, 0.25, 20000, 50, 3)", "\n");
}

TEST(ReplCommandsTest, wave194_finance_barrier_option) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_barrier_option(S,K,B,T,r,sigma,call,knock_in,up)");

    // Down-and-in + down-and-out call must reconstruct the vanilla call price.
    expect_ok(interp, "ki = finance_barrier_option(100, 100, 90, 1, 0.05, 0.2, 1, 1, 0)");
    expect_ok(interp, "ko = finance_barrier_option(100, 100, 90, 1, 0.05, 0.2, 1, 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("ki") + interp.state().scalars.at("ko"),
                10.450583572185565, 1e-4);

    // This also exercises the new 9-arg (nonary) regex literal-dispatch path added
    // specifically for finance_barrier_option.
    expect_contains(interp, "finance_barrier_option(100, 100, 90, 1, 0.05, 0.2, 1, 1, 0)", "\n");
}

TEST(ReplCommandsTest, wave195_special_erfinv) {
    Interpreter interp;
    expect_contains(interp, "help", "special_erfinv(x)");

    const double ref = ms::erfinv(0.5);
    expect_ok(interp, "y = special_erfinv(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_erfinv(0.5)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_erfcinv) {
    Interpreter interp;
    expect_contains(interp, "help", "special_erfcinv(x)");

    // erfcinv(x) == erfinv(1-x) (DLMF 7.17 relationship between erf^-1 and erfc^-1).
    const double ref = ms::erfinv(1.0 - 0.3);
    expect_ok(interp, "y = special_erfcinv(0.3)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_erfcinv(0.3)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_log_gamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_log_gamma(x)");

    // log_gamma(5) == ln(4!) == ln(24).
    const double ref = ms::log_gamma(5.0);
    EXPECT_NEAR(ref, std::log(24.0), 1e-9);
    expect_ok(interp, "y = special_log_gamma(5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_log_gamma(5)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_digamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_digamma(x)");

    // digamma(1) == -EulerGamma.
    const double ref = ms::digamma(1.0);
    EXPECT_NEAR(ref, -0.5772156649015329, 1e-6);
    expect_ok(interp, "y = special_digamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_digamma(1)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_trigamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_trigamma(x)");

    // trigamma(1) == pi^2/6 (implementation uses a series approximation, hence the looser tolerance).
    const double ref = ms::trigamma(1.0);
    constexpr double kPi = 3.14159265358979323846;
    EXPECT_NEAR(ref, kPi * kPi / 6.0, 1e-5);
    expect_ok(interp, "y = special_trigamma(1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_trigamma(1)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_polygamma) {
    Interpreter interp;
    expect_contains(interp, "help", "special_polygamma(n,x)");

    // polygamma(1, x) is the trigamma function.
    const double ref = ms::polygamma(1, 1.0);
    EXPECT_NEAR(ref, ms::trigamma(1.0), 1e-9);
    expect_ok(interp, "y = special_polygamma(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_polygamma(1, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_gamma_inc_reg) {
    Interpreter interp;
    expect_contains(interp, "help", "special_gamma_inc_reg(a,x)");

    // P(1, x) == 1 - exp(-x); use x = ln(2) so the expected value is 0.5.
    // 0.69314718055994530942 is ln(2) to full double precision (std::to_string would
    // truncate to 6 decimals and reintroduce error when re-parsed by the REPL).
    const std::string x_str = "0.69314718055994530942";
    const double x = std::log(2.0);
    const double ref = ms::gamma_inc_reg(1.0, x);
    EXPECT_NEAR(ref, 0.5, 1e-6);
    expect_ok(interp, "y = special_gamma_inc_reg(1, " + x_str + ")");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_gamma_inc_reg(1, " + x_str + ")", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_gamma_inc_reg_upper) {
    Interpreter interp;
    expect_contains(interp, "help", "special_gamma_inc_reg_upper(a,x)");

    // Q(1, x) == exp(-x); use x = ln(2) so the expected value is 0.5.
    const std::string x_str = "0.69314718055994530942";
    const double x = std::log(2.0);
    const double ref = ms::gamma_inc_reg_upper(1.0, x);
    EXPECT_NEAR(ref, 0.5, 1e-6);
    expect_ok(interp, "y = special_gamma_inc_reg_upper(1, " + x_str + ")");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_gamma_inc_reg_upper(1, " + x_str + ")",
                    std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_special_beta_inc_reg) {
    Interpreter interp;
    expect_contains(interp, "help", "special_beta_inc_reg(x,a,b)");

    // I_x(1,1) == x since Beta(1,1) is the uniform distribution on [0,1].
    const double ref = ms::beta_inc_reg(0.3, 1.0, 1.0);
    EXPECT_NEAR(ref, 0.3, 1e-6);
    expect_ok(interp, "y = special_beta_inc_reg(0.3, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "special_beta_inc_reg(0.3, 1, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_uniform_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_uniform_pdf(x,a,b)");

    // uniform_pdf on [1,2] is constant 1/(2-1) = 1.
    const double ref = ms::uniform_pdf(1.5, 1.0, 2.0);
    EXPECT_NEAR(ref, 1.0, 1e-9);
    expect_ok(interp, "y = prob_uniform_pdf(1.5, 1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_uniform_pdf(1.5, 1, 2)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_t_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_t_pdf(x,df)");

    const double ref = ms::t_pdf(1.0, 5.0);
    expect_ok(interp, "y = prob_t_pdf(1, 5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_t_pdf(1, 5)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_t_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_t_ppf(p,df)");

    // The Student-t distribution is symmetric, so its median (p=0.5) is 0.
    const double ref = ms::t_ppf(0.5, 10.0);
    EXPECT_NEAR(ref, 0.0, 1e-9);
    expect_ok(interp, "y = prob_t_ppf(0.5, 10)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_t_ppf(0.5, 10)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_gamma_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_gamma_ppf(p,shape,scale)");

    const double ref = ms::gamma_ppf(0.5, 2.0, 1.0);
    expect_ok(interp, "y = prob_gamma_ppf(0.5, 2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_gamma_ppf(0.5, 2, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_beta_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_beta_ppf(p,alpha,beta)");

    // Beta(1,1) is the uniform distribution on [0,1], so its ppf is the identity.
    const double ref = ms::beta_ppf(0.5, 1.0, 1.0);
    EXPECT_NEAR(ref, 0.5, 1e-9);
    expect_ok(interp, "y = prob_beta_ppf(0.5, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_beta_ppf(0.5, 1, 1)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_f_pdf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_f_pdf(x,d1,d2)");

    const double ref = ms::f_pdf(1.0, 5.0, 5.0);
    expect_ok(interp, "y = prob_f_pdf(1, 5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_f_pdf(1, 5, 5)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_prob_f_ppf) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_f_ppf(p,d1,d2)");

    const double ref = ms::f_ppf(0.5, 5.0, 5.0);
    expect_ok(interp, "y = prob_f_ppf(0.5, 5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);

    expect_contains(interp, "prob_f_ppf(0.5, 5, 5)", std::to_string(ref));
}

TEST(ReplCommandsTest, wave195_finance_mc_lookback_floating_call) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_floating_call(S,T,r,sigma,n_paths,n_steps,seed)");

    // Compare REPL plumbing against the same direct library call with a fixed seed.
    const double ref = ms::finance::mc_lookback_floating_call(100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_floating_call(100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_floating_call(100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, wave195_finance_mc_lookback_floating_put) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_floating_put(S,T,r,sigma,n_paths,n_steps,seed)");

    const double ref = ms::finance::mc_lookback_floating_put(100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_floating_put(100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_floating_put(100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, wave195_finance_mc_lookback_fixed_call) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_fixed_call(S,K,T,r,sigma,n_paths,n_steps,seed)");

    const double ref =
        ms::finance::mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, wave195_finance_mc_lookback_fixed_put) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_fixed_put(S,K,T,r,sigma,n_paths,n_steps,seed)");

    const double ref =
        ms::finance::mc_lookback_fixed_put(100, 100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_fixed_put(100, 100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_fixed_put(100, 100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, wave249_crypto_sha256_and_hmac_sha256) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_sha256(hex_data)");
    expect_contains(interp, "help", "crypto_hmac_sha256(hex_key,hex_data)");

    // NIST empty-string SHA-256
    expect_contains(interp, "crypto_sha256(\"\")",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    // ASCII "abc" as hex
    expect_contains(interp, "crypto_sha256(616263)",
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // RFC 4231 HMAC-SHA256 test case 1
    expect_contains(
        interp,
        "crypto_hmac_sha256(0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b,4869205468657265)",
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

TEST(ReplCommandsTest, wave250_crypto_sha512) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_sha512(hex_data)");

    // Empty-string SHA-512 (same vector as tests/unit/test_crypto.cpp)
    expect_contains(
        interp, "crypto_sha512(\"\")",
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

TEST(ReplCommandsTest, wave251_crypto_aes256_encrypt_block) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_aes256_encrypt_block");

    // NIST FIPS-197 AES-256 block (same vector as tests/unit/test_crypto.cpp)
    expect_contains(
        interp,
        R"cmd(crypto_aes256_encrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "6bc1bee22e409f96e93d7e117393172a"))cmd",
        "a36452d23436433a516cace8bf319e9c");
}

TEST(ReplCommandsTest, wave252_crypto_aes256_decrypt_block) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_aes256_decrypt_block");

    expect_contains(
        interp,
        R"cmd(crypto_aes256_decrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "a36452d23436433a516cace8bf319e9c"))cmd",
        "6bc1bee22e409f96e93d7e117393172a");
}

TEST(ReplCommandsTest, wave253_crypto_aes128_decrypt_block) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_aes128_decrypt_block");

    // FIPS-197 AES-128 encrypt then decrypt round-trip
    expect_contains(
        interp,
        R"cmd(crypto_aes128_encrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734"))cmd",
        "3925841d02dc09fbdc118597196a0b32");
    expect_contains(
        interp,
        R"cmd(crypto_aes128_decrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "3925841d02dc09fbdc118597196a0b32"))cmd",
        "3243f6a8885a308d313198a2e0370734");
}

TEST(ReplCommandsTest, wave255_crypto_x25519_keypair) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_x25519_keypair");

    // RFC 7748 Alice public key (same vector as tests/unit/test_crypto.cpp)
    expect_contains(
        interp,
        R"cmd(crypto_x25519_keypair("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"))cmd",
        "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
}

TEST(ReplCommandsTest, wave256_stats_ts) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_linear_regression(x,y)");
    expect_contains(interp, "help", "stats_pacf(x,max_lag)");
    expect_contains(interp, "help", "stats_kde(samples,grid,h)");
    expect_contains(interp, "help", "stats_bootstrap_ci(x)");

    // y = 2x + 1
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    ASSERT_GT(interp.state().matrices.count("lr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lr").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("lr").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 2), 1.0, 1e-9);

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("p").cols(), 1u);

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    ASSERT_GT(interp.state().matrices.count("k"), 0u);
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("k").cols(), 1u);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("k")(i, 0)));
    }

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    ASSERT_GT(interp.state().matrices.count("ci"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ci").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 2u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ci")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ci")(0, 1)));
}

TEST(ReplCommandsTest, wave256_stats_inference) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_shapiro_wilk(x)");
    expect_contains(interp, "help", "stats_mann_whitney_u(a,b)");
    expect_contains(interp, "help", "stats_one_way_anova(G)");
    expect_contains(interp, "help", "stats_wilcoxon_signed_rank(x,y)");

    // Shapiro-Wilk: nearly-linear sample is non-normal-ish; still returns 1x2 [W, p].
    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    ASSERT_GT(interp.state().matrices.count("sw"), 0u);
    const auto& sw = interp.state().matrices.at("sw");
    ASSERT_EQ(sw.rows(), 1u);
    ASSERT_EQ(sw.cols(), 2u);
    EXPECT_GT(sw(0, 0), 0.0);
    EXPECT_LE(sw(0, 0), 1.0);
    EXPECT_GE(sw(0, 1), 0.0);
    EXPECT_LE(sw(0, 1), 1.0);

    // Mann-Whitney U: fully separated samples -> U=0, small p; 1x3 [U, z, p].
    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    ASSERT_GT(interp.state().matrices.count("mw"), 0u);
    const auto& mw = interp.state().matrices.at("mw");
    ASSERT_EQ(mw.rows(), 1u);
    ASSERT_EQ(mw.cols(), 3u);
    EXPECT_NEAR(mw(0, 0), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(mw(0, 1)));
    EXPECT_LT(mw(0, 2), 0.05);

    // One-way ANOVA: each ROW is a group; 1x4 [F, p, df_b, df_w].
    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    ASSERT_GT(interp.state().matrices.count("an"), 0u);
    const auto& an = interp.state().matrices.at("an");
    ASSERT_EQ(an.rows(), 1u);
    ASSERT_EQ(an.cols(), 4u);
    EXPECT_GT(an(0, 0), 0.0);
    EXPECT_LT(an(0, 1), 0.05);
    EXPECT_NEAR(an(0, 2), 2.0, 1e-12);
    EXPECT_NEAR(an(0, 3), 6.0, 1e-12);

    // Wilcoxon signed-rank: clear paired shift; 1x4 [W, z, p, n_eff].
    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
    const auto& ws = interp.state().matrices.at("ws");
    ASSERT_EQ(ws.rows(), 1u);
    ASSERT_EQ(ws.cols(), 4u);
    EXPECT_TRUE(std::isfinite(ws(0, 0)));
    EXPECT_TRUE(std::isfinite(ws(0, 1)));
    EXPECT_LT(ws(0, 2), 0.05);
    EXPECT_NEAR(ws(0, 3), 8.0, 1e-12);
}

TEST(ReplCommandsTest, wave257_stats_infer_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_friedman(data)");
    expect_contains(interp, "help", "stats_ks_2sample(a,b)");
    expect_contains(interp, "help", "stats_jarque_bera(x)");
    expect_contains(interp, "help", "stats_ljung_box(x,max_lag)");

    // Friedman: hand-computed 4 blocks × 3 treatments → chi2=1.5, df=2, p=exp(-0.75).
    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    ASSERT_GT(interp.state().matrices.count("fr"), 0u);
    const auto& fr = interp.state().matrices.at("fr");
    ASSERT_EQ(fr.rows(), 1u);
    ASSERT_EQ(fr.cols(), 3u);
    EXPECT_NEAR(fr(0, 0), 1.5, 1e-9);
    EXPECT_NEAR(fr(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(fr(0, 2), std::exp(-0.75), 1e-9);

    // Two-sample KS: fully separated samples → large D, small p; 1x2 [D, p].
    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    ASSERT_GT(interp.state().matrices.count("ks"), 0u);
    const auto& ks = interp.state().matrices.at("ks");
    ASSERT_EQ(ks.rows(), 1u);
    ASSERT_EQ(ks.cols(), 2u);
    EXPECT_GT(ks(0, 0), 0.9);
    EXPECT_LT(ks(0, 1), 0.05);

    // Jarque-Bera: heavily skewed exponential-like sample; 1x3 [JB, df, p].
    expect_ok(interp,
              "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 12; 14; 16; 18; 20; 25; 30; 40; 50; "
              "60; 80; 100; 150; 200; 300; 400; 500; 700; 900; 1200]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
    const auto& jb = interp.state().matrices.at("jb");
    ASSERT_EQ(jb.rows(), 1u);
    ASSERT_EQ(jb.cols(), 3u);
    EXPECT_GT(jb(0, 0), 5.0);
    EXPECT_NEAR(jb(0, 1), 2.0, 1e-12);
    EXPECT_LT(jb(0, 2), 0.05);

    // Ljung-Box: cumulative-sum random walk (same as StatsExtTest.ljung_box_cumulative_sum);
    // returns 1x3 [Q, df, p]. Pattern repeats every 3: -0.5, -0.5, 0 (80 samples).
    {
        std::string lbx = "lbx = [";
        double sum = 0.0;
        for (int i = 0; i < 80; ++i) {
            sum += ((i % 3) - 1) * 0.5;
            if (i > 0) {
                lbx += "; ";
            }
            lbx += std::to_string(sum);
        }
        lbx += "]";
        expect_ok(interp, lbx);
    }
    expect_ok(interp, "lb = stats_ljung_box(lbx, 8)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    const auto& lb = interp.state().matrices.at("lb");
    ASSERT_EQ(lb.rows(), 1u);
    ASSERT_EQ(lb.cols(), 3u);
    EXPECT_GT(lb(0, 0), 30.0);
    EXPECT_NEAR(lb(0, 1), 8.0, 1e-12);
    EXPECT_LT(lb(0, 2), 0.01);
}

TEST(ReplCommandsTest, wave258_stats_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_partial_correlation(x,y,z)");
    expect_contains(interp, "help", "stats_weighted_mean(x,w)");
    expect_contains(interp, "help", "stats_trimmed_mean(x,frac)");
    expect_contains(interp, "help", "stats_arfit(x,p)");
    expect_contains(interp, "help", "stats_multiple_regression(X,y)");

    // Partial correlation with z uncorrelated to x,y -> ~ Pearson corr(x,y) = 1.
    expect_ok(interp, "px = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "py = [2; 4; 6; 8; 10; 12; 14; 16]");
    expect_ok(interp, "pz = [1; -1; 1; -1; 1; -1; 1; -1]");
    expect_ok(interp, "pc = stats_partial_correlation(px, py, pz)");
    ASSERT_GT(interp.state().scalars.count("pc"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("pc"), 1.0, 0.05);

    // Weighted mean: weights [1,1,1,1,1] on 1..5 => 3.
    expect_ok(interp, "wm = stats_weighted_mean([1; 2; 3; 4; 5], [1; 1; 1; 1; 1])");
    ASSERT_GT(interp.state().scalars.count("wm"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("wm"), 3.0, 1e-12);

    // Trimmed mean: [1..10], frac=0.1 drops one each end -> mean of 2..9 = 5.5.
    expect_ok(interp, "tm = stats_trimmed_mean([1; 2; 3; 4; 5; 6; 7; 8; 9; 10], 0.1)");
    ASSERT_GT(interp.state().scalars.count("tm"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("tm"), 5.5, 1e-12);

    // AR(1) on short series: px1 coefficient column, finite.
    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    ASSERT_GT(interp.state().matrices.count("phi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("phi").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("phi")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("phi")(1, 0)));

    // Multiple regression: y = 1 + 2*x1 + 3*x2; X includes intercept column.
    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    ASSERT_GT(interp.state().matrices.count("beta"), 0u);
    EXPECT_EQ(interp.state().matrices.at("beta").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("beta").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
    EXPECT_NEAR(interp.state().matrices.at("beta")(1, 0), 2.0, 0.01);
    EXPECT_NEAR(interp.state().matrices.at("beta")(2, 0), 3.0, 0.01);
}

TEST(ReplCommandsTest, wave261_stats_weighted_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_weighted_variance(x,w)");
    expect_contains(interp, "help", "stats_weighted_correlation(x,y,w)");
    expect_contains(interp, "help", "stats_bootstrap_mean(x[,n_boot[,seed]])");

    // Uniform weights: weighted variance matches sample variance.
    expect_ok(interp, "sx = [2; 4; 6; 8; 10]");
    expect_ok(interp, "sw = [1; 1; 1; 1; 1]");
    expect_ok(interp, "wv = stats_weighted_variance(sx, sw)");
    expect_ok(interp, "v = stats_var(sx)");
    ASSERT_GT(interp.state().scalars.count("wv"), 0u);
    ASSERT_GT(interp.state().scalars.count("v"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("wv"), interp.state().scalars.at("v"), 1e-12);

    // Uniform weights: weighted correlation matches Pearson correlation.
    expect_ok(interp, "cx = [1; 2; 3; 4; 5]");
    expect_ok(interp, "cy = [2; 4; 6; 8; 10]");
    expect_ok(interp, "wc = stats_weighted_correlation(cx, cy, sw)");
    expect_ok(interp, "c = stats_correlation(cx, cy)");
    ASSERT_GT(interp.state().scalars.count("wc"), 0u);
    ASSERT_GT(interp.state().scalars.count("c"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("wc"), interp.state().scalars.at("c"), 1e-12);

    // Fixed-seed bootstrap mean is deterministic.
    expect_ok(interp, "bx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "bm1 = stats_bootstrap_mean(bx, 500, 42)");
    expect_ok(interp, "bm2 = stats_bootstrap_mean(bx, 500, 42)");
    ASSERT_GT(interp.state().scalars.count("bm1"), 0u);
    ASSERT_GT(interp.state().scalars.count("bm2"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bm1"), interp.state().scalars.at("bm2"), 1e-15);
}

TEST(ReplCommandsTest, wave260_stats_vif) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_vif(X,j)");
    expect_contains(interp, "help", "stats_variance_inflation_factor");

    // Orthogonal columns (constant vs ±1): auxiliary R^2 ~ 0 => VIF near 1.
    expect_ok(interp, "Xo = [1, 1; 1, -1; 1, 1; 1, -1; 1, 1; 1, -1]");
    expect_ok(interp, "v0 = stats_vif(Xo, 0)");
    ASSERT_GT(interp.state().scalars.count("v0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v0"), 1.0, 0.2);

    expect_ok(interp, "v1 = stats_variance_inflation_factor(Xo, 1)");
    ASSERT_GT(interp.state().scalars.count("v1"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v1"), 1.0, 0.2);
    EXPECT_NEAR(interp.state().scalars.at("v0"), interp.state().scalars.at("v1"), 1e-12);

    // Near-collinear third column -> large VIF.
    expect_ok(interp,
              "Xc = [1, 2, 3.00000001; 2, 3, 5.00000002; 3, 4, 7.00000003; "
              "4, 5, 9.00000004; 5, 6, 11.00000005]");
    expect_ok(interp, "vc = stats_vif(Xc, 2)");
    ASSERT_GT(interp.state().scalars.count("vc"), 0u);
    EXPECT_GT(interp.state().scalars.at("vc"), 100.0);
}

TEST(ReplCommandsTest, wave257_stats_variance) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_levene(G)");
    expect_contains(interp, "help", "stats_bartlett(G)");
    expect_contains(interp, "help", "stats_fligner(G)");

    // Unequal-variance groups (row = group); Levene 1x4 [F, p, df_b, df_w].
    expect_ok(interp, "G = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(G)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    const auto& lv = interp.state().matrices.at("lv");
    ASSERT_EQ(lv.rows(), 1u);
    ASSERT_EQ(lv.cols(), 4u);
    EXPECT_GT(lv(0, 0), 0.0);
    EXPECT_LT(lv(0, 1), 0.05);
    EXPECT_NEAR(lv(0, 2), 2.0, 1e-12);
    EXPECT_NEAR(lv(0, 3), 9.0, 1e-12);

    // Bartlett 1x3 [chi2, df, p].
    expect_ok(interp, "bt = stats_bartlett(G)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    const auto& bt = interp.state().matrices.at("bt");
    ASSERT_EQ(bt.rows(), 1u);
    ASSERT_EQ(bt.cols(), 3u);
    EXPECT_GT(bt(0, 0), 0.0);
    EXPECT_NEAR(bt(0, 1), 2.0, 1e-12);
    EXPECT_LT(bt(0, 2), 0.05);

    // Fligner-Killeen 1x3 [chi2, df, p].
    expect_ok(interp, "fk = stats_fligner(G)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);
    const auto& fk = interp.state().matrices.at("fk");
    ASSERT_EQ(fk.rows(), 1u);
    ASSERT_EQ(fk.cols(), 3u);
    EXPECT_GT(fk(0, 0), 0.0);
    EXPECT_NEAR(fk(0, 1), 2.0, 1e-12);
    EXPECT_LT(fk(0, 2), 0.05);
}

TEST(ReplCommandsTest, wave256_image_transform) {
    Interpreter interp;
    expect_contains(interp, "help", "imflip(M,horizontal)");
    expect_contains(interp, "help", "imrotate90(M)");
    expect_contains(interp, "help", "threshold_binary(M,t)");
    expect_contains(interp, "help", "adapthisteq(M)");

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("F").cols(), 2u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "R = imrotate90(M)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 2u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("R")(1, 0), 0.0);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(1, 1), 1.0);

    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("A").cols(), 2u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("A")(0, 0)));
}

TEST(ReplCommandsTest, wave257_prob_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "prob_lognormal_pdf(x,mu,sigma)");
    expect_contains(interp, "help", "prob_weibull_cdf(x,lambda,k)");
    expect_contains(interp, "help", "prob_laplace_ppf(p,mu,b)");
    expect_contains(interp, "help", "prob_logistic_cdf(x,mu,s)");
    expect_contains(interp, "help", "prob_gumbel_ppf(p,mu,beta)");
    expect_contains(interp, "help", "prob_cauchy_pdf(x,x0,gamma)");
    expect_contains(interp, "help", "prob_pareto_cdf(x,x_m,alpha)");
    expect_contains(interp, "help", "prob_rayleigh_cdf(x,sigma)");
    expect_contains(interp, "help", "prob_gamma_cdf(x,shape,scale)");
    expect_contains(interp, "help", "prob_beta_cdf(x,alpha,beta)");
    expect_contains(interp, "help", "prob_f_cdf(x,d1,d2)");

    // Closed-form spot checks (assignment path).
    expect_ok(interp, "lnq = prob_lognormal_ppf(0.5, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lnq"), 1.0, 1e-9);  // median = exp(mu)

    expect_ok(interp, "wc = prob_weibull_cdf(1, 1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("wc"), 1.0 - std::exp(-1.0), 1e-9);

    expect_ok(interp, "lp = prob_laplace_pdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), 0.5, 1e-9);  // 1/(2b)

    expect_ok(interp, "lc = prob_logistic_cdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 0.5, 1e-9);

    expect_ok(interp, "gq = prob_gumbel_ppf(0.5, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("gq"), -std::log(std::log(2.0)), 1e-9);

    expect_ok(interp, "cp = prob_cauchy_pdf(0, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), 1.0 / 3.14159265358979323846, 1e-9);

    expect_ok(interp, "pc = prob_pareto_cdf(2, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), 0.5, 1e-9);  // 1-(x_m/x)^alpha

    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), 1.0 - std::exp(-0.5), 1e-9);

    expect_ok(interp, "bc = prob_beta_cdf(0.5, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bc"), 0.5, 1e-9);  // Beta(1,1)=U[0,1]

    expect_ok(interp, "gc = prob_gamma_cdf(0, 2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("gc"), 0.0, 1e-9);

    const double f_ref = ms::f_cdf(1.0, 5.0, 5.0);
    expect_ok(interp, "fc = prob_f_cdf(1, 5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("fc"), f_ref, 1e-9);

    // Direct-call path smoke.
    expect_contains(interp, "prob_laplace_cdf(0, 0, 1)", "0.5");
    expect_contains(interp, "prob_cauchy_cdf(0, 0, 1)", "0.5");
}

TEST(ReplCommandsTest, wave257_image_segment) {
    Interpreter interp;
    expect_contains(interp, "help", "label_components(B)");
    expect_contains(interp, "help", "watershed(G,M)");
    expect_contains(interp, "help", "slic(M,K");

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("L")(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("L")(0, 1), interp.state().matrices.at("L")(1, 2));
    EXPECT_NE(interp.state().matrices.at("L")(0, 1), interp.state().matrices.at("L")(3, 0));

    expect_ok(interp, "G = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; "
                      "0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G, Mk)");
    ASSERT_GT(interp.state().matrices.count("W"), 0u);
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("W").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("W")(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("W")(0, 0), 1.0);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 4u);
    EXPECT_GE(interp.state().matrices.at("S")(0, 0), 1.0);

    expect_ok(interp, "S2 = slic(RGB, 4, 10)");
    ASSERT_GT(interp.state().matrices.count("S2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S2").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("S2").cols(), 4u);
}

TEST(ReplCommandsTest, wave257_image_hough) {
    Interpreter interp;
    expect_contains(interp, "help", "hough_lines(M[,edge])");
    expect_contains(interp, "help", "hough_circles(M[,r_min,r_max])");
    expect_contains(interp, "help", "harris(M[,k[,thr]])");
    expect_contains(interp, "help", "shi_tomasi(M,n[,q])");

    // 10x10 horizontal edge at row 5; vote_threshold=5 so short lines still peak.
    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("L").rows(), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 1), 1.5707963267948966, 0.05);  // ~pi/2
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 5.0, 1.5);

    // Blank image: no circles.
    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 0u);

    // Bright quadrant corner -> Harris / Shi-Tomasi keypoints.
    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("H").rows(), 0u);
    EXPECT_GT(interp.state().matrices.at("H")(0, 2), 0.0);

    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("S").rows(), 0u);
    EXPECT_LE(interp.state().matrices.at("S").rows(), 5u);
}

TEST(ReplCommandsTest, wave258_image_morph_hist) {
    Interpreter interp;
    expect_contains(interp, "help", "imtophat(M[,k])");
    expect_contains(interp, "help", "imbothat(M[,k])");
    expect_contains(interp, "help", "imadjust(M,in_lo,in_hi[,out_lo,out_hi])");
    expect_contains(interp, "help", "imhist(M[,nbins])");

    // Bright center pixel on dark background — top-hat keeps the peak.
    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);
    EXPECT_GE(interp.state().matrices.at("T")(1, 1), 0.0);

    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
    EXPECT_GE(interp.state().matrices.at("B")(0, 0), 0.0);

    // Identity stretch then brighten lower bound.
    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 1), 0.8, 1e-5);

    expect_ok(interp, "A2 = imadjust(G, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("A2"), 0u);
    EXPECT_GT(interp.state().matrices.at("A2")(0, 0), interp.state().matrices.at("A")(0, 0));

    // Four distinct levels into 4 bins → counts [2,1,1,0] (same as ImageHist.CustomBinCount).
    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 1u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(3, 0), 0.0);
}

TEST(ReplCommandsTest, wave258_gray2rgb_and_impad) {
    Interpreter interp;
    expect_contains(interp, "help", "gray2rgb(M)");
    expect_contains(interp, "help", "impad(M,pad");

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    ASSERT_GT(interp.state().matrices.count("RGB2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("RGB2").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("RGB2")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("RGB2")(0, 1), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("RGB2")(0, 2), 0.299, 1e-6);

    // Use 0..1 intensities so matrix↔Image round-trip preserves values.
    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("P").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("P")(0, 0), 0.0);
    EXPECT_NEAR(interp.state().matrices.at("P")(1, 1), 0.1, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("P")(2, 2), 0.4, 1e-6);
}

TEST(ReplCommandsTest, wave259_numthy_factor) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor(n)");

    expect_ok(interp, "f = numthy_factor(12)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& factors = interp.state().matrices.at("f");
    EXPECT_EQ(factors.rows(), 3u);
    EXPECT_EQ(factors.cols(), 1u);
    EXPECT_NEAR(factors(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(2, 0), 3.0, 1e-9);

    expect_ok(interp, "p = numthy_factor(17)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 17.0, 1e-9);
}

TEST(ReplCommandsTest, wave260_numthy_divisors) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_divisors(n)");
    expect_contains(interp, "help", "numthy_num_divisors(n)");
    expect_contains(interp, "help", "numthy_sum_divisors(n)");

    expect_ok(interp, "d = numthy_divisors(12)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    const auto& divs = interp.state().matrices.at("d");
    EXPECT_EQ(divs.rows(), 6u);
    EXPECT_EQ(divs.cols(), 1u);
    const double expected[] = {1, 2, 3, 4, 6, 12};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(divs(i, 0), expected[i], 1e-9);
    }

    expect_ok(interp, "tau = numthy_num_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("tau"), 6.0, 1e-9);

    expect_ok(interp, "sigma = numthy_sum_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("sigma"), 28.0, 1e-9);

    expect_ok(interp, "p = numthy_divisors(17)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("p")(1, 0), 17.0, 1e-9);
}

TEST(ReplCommandsTest, wave261_numthy_factor_exp_farey_carmichael) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor_exp(n)");
    expect_contains(interp, "help", "numthy_farey(n)");
    expect_contains(interp, "help", "numthy_is_carmichael(n)");

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    ASSERT_GT(interp.state().matrices.count("fe"), 0u);
    const auto& factor_exp = interp.state().matrices.at("fe");
    EXPECT_EQ(factor_exp.rows(), 2u);
    EXPECT_EQ(factor_exp.cols(), 2u);
    EXPECT_NEAR(factor_exp(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factor_exp(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(factor_exp(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(factor_exp(1, 1), 1.0, 1e-9);

    expect_ok(interp, "f = numthy_farey(4)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& farey = interp.state().matrices.at("f");
    EXPECT_EQ(farey.rows(), 7u);
    EXPECT_EQ(farey.cols(), 2u);
    EXPECT_NEAR(farey(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(farey(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(farey(6, 0), 1.0, 1e-9);
    EXPECT_NEAR(farey(6, 1), 1.0, 1e-9);

    expect_contains(interp, "numthy_is_carmichael(561)", "1");
    expect_contains(interp, "numthy_is_carmichael(97)", "0");
}

TEST(ReplCommandsTest, wave262_numthy_stern_lucas_order_lambda) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_stern_brocot(n)");
    expect_contains(interp, "help", "numthy_lucas_sequence(k,P,Q)");
    expect_contains(interp, "help", "numthy_multiplicative_order(a,n)");
    expect_contains(interp, "help", "numthy_carmichael_lambda(n)");

    expect_ok(interp, "sb = numthy_stern_brocot(7)");
    ASSERT_GT(interp.state().matrices.count("sb"), 0u);
    const auto& stern = interp.state().matrices.at("sb");
    EXPECT_EQ(stern.rows(), 7u);
    EXPECT_EQ(stern.cols(), 2u);
    EXPECT_NEAR(stern(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(stern(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(stern(6, 0), 3.0, 1e-9);
    EXPECT_NEAR(stern(6, 1), 1.0, 1e-9);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    ASSERT_GT(interp.state().matrices.count("lu"), 0u);
    const auto& lucas = interp.state().matrices.at("lu");
    EXPECT_EQ(lucas.rows(), 1u);
    EXPECT_EQ(lucas.cols(), 2u);
    EXPECT_NEAR(lucas(0, 0), 5.0, 1e-9);
    EXPECT_NEAR(lucas(0, 1), 11.0, 1e-9);

    expect_ok(interp, "ord = numthy_multiplicative_order(3, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ord"), 6.0, 1e-9);
    expect_contains(interp, "numthy_multiplicative_order(3, 7)", "6");

    expect_ok(interp, "lam = numthy_carmichael_lambda(15)");
    EXPECT_NEAR(interp.state().scalars.at("lam"), 4.0, 1e-9);
    expect_contains(interp, "numthy_carmichael_lambda(15)", "4");
}

TEST(ReplCommandsTest, wave263_numthy_pell_quadratic) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_pell_solve(D)");
    expect_contains(interp, "help", "numthy_quadratic_residues(p)");

    expect_ok(interp, "sol = numthy_pell_solve(61)");
    ASSERT_GT(interp.state().matrices.count("sol"), 0u);
    const auto& pell = interp.state().matrices.at("sol");
    EXPECT_EQ(pell.rows(), 1u);
    EXPECT_EQ(pell.cols(), 2u);
    const double x = pell(0, 0);
    const double y = pell(0, 1);
    EXPECT_NEAR(x, 1766319049.0, 1e-3);
    EXPECT_NEAR(y, 226153980.0, 1e-3);
    EXPECT_NEAR(x * x - 61.0 * y * y, 1.0, 1e6);

    expect_ok(interp, "qr = numthy_quadratic_residues(7)");
    ASSERT_GT(interp.state().matrices.count("qr"), 0u);
    const auto& residues = interp.state().matrices.at("qr");
    EXPECT_EQ(residues.rows(), 3u);
    EXPECT_EQ(residues.cols(), 1u);
    EXPECT_NEAR(residues(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(residues(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(residues(2, 0), 4.0, 1e-9);
}

TEST(ReplCommandsTest, wave264_numthy_cornacchia) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_cornacchia(d,p)");

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(sol(0, 0) * sol(0, 0) + 1.0 * sol(0, 1) * sol(0, 1), 5.0, 1e-9);

    expect_ok(interp, "xy13 = numthy_cornacchia(1, 13)");
    ASSERT_GT(interp.state().matrices.count("xy13"), 0u);
    const auto& s13 = interp.state().matrices.at("xy13");
    EXPECT_EQ(s13.rows(), 1u);
    EXPECT_EQ(s13.cols(), 2u);
    EXPECT_NEAR(s13(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(s13(0, 1), 2.0, 1e-9);
}

TEST(ReplCommandsTest, wave258_radon_iradon) {
    Interpreter interp;
    expect_contains(interp, "help", "radon(M,theta)");
    expect_contains(interp, "help", "iradon(S,theta)");

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 8u);
    for (size_t r = 0; r < interp.state().matrices.at("S").rows(); ++r) {
        for (size_t c = 0; c < interp.state().matrices.at("S").cols(); ++c) {
            EXPECT_TRUE(std::isfinite(interp.state().matrices.at("S")(r, c)));
        }
    }

    expect_ok(interp, "R = iradon(S, Th)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 8u);
    for (size_t r = 0; r < interp.state().matrices.at("R").rows(); ++r) {
        for (size_t c = 0; c < interp.state().matrices.at("R").cols(); ++c) {
            EXPECT_TRUE(std::isfinite(interp.state().matrices.at("R")(r, c)));
        }
    }
}

TEST(ReplCommandsTest, wave259_combo_binomial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_binomial(n,k)");

    expect_ok(interp, "bin = combo_binomial(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("bin"), 10.0, 1e-9);

    expect_contains(interp, "combo_binomial(5, 2)", "10");
}

TEST(ReplCommandsTest, wave259_combo_bell_num) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_bell_num(n)");

    expect_ok(interp, "bn = combo_bell_num(4)");
    EXPECT_NEAR(interp.state().scalars.at("bn"), 15.0, 1e-9);

    expect_contains(interp, "combo_bell_num(4)", "15");
}

TEST(ReplCommandsTest, wave261_combo_eulerian) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_eulerian(n,k)");

    expect_ok(interp, "eul = combo_eulerian(4, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eul"), 11.0, 1e-9);

    expect_contains(interp, "combo_eulerian(4, 1)", "11");
}

TEST(ReplCommandsTest, wave261_combo_gray_code) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_gray_code(n)");

    expect_ok(interp, "g2 = combo_gray_code(2)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("g2").cols(), 2u);
}

TEST(ReplCommandsTest, wave261_combo_dyck_paths) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_dyck_paths(n)");

    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    ASSERT_GT(interp.state().matrices.count("d3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("d3").cols(), 6u);
}

TEST(ReplCommandsTest, wave262_combo_necklaces) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_necklaces(n,k)");

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    ASSERT_GT(interp.state().matrices.count("neck"), 0u);
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("neck").cols(), 2u);
}

TEST(ReplCommandsTest, wave263_combo_bracelets_lyndon) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_bracelets(n,k)");
    expect_contains(interp, "help", "combo_lyndon_words(n,k)");

    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("br").cols(), 3u);

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("lw").cols(), 3u);
}

TEST(ReplCommandsTest, wave262_combo_de_bruijn_sequence) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_de_bruijn_sequence(k,n)");

    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    ASSERT_GT(interp.state().matrices.count("db"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("db").cols(), 1u);
}

TEST(ReplCommandsTest, wave259_imgradient_morph) {
    Interpreter interp;
    expect_contains(interp, "help", "imgradient_morph(M[,k])");

    // Bright center on dark background — gradient responds at edges.
    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "G = imgradient_morph(M)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("G").cols(), 5u);
    EXPECT_GE(interp.state().matrices.at("G")(1, 2), 0.0);

    // Matches imdilate - imerode composition.
    expect_ok(interp, "D = imdilate(M, 3)");
    expect_ok(interp, "E = imerode(M, 3)");
    expect_ok(interp, "G2 = imgradient_morph(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("G2")(1, 2),
                interp.state().matrices.at("D")(1, 2) - interp.state().matrices.at("E")(1, 2),
                1e-5);
}

TEST(ReplCommandsTest, wave259_hess_schur) {
    Interpreter interp;
    expect_contains(interp, "help", "hess(A)");
    expect_contains(interp, "help", "T, Q = schur(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "hess(A)", "H =");
    expect_ok(interp, "H = hess(A)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);

    expect_contains(interp, "schur(A)", "T =");
    expect_ok(interp, "T = schur(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Ts, Qs = schur(A)");
    ASSERT_GT(interp.state().matrices.count("Ts"), 0u);
    ASSERT_GT(interp.state().matrices.count("Qs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ts").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Qs").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Qs").cols(), 3u);
}

TEST(ReplCommandsTest, wave260_bidiag) {
    Interpreter interp;
    expect_contains(interp, "help", "U, B, V = bidiag(A)");
    expect_contains(interp, "help", "bidiag(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "bidiag(A)", "B =");
    expect_ok(interp, "B = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "U, Bd, V = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("U"), 0u);
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("U").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Bd").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 3u);
}

TEST(ReplCommandsTest, wave260_solve_sylvester) {
    Interpreter interp;
    expect_contains(interp, "help", "solve_sylvester(A,B,C)");

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    // A*X + X*B with X=[1,2;3,4] => C=[4,10;15,24]
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 1), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(1, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(1, 1), 4.0, 1e-8);
}

TEST(ReplCommandsTest, wave260_minres) {
    Interpreter interp;
    expect_contains(interp, "help", "minres(A");

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xm").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("xm")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("xm")(1, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("xm")(2, 0)));
}

TEST(ReplCommandsTest, wave261_cg) {
    Interpreter interp;
    expect_contains(interp, "help", "cg(A");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xc").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xc").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xc")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xc")(2, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, wave261_gmres) {
    Interpreter interp;
    expect_contains(interp, "help", "gmres(A");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xg").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xg").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xg")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xg")(2, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, wave261_jacobi) {
    Interpreter interp;
    expect_contains(interp, "help", "jacobi(A");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xj").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xj")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xj")(2, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, wave261_eig) {
    Interpreter interp;
    expect_contains(interp, "help", "D, V = eig(A)");
    expect_contains(interp, "help", "eig(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "eig(A)", "eigenvalues:");
    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 1u);

    expect_ok(interp, "De, Ve = eig(A)");
    ASSERT_GT(interp.state().matrices.count("De"), 0u);
    ASSERT_GT(interp.state().matrices.count("Ve"), 0u);
    EXPECT_EQ(interp.state().matrices.at("De").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Ve").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Ve").cols(), 3u);
}

TEST(ReplCommandsTest, wave261_ldl) {
    Interpreter interp;
    expect_contains(interp, "help", "L, D = ldl(A)");
    expect_contains(interp, "help", "ldl(A)");

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_contains(interp, "ldl(S)", "L =");
    expect_contains(interp, "ldl(S)", "D =");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 1), 0.0, 1e-10);

    expect_ok(interp, "Ll, Dl = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("Ll"), 0u);
    ASSERT_GT(interp.state().matrices.count("Dl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ll").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Dl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Dl").cols(), 1u);

    expect_ok(interp, "Lp, Dp, Pp = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pp").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Pp").cols(), 2u);
}

TEST(ReplCommandsTest, wave262_combo_motzkin_paths) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_motzkin_paths(n)");

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    ASSERT_GT(interp.state().matrices.count("mp3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("mp3").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("mp3")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp3")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp3")(0, 2), -1.0, 1e-9);
}

TEST(ReplCommandsTest, wave262_combo_set_partitions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_set_partitions(n)");

    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    ASSERT_GT(interp.state().matrices.count("sp2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("sp2").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(1, 1), 0.0, 1e-9);

    expect_ok(interp, "sp3 = combo_set_partitions(3)");
    ASSERT_GT(interp.state().matrices.count("sp3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp3").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("sp3").cols(), 3u);
}

TEST(ReplCommandsTest, wave263_combo_restricted_involutions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_restricted_partitions(n,k)");
    expect_contains(interp, "help", "combo_involutions(n)");

    expect_ok(interp, "rp = combo_restricted_partitions(5,2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rp").rows(), 2u);
    {
        const auto& m = interp.state().matrices.at("rp");
        for (size_t r = 0; r < m.rows(); ++r) {
            double sum = 0.0;
            for (size_t c = 0; c < m.cols(); ++c) {
                sum += m(r, c);
            }
            EXPECT_NEAR(sum, 5.0, 1e-9);
        }
    }

    expect_ok(interp, "inv = combo_involutions(4)");
    EXPECT_NEAR(interp.state().scalars.at("inv"), 10.0, 1e-9);
    expect_contains(interp, "combo_involutions(4)", "10");
}

TEST(ReplCommandsTest, wave262_lsq) {
    Interpreter interp;
    expect_contains(interp, "help", "lsq(A");

    expect_ok(interp, "A = [0, 1; 1, 1; 2, 1; 3, 1]");
    expect_ok(interp, "b = [1; 3; 5; 7]");
    expect_contains(interp, "lsq(A, b)", "x =");
    expect_ok(interp, "x = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 2.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 1.0, 1e-5);
}

TEST(ReplCommandsTest, wave262_diag) {
    Interpreter interp;
    expect_contains(interp, "help", "diag(v)");

    expect_ok(interp, "v = [2; 3; 5]");
    expect_contains(interp, "diag(v)", "D =");
    expect_ok(interp, "D = diag(v)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(2, 2), 5.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 0), 0.0, 1e-12);
}

TEST(ReplCommandsTest, wave262_finance_bachelier_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bachelier_call(F,K,T,r,sigma)");

    // ATM closed form: e^{-rT} * sigma*sqrt(T) / sqrt(2*pi) at F=K=100, r=0.04, sigma=0.25, T=1.5
    expect_ok(interp, "bc = finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("bc"), 0.11503712918258642, 1e-6);

    expect_contains(interp, "finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)", "0.115");
}

TEST(ReplCommandsTest, wave262_finance_bachelier_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bachelier_put(F,K,T,r,sigma)");

    expect_ok(interp, "bp = finance_bachelier_put(100, 100, 1.5, 0.04, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 0.11503712918258642, 1e-6);

    expect_contains(interp, "finance_bachelier_put(100, 100, 1.5, 0.04, 0.25)", "0.115");
}

TEST(ReplCommandsTest, wave262_finance_vasicek_bond_price) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_vasicek_bond_price(r,a,b,sigma,tau)");

    expect_ok(interp, "vz = finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("vz"), 0.9512737476480422, 1e-6);

    expect_contains(interp, "finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, 1.0)", "0.951");
}

TEST(ReplCommandsTest, wave262_finance_cir_bond_price) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_cir_bond_price(r,a,b,sigma,tau)");

    expect_ok(interp, "cr = finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), 0.951243269844748, 1e-6);

    expect_contains(interp, "finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, 1.0)", "0.951");
}

TEST(ReplCommandsTest, wave262_finance_trinomial_option) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_trinomial_option(S,K,T,r,sigma,n_steps,is_call,is_american)");

    // European call should be close to the analytic BS call price.
    expect_ok(interp, "tri = finance_trinomial_option(100, 100, 1, 0.05, 0.2, 200, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("tri"), 10.450583572185565, 0.5);

    expect_contains(interp, "finance_trinomial_option(100, 100, 1, 0.05, 0.2, 200, 1, 0)", "\n");
}

TEST(ReplCommandsTest, wave262_poly_resultant) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_resultant(p,q)");

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "q = [10; -7; 1]");
    expect_ok(interp, "r = poly_resultant(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("r"), 0.0, 1e-6);

    expect_contains(interp, "poly_resultant([6; -5; 1], [10; -7; 1])", "0");
}

TEST(ReplCommandsTest, wave262_poly_discriminant) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_discriminant(p)");

    expect_ok(interp, "sq = [1; -2; 1]");
    expect_ok(interp, "d = poly_discriminant(sq)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 0.0, 1e-6);

    expect_contains(interp, "poly_discriminant([1; -2; 1])", "0");
}

TEST(ReplCommandsTest, wave263_special_voigt_airy) {
    Interpreter interp;
    expect_contains(interp, "help", "special_voigt(x,sigma,gamma)");
    expect_contains(interp, "help", "special_pseudo_voigt_auto(x,sigma,gamma)");
    expect_contains(interp, "help", "special_airy_ai(x)");

    constexpr double kPi = 3.14159265358979323846;
    const double voigt_ref = ms::voigt(0.0, 1.0, 0.0);
    EXPECT_NEAR(voigt_ref, 1.0 / std::sqrt(2.0 * kPi), 1e-9);
    expect_ok(interp, "y = special_voigt(0,1,0)");
    EXPECT_NEAR(interp.state().scalars.at("y"), voigt_ref, 1e-9);
    expect_contains(interp, "special_voigt(0,1,0)", std::to_string(voigt_ref));

    const double airy_ref = ms::airy_ai(0.0);
    EXPECT_NEAR(airy_ref, 0.355028053887817, 1e-9);
    expect_ok(interp, "z = special_airy_ai(0)");
    EXPECT_NEAR(interp.state().scalars.at("z"), airy_ref, 1e-9);
    expect_contains(interp, "special_airy_ai(0)", std::to_string(airy_ref));
}

TEST(ReplCommandsTest, wave263_poly_lagrange_newton) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_lagrange(xs,ys)");
    expect_contains(interp, "help", "poly_interp_newton(xs,ys)");

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "p = poly_lagrange(xs, ys)");
    expect_ok(interp, "v = poly_eval(p, 1)");
    EXPECT_NEAR(interp.state().scalars.at("v"), 2.0, 1e-6);

    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    expect_ok(interp, "vn = poly_eval(pn, 1)");
    EXPECT_NEAR(interp.state().scalars.at("vn"), 2.0, 1e-6);
}

TEST(ReplCommandsTest, wave263_finance_bl_portfolio) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bl_implied_returns(cov,w_mkt,delta)");
    expect_contains(interp, "help", "finance_bl_posterior_returns(pi,cov,P,Q,tau)");

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pi_bl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pi_bl").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(1, 0), 0.035, 1e-6);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("post").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("post")(0, 0), 0.075, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("post")(1, 0), 0.07625, 1e-6);
}

TEST(ReplCommandsTest, wave263_finance_merton_historical) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_merton_distance_to_default(V,sigma_v,D,r,T)");
    expect_contains(interp, "help", "finance_historical_var(returns,confidence)");
    expect_contains(interp, "help", "finance_historical_cvar(returns,confidence)");

    expect_ok(interp, "dd = finance_merton_distance_to_default(150, 0.20, 100, 0.05, 1)");
    EXPECT_NEAR(interp.state().scalars.at("dd"), 2.177325543255, 1e-6);
    expect_contains(interp, "finance_merton_distance_to_default(150, 0.20, 100, 0.05, 1)", "2.177");

    expect_ok(interp, "ret = [-0.20; -0.15; -0.10; -0.05; 0.0; 0.05; 0.10; 0.15; 0.20; 0.25]");
    expect_ok(interp, "hv = finance_historical_var(ret, 0.95)");
    EXPECT_NEAR(interp.state().scalars.at("hv"), 0.20, 1e-6);
    expect_contains(interp, "finance_historical_var(ret, 0.95)", "0.2");

    expect_ok(interp, "hc = finance_historical_cvar(ret, 0.95)");
    EXPECT_NEAR(interp.state().scalars.at("hc"), 0.20, 1e-6);
    expect_contains(interp, "finance_historical_cvar(ret, 0.95)", "0.2");
}

TEST(ReplCommandsTest, wave264_poly_roots_fit_gcd) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_roots(p)");
    expect_contains(interp, "help", "poly_fit(xs,ys,degree)");
    expect_contains(interp, "help", "poly_interp_hermite(xs,ys,dys)");
    expect_contains(interp, "help", "poly_gcd(a,b)");
    expect_contains(interp, "help", "poly_squarefree(p)");

    // (x-2)(x-3) = x^2 - 5x + 6
    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rts").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rts").cols(), 2u);
    {
        std::vector<double> reals = {interp.state().matrices.at("rts")(0, 0),
                                     interp.state().matrices.at("rts")(1, 0)};
        std::sort(reals.begin(), reals.end());
        EXPECT_NEAR(reals[0], 2.0, 1e-5);
        EXPECT_NEAR(reals[1], 3.0, 1e-5);
        EXPECT_NEAR(interp.state().matrices.at("rts")(0, 1), 0.0, 1e-5);
        EXPECT_NEAR(interp.state().matrices.at("rts")(1, 1), 0.0, 1e-5);
    }

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_EQ(interp.state().matrices.at("c").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("c")(1, 0), 2.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    expect_ok(interp, "vh = poly_eval(ph, 2)");
    EXPECT_NEAR(interp.state().scalars.at("vh"), 5.0, 1e-6);

    // gcd((x-2)(x-3), (x-2)(x-5)) ~ (x-2)
    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
    ASSERT_GE(interp.state().matrices.at("g").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("g")(0, 0), -2.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    // square-free part of (x-2)^2(x-3)
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    expect_ok(interp, "v2 = poly_eval(sf, 2)");
    expect_ok(interp, "v3 = poly_eval(sf, 3)");
    EXPECT_NEAR(interp.state().scalars.at("v2"), 0.0, 1e-4);
    EXPECT_NEAR(interp.state().scalars.at("v3"), 0.0, 1e-4);
}
