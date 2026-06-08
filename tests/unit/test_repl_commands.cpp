#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <string>

#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"
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
