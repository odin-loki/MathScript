#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

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

TEST(ReplCommandsTest, source_8_missing_does_not_crash) {
    Interpreter interp;
    const auto result = interp.execute("source 8");
    EXPECT_FALSE(result.has_value());
}

TEST(ReplCommandsTest, source_self_is_circular) {
    const auto path =
        (std::filesystem::temp_directory_path() / "mathscript_src_cycle.ms").string();
    {
        std::ofstream out(path);
        out << "source " << path << "\n";
    }

    Interpreter interp;
    expect_error_contains(interp, "source " + path, "circular source");
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, run_file_nested_source_is_capped) {
    const auto dir = std::filesystem::temp_directory_path() / "mathscript_src_nest";
    std::filesystem::create_directories(dir);
    std::vector<std::filesystem::path> files;
    for (int i = 0; i < 9; ++i) {
        files.push_back(dir / ("f" + std::to_string(i) + ".ms"));
    }
    for (int i = 0; i < 8; ++i) {
        std::ofstream out(files[static_cast<size_t>(i)]);
        out << "source " << files[static_cast<size_t>(i + 1)].string() << "\n";
    }
    {
        std::ofstream out(files.back());
        out << "x = 1\n";
    }

    Interpreter interp;
    expect_error_contains(interp, "source " + files.front().string(), "nesting too deep");
    for (const auto& path : files) {
        std::filesystem::remove(path);
    }
    std::filesystem::remove(dir);
}

TEST(ReplCommandsTest, run_file_one_nested_source_ok) {
    const auto dir = std::filesystem::temp_directory_path() / "mathscript_src_one_nest";
    std::filesystem::create_directories(dir);
    const auto inner = dir / "inner.ms";
    const auto outer = dir / "outer.ms";
    {
        std::ofstream out(inner);
        out << "x = 7\n";
    }
    {
        std::ofstream out(outer);
        out << "source " << inner.string() << "\n";
    }

    Interpreter interp;
    expect_contains(interp, "source " + outer.string(), "ran script from");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 7.0);
    std::filesystem::remove(inner);
    std::filesystem::remove(outer);
    std::filesystem::remove(dir);
}

TEST(ReplCommandsTest, run_file_rejects_oversized_script) {
    const auto path =
        (std::filesystem::temp_directory_path() / "mathscript_src_too_big.ms").string();
    {
        std::ofstream out(path, std::ios::binary);
        out << std::string(256 * 1024 + 1, 'x');
    }

    Interpreter interp;
    expect_error_contains(interp, "source " + path, "too large");
    std::filesystem::remove(path);
}

TEST(ReplCommandsTest, run_file_rejects_overlong_line) {
    const auto path =
        (std::filesystem::temp_directory_path() / "mathscript_src_long_line.ms").string();
    {
        std::ofstream out(path);
        out << "x = " << std::string(8200, '1') << "\n";
    }

    Interpreter interp;
    expect_error_contains(interp, "source " + path, "line too long");
    std::filesystem::remove(path);
}

#ifndef _WIN32
TEST(ReplCommandsTest, run_file_rejects_device_node) {
    Interpreter interp;
    expect_error_contains(interp, "source /dev/zero", "cannot open");
}
#endif

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

TEST(ReplCommandsTest, audit_w312_rle_roundtrip) {
    Interpreter interp;
    expect_contains(interp, "help", "rle_encode_vec(M)");
    expect_contains(interp, "help", "rle_decode_vec(M)");

    // Non-trivial payload with runs and unique bytes (ASCII "ABCDAAAB").
    expect_ok(interp, "raw = [65; 66; 67; 68; 65; 65; 65; 66]");
    expect_ok(interp, "enc = rle_encode_vec(raw)");
    expect_ok(interp, "dec = rle_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    const auto& restored = interp.state().matrices.at("dec");
    ASSERT_EQ(restored.rows(), 8u);
    ASSERT_EQ(restored.cols(), 1u);
    const double expected[] = {65, 66, 67, 68, 65, 65, 65, 66};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(restored(i, 0), expected[i]);
    }
}

TEST(ReplCommandsTest, audit_w312_lzw_roundtrip) {
    Interpreter interp;
    expect_contains(interp, "help", "lzw_encode_vec(M)");
    expect_contains(interp, "help", "lzw_decode_vec(C)");

    expect_ok(interp, "raw = [65; 66; 67; 68; 65; 65; 65; 66]");
    expect_ok(interp, "enc = lzw_encode_vec(raw)");
    expect_ok(interp, "dec = lzw_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    const auto& restored = interp.state().matrices.at("dec");
    ASSERT_EQ(restored.rows(), 8u);
    ASSERT_EQ(restored.cols(), 1u);
    const double expected[] = {65, 66, 67, 68, 65, 65, 65, 66};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(restored(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, audit_w314_bigint_lcm) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_lcm(a,b)");
    expect_ok(interp, "l = bigint_lcm(48, 18)");
    // lcm(48,18) = |48*18|/gcd(48,18) = 864/6 = 144
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("l"), 144.0);
}

TEST(ReplCommandsTest, audit_w314_bigint_is_even_odd) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_is_even(n)");
    expect_contains(interp, "help", "bigint_is_odd(n)");
    expect_ok(interp, "ev = bigint_is_even(4)");
    expect_ok(interp, "od = bigint_is_odd(4)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("ev"), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("od"), 0.0);
}

TEST(ReplCommandsTest, audit_w314_bigint_isqrt) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_isqrt(n)");
    expect_ok(interp, "r = bigint_isqrt(16)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("r"), 4.0);
}

TEST(ReplCommandsTest, audit_w314_bigint_pow_prime) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_pow(base,exp)");
    expect_contains(interp, "help", "bigint_is_prime(n)");
    expect_ok(interp, "p = bigint_pow(2, 10)");
    expect_ok(interp, "pr = bigint_is_prime(7)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("p"), 1024.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("pr"), 1.0);
}

TEST(ReplCommandsTest, audit_w314_cplx_cauchy_integral) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_cauchy_integral(z0re,z0im)");
    expect_ok(interp, "c0 = cplx_cauchy_integral(0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("c0"), 1.0, 0.2);
}

TEST(ReplCommandsTest, audit_w315_geo_dist2d) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist2d(x1,y1,x2,y2)");
    expect_ok(interp, "d = geo_dist2d(0, 0, 3, 4)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("d"), 5.0);
}

TEST(ReplCommandsTest, audit_w315_geo_convex_hull) {
    Interpreter interp;
    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
}

TEST(ReplCommandsTest, audit_w316_cplx_mobius_identity) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_mobius_re(a,b,c,d,zre,zim)");
    expect_ok(interp, "w = cplx_mobius_re(1, 0, 0, 1, 2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("w"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, audit_w317_graph_is_planar_path) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_planar(A)");
    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "p = graph_is_planar(A)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("p"), 1.0);
}

TEST(ReplCommandsTest, audit_w317_crypto_sha256_empty) {
    Interpreter interp;
    expect_contains(interp, "crypto_sha256(\"\")",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(ReplCommandsTest, audit_w318_combo_catalan) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_catalan(n)");
    expect_ok(interp, "c = combo_catalan(3)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("c"), 5.0);
}

TEST(ReplCommandsTest, audit_w318_ode_rk2_exponential) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_rk2(\"formula\",t0,y0,t_end,steps)");
    expect_ok(interp, "traj = ode_rk2(\"y\", 0, 1, 1, 200)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    const auto& traj = interp.state().matrices.at("traj");
    ASSERT_GE(traj.rows(), 2u);
    ASSERT_GE(traj.cols(), 2u);
    EXPECT_NEAR(traj(traj.rows() - 1, 1), std::exp(1.0), 1e-3);
}

TEST(ReplCommandsTest, bignum_ops) {
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

TEST(ReplCommandsTest, matrix_literal_negative) {
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

TEST(ReplCommandsTest, mtf_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "mtf_encode_vec(M)");

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, mtf_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "mtf_decode_vec(M)");

    expect_ok(interp, "B = [10, 12, 15, 20]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
}

TEST(ReplCommandsTest, bwt_primary_index) {
    Interpreter interp;
    expect_contains(interp, "help", "bwt_primary_index(M)");

    expect_ok(interp, "B = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "pi = bwt_primary_index(B)");
    const double pi = interp.state().scalars.at("pi");
    EXPECT_GE(pi, 0.0);
    EXPECT_LT(pi, 7.0);
}

TEST(ReplCommandsTest, lzw_encode_vec) {
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

TEST(ReplCommandsTest, lzw_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "lzw_decode_vec(C)");

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "E = lzw_encode_vec(M)");
    expect_ok(interp, "R = lzw_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 6u);
}

TEST(ReplCommandsTest, lz77_encode_decode_vec) {
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

TEST(ReplCommandsTest, lz77_encode_vec_custom) {
    Interpreter interp;
    expect_contains(interp, "help", "lz77_encode_vec(M,window,lookahead)");

    expect_ok(interp, "lzM2 = [120; 121; 122; 120; 121; 122; 120; 121]");
    expect_ok(interp, "T2 = lz77_encode_vec(lzM2, 64, 8)");
    ASSERT_GT(interp.state().matrices.count("T2"), 0u);
    expect_ok(interp, "R2 = lz77_decode_vec(T2)");
    ASSERT_GT(interp.state().matrices.count("R2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R2").rows(), 8u);
}

TEST(ReplCommandsTest, medfilt2) {
    Interpreter interp;
    expect_contains(interp, "help", "medfilt2(M,k)");

    expect_ok(interp,
              "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, count_components) {
    Interpreter interp;
    expect_contains(interp, "help", "count_components(B)");

    expect_ok(interp, "B = [1, 1, 0, 0, 0, 0; 1, 1, 0, 0, 0, 0; 0, 0, 0, 0, 0, 0; 0, 0, 0, 0, 0, 1; 0, 0, 0, 0, 1, 1; 0, 0, 0, 0, 0, 1]");
    expect_ok(interp, "n = count_components(B)");
    EXPECT_NEAR(interp.state().scalars.at("n"), 2.0, 1e-9);
}

TEST(ReplCommandsTest, image_morph_hist) {
    Interpreter interp;
    expect_contains(interp, "help", "imtophat(M[,k])");
    expect_contains(interp, "help", "imbothat(M[,k])");
    expect_contains(interp, "help", "imadjust(M,in_lo,in_hi[,out_lo,out_hi])");
    expect_contains(interp, "help", "imhist(M[,nbins])");

    // Bright center pixel on dark background â€” top-hat keeps the peak.
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

    // Four distinct levels into 4 bins â†’ counts [2,1,1,0] (same as ImageHist.CustomBinCount).
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

TEST(ReplCommandsTest, imgradient_morph) {
    Interpreter interp;
    expect_contains(interp, "help", "imgradient_morph(M[,k])");

    // Bright center on dark background â€” gradient responds at edges.
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

TEST(ReplCommandsTest, lsq) {
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

TEST(ReplCommandsTest, special_hypergeo_whittaker) {
    Interpreter interp;
    expect_contains(interp, "help", "hypergeo_0f1(b,z)");
    expect_contains(interp, "help", "hypergeo_1f1(a,z)");
    expect_contains(interp, "help", "hypergeo_2f1(a,b,c,z)");
    expect_contains(interp, "help", "kummer_m(a,b,z)");
    expect_contains(interp, "help", "whittaker_m(kappa,mu,z)");
    expect_contains(interp, "help", "whittaker_w(kappa,mu,z)");

    const double h0_ref = ms::hypergeo_0f1(2.0, 1.0);
    EXPECT_NEAR(h0_ref, 1.590636854637329, 1e-3);
    expect_ok(interp, "h0 = hypergeo_0f1(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), h0_ref, 1e-3);
    expect_contains(interp, "hypergeo_0f1(2, 1)", std::to_string(h0_ref));

    const double h1_ref = ms::hypergeo_1f1(1.0, 0.0);
    EXPECT_NEAR(h1_ref, 1.0, 1e-12);
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), h1_ref, 1e-9);

    const double h2_ref = ms::hypergeo_2f1(1.0, 1.0, 2.0, 0.5);
    EXPECT_NEAR(h2_ref, -std::log(0.5) / 0.5, 1e-3);
    expect_ok(interp, "h2 = hypergeo_2f1(1, 1, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h2"), h2_ref, 1e-3);
    expect_contains(interp, "hypergeo_2f1(1, 1, 2, 0.5)", std::to_string(h2_ref));

    const double m_ref = ms::kummer_m(1.0, 2.0, 0.5);
    EXPECT_NEAR(m_ref, 1.2974425414002564, 1e-3);
    expect_ok(interp, "m = kummer_m(1, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("m"), m_ref, 1e-3);
    expect_contains(interp, "kummer_m(1, 2, 0.5)", std::to_string(m_ref));

    const double wm_ref = ms::whittaker_m(0.0, 0.5, 1.0);
    EXPECT_NEAR(wm_ref, 1.0421906109874948, 1e-3);
    expect_ok(interp, "wm = whittaker_m(0, 0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("wm"), wm_ref, 1e-3);
    expect_contains(interp, "whittaker_m(0, 0.5, 1)", std::to_string(wm_ref));

    const double ww_ref = ms::whittaker_w(0.0, 0.5, 1.0);
    EXPECT_NEAR(ww_ref, 0.6065306597126334, 1e-3);
    expect_ok(interp, "ww = whittaker_w(0, 0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ww"), ww_ref, 1e-3);
    expect_contains(interp, "whittaker_w(0, 0.5, 1)", std::to_string(ww_ref));
}

TEST(ReplCommandsTest, special_mathieu_spheroidal_pcf) {
    Interpreter interp;
    expect_contains(interp, "help", "mathieu_a(n,q)");
    expect_contains(interp, "help", "spheroidal_lambda(n,m,c)");
    expect_contains(interp, "help", "spheroidal_s1(n,m,c,x)");
    expect_contains(interp, "help", "spheroidal_s2(n,m,c,x)");
    expect_contains(interp, "help", "pcf_u(a,x)");
    expect_contains(interp, "help", "pcf_v(a,x)");
    expect_contains(interp, "help", "pcf_w(a,x)");

    const double q = 0.1;
    const double a_ref = ms::mathieu_a(1, q);
    EXPECT_NEAR(a_ref, 1.0987343129634084, 1e-3);
    expect_ok(interp, "a = mathieu_a(1, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("a"), a_ref, 1e-3);

    const double lam_ref = ms::spheroidal_lambda(1, 1, 5.0);
    EXPECT_NEAR(lam_ref, -7.493388284110646, 3e-2);
    expect_ok(interp, "lam = spheroidal_lambda(1, 1, 5.0)");
    EXPECT_NEAR(interp.state().scalars.at("lam"), lam_ref, 3e-2);

    const double s1_ref = ms::spheroidal_s1(1, 1, 5.0, 0.5);
    EXPECT_NEAR(s1_ref, 0.03747174337125646, 5e-2);
    expect_ok(interp, "s1 = spheroidal_s1(1, 1, 5.0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), s1_ref, 5e-2);

    const double s2_ref = ms::spheroidal_s2(1, 1, 5.0, 0.5);
    EXPECT_TRUE(std::isfinite(s2_ref));
    expect_ok(interp, "s2 = spheroidal_s2(1, 1, 5.0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), s2_ref, 5e-2);

    const double u_ref = ms::pcf_u(0.5, 1.0);
    const double v_ref = ms::pcf_v(0.5, 1.0);
    const double w_ref = ms::pcf_w(0.5, 1.0);
    EXPECT_TRUE(std::isfinite(u_ref));
    EXPECT_TRUE(std::isfinite(v_ref));
    EXPECT_TRUE(std::isfinite(w_ref));
    expect_ok(interp, "u = pcf_u(0.5, 1.0)");
    expect_ok(interp, "v = pcf_v(0.5, 1.0)");
    expect_ok(interp, "w = pcf_w(0.5, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("u"), u_ref, 1e-3);
    EXPECT_NEAR(interp.state().scalars.at("v"), v_ref, 1e-3);
    EXPECT_NEAR(interp.state().scalars.at("w"), w_ref, 1e-3);
}

TEST(ReplCommandsTest, special_mathieu_heun) {
    Interpreter interp;
    expect_contains(interp, "help", "mathieu_se(n,q,x)");
    expect_contains(interp, "help", "mathieu_b(n,q)");
    expect_contains(interp, "help", "mathieu_mc(n,q,x)");
    expect_contains(interp, "help", "mathieu_ms(n,q,x)");
    expect_contains(interp, "help", "heun_c(q,alpha,beta,gamma,delta,z)");
    expect_contains(interp, "help", "heun_d(q,alpha,gamma,delta,z)");
    expect_contains(interp, "help", "heun_b(q,alpha,beta,delta,z)");
    expect_contains(interp, "help", "heun_t(q,alpha,beta,gamma,z)");
    expect_contains(interp, "help", "painleve2(x,y0,yp0,alpha)");

    EXPECT_NEAR(ms::mathieu_b(1, 0.0), 1.0, 1e-8);
    expect_ok(interp, "b = mathieu_b(1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("b"), 1.0, 1e-6);

    const double se_ref = ms::mathieu_se(1, 0.1, 0.5);
    EXPECT_TRUE(std::isfinite(se_ref));
    expect_ok(interp, "se = mathieu_se(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("se"), se_ref, 1e-3);

    const double mc_ref = ms::mathieu_mc(1, 0.2, 0.3);
    const double ms_ref = ms::mathieu_ms(1, 0.2, 0.3);
    expect_ok(interp, "mc = mathieu_mc(1, 0.2, 0.3)");
    expect_ok(interp, "msv = mathieu_ms(1, 0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("mc"), mc_ref, 1e-3);
    EXPECT_NEAR(interp.state().scalars.at("msv"), ms_ref, 1e-3);

    const double hc_ref = ms::heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 0.2);
    EXPECT_NEAR(hc_ref, 0.9472246160936063, 5e-3);
    expect_ok(interp, "hc = heun_c(0.1, 0.2, 0.3, 0.4, 0.5, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("hc"), hc_ref, 5e-3);

    const double hb_ref = ms::heun_b(0.1, 0.2, 0.3, 0.4, 0.2);
    EXPECT_NEAR(hb_ref, 1.0738133528291396, 5e-3);
    expect_ok(interp, "hb = heun_b(0.1, 0.2, 0.3, 0.4, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("hb"), hb_ref, 5e-3);

    const double hd_ref = ms::heun_d(0.1, 0.2, 0.3, 0.4, 0.2);
    expect_ok(interp, "hd = heun_d(0.1, 0.2, 0.3, 0.4, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("hd"), hd_ref, 5e-3);

    const double ht_ref = ms::heun_t(0.1, 0.2, 0.3, 0.4, 0.25);
    expect_ok(interp, "ht = heun_t(0.1, 0.2, 0.3, 0.4, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("ht"), ht_ref, 5e-3);

    const double p2_ref = ms::painleve2(0.5, 0.0, -0.5, 0.0);
    EXPECT_NEAR(p2_ref, -0.2530083397481052, 1e-3);
    expect_ok(interp, "p2 = painleve2(0.5, 0.0, -0.5, 0.0)");
    EXPECT_NEAR(interp.state().scalars.at("p2"), p2_ref, 1e-3);
}

TEST(ReplCommandsTest, special_painleve_dawsonx) {
    Interpreter interp;
    expect_contains(interp, "help", "dawsonx(x)");
    expect_contains(interp, "help", "painleve3(x,y0,yp0,alpha,beta)");
    expect_contains(interp, "help", "painleve4(x,y0,yp0,alpha,beta)");
    expect_contains(interp, "help", "painleve5(x,y0,yp0,alpha,beta,gamma,delta)");
    expect_contains(interp, "help", "painleve6(x,y0,yp0,alpha,beta,gamma,delta)");

    const double dx_ref = ms::dawsonx(0.5);
    EXPECT_TRUE(std::isfinite(dx_ref));
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), dx_ref, 5e-3);

    const double p3_ref = ms::painleve3(0.5, 0.5, -0.1, 0.5, 0.3);
    EXPECT_NEAR(p3_ref, 1.398748842793728, 5e-3);
    expect_ok(interp, "p3 = painleve3(0.5, 0.5, -0.1, 0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("p3"), p3_ref, 5e-3);

    const double p4_ref = ms::painleve4(0.5, 0.8, -0.05, 0.2, 0.4);
    EXPECT_NEAR(p4_ref, 0.786121344510419, 5e-3);
    expect_ok(interp, "p4 = painleve4(0.5, 0.8, -0.05, 0.2, 0.4)");
    EXPECT_NEAR(interp.state().scalars.at("p4"), p4_ref, 5e-3);

    const double p5_ref = ms::painleve5(0.5, 0.5, -0.05, 0.01, 0.02, 0.03, 0.04);
    EXPECT_NEAR(p5_ref, 0.5558194327648597, 5e-3);
    expect_ok(interp, "p5 = painleve5(0.5, 0.5, -0.05, 0.01, 0.02, 0.03, 0.04)");
    EXPECT_NEAR(interp.state().scalars.at("p5"), p5_ref, 5e-3);

    const double p6_ref = ms::painleve6(2.5, 0.5, -0.05, 0.1, 0.2, 0.3, 0.4);
    EXPECT_NEAR(p6_ref, 0.5003268969869713, 5e-3);
    expect_ok(interp, "p6 = painleve6(2.5, 0.5, -0.05, 0.1, 0.2, 0.3, 0.4)");
    EXPECT_NEAR(interp.state().scalars.at("p6"), p6_ref, 5e-3);
}

TEST(ReplCommandsTest, special_hypergeo_meijer) {
    Interpreter interp;
    expect_contains(interp, "help", "tricomi_u(a,b,z)");
    expect_contains(interp, "help", "meijer_g(a,b,z)");
    expect_contains(interp, "help", "fox_h(a,b,z)");
    expect_contains(interp, "help", "hypergeo_0f1n(n,a,z)");
    expect_contains(interp, "help", "hypergeo_1f1n(n,a,z)");

    const double tu_ref = ms::tricomi_u(1.0, 2.0, 0.5);
    EXPECT_NEAR(tu_ref, 2.0, 1e-3);
    expect_ok(interp, "tu = tricomi_u(1, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("tu"), tu_ref, 1e-3);
    expect_contains(interp, "tricomi_u(1, 2, 0.5)", std::to_string(tu_ref));

    const double mg_ref = ms::meijer_g(1.0, 2.0, 0.5);
    EXPECT_NEAR(mg_ref, 0.39346934028736663, 1e-3);
    expect_ok(interp, "mg = meijer_g(1, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mg"), mg_ref, 1e-3);
    expect_contains(interp, "meijer_g(1, 2, 0.5)", std::to_string(mg_ref));

    const double fh_ref = ms::fox_h(1.0, 2.0, 0.5);
    EXPECT_NEAR(fh_ref, mg_ref, 1e-12);
    expect_ok(interp, "fh = fox_h(1, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("fh"), fh_ref, 1e-3);
    expect_contains(interp, "fox_h(1, 2, 0.5)", std::to_string(fh_ref));

    const double h0n_ref = ms::hypergeo_0f1n(2, 1.5, 0.2);
    EXPECT_TRUE(std::isfinite(h0n_ref));
    expect_ok(interp, "h0n = hypergeo_0f1n(2, 1.5, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("h0n"), h0n_ref, 1e-3);
    expect_contains(interp, "hypergeo_0f1n(2, 1.5, 0.2)", std::to_string(h0n_ref));

    const double h1n_ref = ms::hypergeo_1f1n(1, 1.0, 0.3);
    EXPECT_TRUE(std::isfinite(h1n_ref));
    expect_ok(interp, "h1n = hypergeo_1f1n(1, 1, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("h1n"), h1n_ref, 1e-3);
    expect_contains(interp, "hypergeo_1f1n(1, 1, 0.3)", std::to_string(h1n_ref));
}

TEST(ReplCommandsTest, optim_cg_rmsprop) {
    Interpreter interp;

    expect_contains(interp, "help", "conjugate_gradient(\"formula\",x0");
    expect_contains(interp, "help", "rmsprop(\"formula\",x0");
    expect_contains(interp, "help", "adadelta(\"formula\",x0");

    const auto cg = interp.execute(R"cmd(conjugate_gradient("(x0-3)*(x0-3)", [0]))cmd");
    ASSERT_TRUE(cg.has_value());
    {
        const std::size_t start = cg->find('[');
        ASSERT_NE(start, std::string::npos);
        const std::size_t end = cg->find(']', start);
        ASSERT_NE(end, std::string::npos);
        EXPECT_NEAR(std::stod(cg->substr(start + 1, end - start - 1)), 3.0, 1e-3);
    }

    const auto rp = interp.execute(R"cmd(rmsprop("(x0-3)*(x0-3)", [0], 0.05, 500))cmd");
    ASSERT_TRUE(rp.has_value());
    {
        const std::size_t start = rp->find('[');
        ASSERT_NE(start, std::string::npos);
        const std::size_t end = rp->find(']', start);
        ASSERT_NE(end, std::string::npos);
        EXPECT_NEAR(std::stod(rp->substr(start + 1, end - start - 1)), 3.0, 1e-2);
    }

    expect_ok(interp, R"cmd(adadelta("(x0-3)*(x0-3)", [0], 1.0, 8000))cmd");
    expect_contains(interp, R"cmd(adadelta("(x0-3)*(x0-3)", [0], 1.0, 8000))cmd", "f_val =");
}

TEST(ReplCommandsTest, optim_roots_global) {
    Interpreter interp;
    expect_contains(interp, "help", "bisection(\"formula\",a,b[,tol[,max_iter]])");
    expect_contains(interp, "help", "brentq(\"formula\",a,b[,tol[,max_iter]])");
    expect_contains(interp, "help", "illinois(\"formula\",a,b[,tol[,max_iter]])");
    expect_contains(interp, "help", "secant(\"formula\",x0,x1[,tol[,max_iter]])");
    expect_contains(interp, "help", "halley(\"f\",\"df\",\"d2f\",x0[,tol[,max_iter]])");
    expect_contains(interp, "help", "fixed_point(\"formula\",x0[,tol[,max_iter]])");
    expect_contains(interp, "help",
                    "simulated_annealing(\"formula\",x0[,T0[,cooling[,max_iter[,seed]]]])");
    expect_contains(interp, "help",
                    "differential_evolution(\"formula\",bounds[,pop[,F[,CR[,max_iter[,seed]]]]])");
    expect_contains(interp, "help",
                    "particle_swarm(\"formula\",bounds[,n_particles[,max_iter[,seed]]])");

    const auto bis = interp.execute("bisection(\"x0 - 3\", 0, 10)");
    ASSERT_TRUE(bis.has_value());
    EXPECT_NEAR(parse_scalar_x_opt(*bis), 3.0, 1e-6);
    expect_error_contains(interp, "bisection(\"sin(\", 0, 10)", "bisection");

    const auto brent = interp.execute("brentq(\"x0*x0*x0 - 2*x0 - 5\", 1, 3)");
    ASSERT_TRUE(brent.has_value());
    EXPECT_NEAR(parse_scalar_x_opt(*brent), 2.094551, 1e-4);
    expect_error_contains(interp, "brentq(\"bad(@)\", 1, 3)", "brentq");

    const auto ill = interp.execute("illinois(\"x0*x0 - 2\", 0, 2)");
    ASSERT_TRUE(ill.has_value());
    EXPECT_NEAR(parse_scalar_x_opt(*ill), std::sqrt(2.0), 1e-6);
    expect_error_contains(interp, "illinois(\"x0 +\", 0, 2)", "illinois");

    const auto sec = interp.execute("secant(\"x0*x0 - 4\", 1, 3)");
    ASSERT_TRUE(sec.has_value());
    EXPECT_NEAR(std::abs(parse_scalar_x_opt(*sec)), 2.0, 1e-4);
    expect_error_contains(interp, "secant(\"x0*\", 1, 3)", "secant");

    const auto hal =
        interp.execute("halley(\"x0*x0*x0 - x0 - 1\", \"3*x0*x0 - 1\", \"6*x0\", 1.5)");
    ASSERT_TRUE(hal.has_value());
    EXPECT_NEAR(parse_scalar_x_opt(*hal), 1.324717957, 1e-4);
    expect_error_contains(interp, "halley(\"x0\", \"1\", \"0\", bad)", "halley");

    const auto fp = interp.execute("fixed_point(\"(x0 + 2/x0)/2\", 1.5)");
    ASSERT_TRUE(fp.has_value());
    EXPECT_NEAR(parse_scalar_x_opt(*fp), std::sqrt(2.0), 1e-6);
    expect_error_contains(interp, "fixed_point(\"x0/\", 1.5)", "fixed_point");

    const auto sa = interp.execute(
        "simulated_annealing(\"x0*x0 + x1*x1\", [2, 2], 1, 0.99, 5000, 42)");
    ASSERT_TRUE(sa.has_value());
    EXPECT_LT(parse_optim_f_val(*sa), 2.0);
    expect_error_contains(interp, "simulated_annealing(\"x0*\", [1, 1])", "simulated_annealing");

    const auto de = interp.execute(
        "differential_evolution(\"x0*x0 + x1*x1\", [[-5, 5], [-5, 5]], 20, 0.8, 0.9, 500, 42)");
    ASSERT_TRUE(de.has_value());
    EXPECT_LT(parse_optim_f_val(*de), 0.5);
    expect_error_contains(interp, "differential_evolution(\"x0*\", [[0, 1]])",
                          "differential_evolution");

    const auto pso = interp.execute(
        "particle_swarm(\"x0*x0 + x1*x1\", [[-5, 5], [-5, 5]], 20, 200, 42)");
    ASSERT_TRUE(pso.has_value());
    EXPECT_LT(parse_optim_f_val(*pso), 1.0);
    expect_error_contains(interp, "particle_swarm(\"x0*\", [[0, 1], [2, 3]])", "particle_swarm");
}

TEST(ReplCommandsTest, special_weierstrass_zeta_sigma) {
    Interpreter interp;
    expect_contains(interp, "help", "weierstrass_zeta(z,g2,g3)");
    expect_contains(interp, "help", "weierstrass_sigma(z,g2,g3)");

    const double zeta_ref = ms::weierstrass_zeta(0.5, 1.0, 0.0);
    const double sigma_ref = ms::weierstrass_sigma(0.5, 1.0, 0.0);
    expect_ok(interp, "wz = weierstrass_zeta(0.5, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wz"), zeta_ref, 1e-9);
    expect_ok(interp, "ws = weierstrass_sigma(0.5, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("ws"), sigma_ref, 1e-9);
    expect_contains(interp, "weierstrass_zeta(0.5, 1, 0)", "\n");
}

TEST(ReplCommandsTest, special_ellip_d) {
    Interpreter interp;
    expect_contains(interp, "help", "ellip_d(k)");
    const double k = 0.5;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(k), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_6) {
    Interpreter interp;

    expect_ok(interp, "gc = gegenbauer_c(3, 0.5, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("gc"), ms::gegenbauer_c(3, 0.5, 0.25), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_7) {
    Interpreter interp;

    expect_ok(interp, "la = laguerre_la(2, 1.0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("la"), ms::laguerre_la(2, 1.0, 0.5), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_8) {
    Interpreter interp;

    expect_ok(interp, "ln = laguerre_ln(2, 0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ln"), ms::laguerre_ln(2, 0, 0.5), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_10) {
    Interpreter interp;

    expect_ok(interp, "h0 = hypergeo_0f1n(2, 1.5, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1n(2, 1.5, 0.2), 1e-9);

    expect_ok(interp, "h1 = hypergeo_1f1n(1, 0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1n(1, 0.5, 0.3), 1e-9);
}

TEST(ReplCommandsTest, special_scalar_14) {
    Interpreter interp;

    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-9);

    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-9);
}

TEST(ReplCommandsTest, kelvin_struve_scalar) {
    Interpreter interp;

    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1.0), 1e-9);

    expect_ok(interp, "yn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("yn"), ms::struve_yn(0, 1.0), 1e-9);

    expect_ok(interp, "bei = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bei"), ms::kelvin_bei(0, 1.0), 1e-9);

    expect_ok(interp, "ber = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ber"), ms::kelvin_ber(0, 1.0), 1e-9);

    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1.0), 1e-9);

    expect_ok(interp, "kei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kei"), ms::kelvin_kei(0, 1.0), 1e-9);
}

TEST(ReplCommandsTest, mathieu_scalar) {
    Interpreter interp;

    const double a_ref = ms::mathieu_a(1, 0.1);
    expect_ok(interp, "a = mathieu_a(1, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("a"), a_ref, 1e-3);

    expect_ok(interp, "b = mathieu_b(1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("b"), ms::mathieu_b(1, 0.0), 1e-8);
}

TEST(ReplCommandsTest, mathieu_three_arg_scalar) {
    Interpreter interp;

    const double ce_ref = ms::mathieu_ce(1, 0.1, 0.5);
    expect_ok(interp, "ce = mathieu_ce(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ce"), ce_ref, 1e-8);

    const double se_ref = ms::mathieu_se(1, 0.1, 0.5);
    expect_ok(interp, "se = mathieu_se(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("se"), se_ref, 1e-8);

    const double mc_ref = ms::mathieu_mc(0, 0.1, 0.0);
    expect_ok(interp, "mc = mathieu_mc(0, 0.1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("mc"), mc_ref, 1e-6);

    const double mms_ref = ms::mathieu_ms(1, 0.1, 0.5);
    expect_ok(interp, "mms = mathieu_ms(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mms"), mms_ref, 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_spheroidal_scalar) {
    Interpreter interp;

    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);

    expect_ok(interp, "sl = spheroidal_lambda(1, 0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::spheroidal_lambda(1, 0, 0.1), 1e-6);
}

TEST(ReplCommandsTest, spheroidal_s1_scalar) {
    Interpreter interp;

    expect_ok(interp, "s1 = spheroidal_s1(1, 0, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), ms::spheroidal_s1(1, 0, 0.1, 0.5), 1e-6);
}

TEST(ReplCommandsTest, spheroidal_s2_scalar) {
    Interpreter interp;

    expect_ok(interp, "s2 = spheroidal_s2(1, 0, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), ms::spheroidal_s2(1, 0, 0.1, 0.5), 1e-5);
}

TEST(ReplCommandsTest, mathieu_a_scalar) {
    Interpreter interp;

    expect_ok(interp, "a = mathieu_a(1, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("a"), ms::mathieu_a(1, 0.1), 1e-3);
}

TEST(ReplCommandsTest, hermite_he_scalar) {
    Interpreter interp;

    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar) {
    Interpreter interp;

    expect_ok(interp, "sj = sph_bessel_j(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar) {
    Interpreter interp;
    expect_ok(interp, "sy = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar) {
    Interpreter interp;
    expect_ok(interp, "p = polylog(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("p"), ms::polylog(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar) {
    Interpreter interp;
    expect_ok(interp, "d = debye(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("d"), ms::debye(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar) {
    Interpreter interp;
    expect_ok(interp, "ll = laguerre_l(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ll"), ms::laguerre_l(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, laguerre_ln_scalar) {
    Interpreter interp;
    expect_ok(interp, "ln = laguerre_ln(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ln"), ms::laguerre_ln(2, 1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, laguerre_la_scalar) {
    Interpreter interp;
    expect_ok(interp, "la = laguerre_la(2, 0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("la"), ms::laguerre_la(2, 0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar) {
    Interpreter interp;
    expect_ok(interp, "yn = struve_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("yn"), ms::struve_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar) {
    Interpreter interp;
    expect_ok(interp, "bei = kelvin_bei(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("bei"), ms::kelvin_bei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar) {
    Interpreter interp;
    expect_ok(interp, "ber = kelvin_ber(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("ber"), ms::kelvin_ber(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar) {
    Interpreter interp;
    expect_ok(interp, "kei = kelvin_kei(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("kei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar) {
    Interpreter interp;
    expect_ok(interp, "h = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar) {
    Interpreter interp;
    expect_ok(interp, "h = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.3), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, tophat_imadjust) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);
}

TEST(ReplCommandsTest, imhist_gray2rgb) {
    Interpreter interp;

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.3), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, tophat_imadjust_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);
}

TEST(ReplCommandsTest, imhist_gray2rgb_2) {
    Interpreter interp;

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_3) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_4) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_3) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_3) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_3) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_3) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_5) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_4) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_4) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_4) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_4) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_6) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_5) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_5) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_5) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_5) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_7) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_6) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_6) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_6) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_6) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_8) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_7) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_7) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_7) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_7) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_9) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_8) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_8) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_8) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_8) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_10) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_9) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_9) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_9) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_9) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_11) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_10) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_10) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_10) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_10) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_12) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_11) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_11) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_11) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_11) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_13) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_12) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_12) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_12) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_12) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_14) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_13) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_13) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_13) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_13) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_15) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_14) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_14) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_14) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_14) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_16) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_15) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_15) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_15) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_15) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_17) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_16) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_16) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_16) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_16) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_18) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_17) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_17) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_17) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_17) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_19) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_18) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_18) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_18) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_18) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_20) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_19) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_19) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_19) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_19) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_21) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_20) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_20) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_20) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_20) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_22) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_21) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_21) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_21) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_21) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_23) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_22) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_22) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_22) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_22) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_24) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_23) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_23) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_23) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_23) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_100) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_101) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_25) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_24) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_24) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_24) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_24) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_102) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_103) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_104) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_105) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_26) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_25) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_25) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_25) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_25) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_106) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_107) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_108) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_109) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_27) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_26) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_26) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_26) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_26) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_110) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_111) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_112) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_113) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_28) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_27) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_27) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_27) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_27) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_114) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_115) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_116) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_117) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_29) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_28) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_28) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_28) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_28) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_118) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_119) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_120) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_121) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_30) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_29) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_29) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_29) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_29) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_122) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_123) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_124) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_125) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_31) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_30) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_30) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_30) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_30) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_126) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_in_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_jn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_kn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, spherical_yn_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, polylog_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}

TEST(ReplCommandsTest, debye_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("db"), ms::debye(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "t2 = theta2(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t2"), ms::theta2(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "t3 = theta3(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t3"), ms::theta3(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_inc_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "kbi = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbi"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_127) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "kker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kker"), ms::kelvin_ker(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "kkei = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kkei"), ms::kelvin_kei(0, 1.0), 1e-8);
}

TEST(ReplCommandsTest, hermite_he_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_j_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-8);
}

TEST(ReplCommandsTest, sph_bessel_y_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, zeta_hurwitz_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hf_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_f_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_128) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_97) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_129) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, imflip_kruskal_32) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(ReplCommandsTest, ellip_pi_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}

TEST(ReplCommandsTest, imgaussfilt_medfilt2_31) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(ReplCommandsTest, ellip_k_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_d_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ed"), ms::ellip_d(0.5), 1e-8);
}

TEST(ReplCommandsTest, dawson_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}

TEST(ReplCommandsTest, ellip_e_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}

TEST(ReplCommandsTest, imtophat_imbothat_imgradient_morph_31) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    expect_ok(interp, "B = imbothat(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    expect_ok(interp, "G = imgradient_morph(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 3u);
}

TEST(ReplCommandsTest, airy_aip_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "aap = airy_aip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("aap"), ms::airy_aip(0.5), 1e-8);
}

TEST(ReplCommandsTest, imadjust_imhist_31) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "A = imadjust(G, 0, 1)");
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 0.2, 1e-5);

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
}

TEST(ReplCommandsTest, mathieu_b_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "mb = mathieu_b(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), ms::mathieu_b(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, gray2rgb_impad_31) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.8; 0.4, 0.6]");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
}

TEST(ReplCommandsTest, dawsonx_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "dx = dawsonx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dx"), ms::dawsonx(0.5), 1e-8);
}

TEST(ReplCommandsTest, erfcx_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}

TEST(ReplCommandsTest, airy_bip_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "bip = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bip"), ms::airy_bip(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_1f1_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, theta2_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta3_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, theta4_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}

TEST(ReplCommandsTest, hypergeo_0f1_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "h0 = hypergeo_0f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h0"), ms::hypergeo_0f1(1, 0.25), 1e-8);
}

TEST(ReplCommandsTest, clausen_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "cl = clausen(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), ms::clausen(0.5), 1e-8);
}

TEST(ReplCommandsTest, theta1_prime_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}

TEST(ReplCommandsTest, ellip_pi_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}

TEST(ReplCommandsTest, lambert_w_scalar_99) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ker_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_ker(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_kei_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "ki = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ki"), ms::kelvin_kei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, hermite_hn_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "hn = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::hermite_hn(2, 0.5), 1e-8);
}

TEST(ReplCommandsTest, anger_j_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}

TEST(ReplCommandsTest, weber_e_scalar_130) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_bei_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_k_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_hn_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "shn = struve_hn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("shn"), ms::struve_hn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, struve_yn_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(0, 1), 1e-8);
}

TEST(ReplCommandsTest, kelvin_ber_scalar_98) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}

TEST(ReplCommandsTest, laguerre_l_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}

TEST(ReplCommandsTest, unary_display_factorizations) {
    Interpreter interp;
    expect_ok(interp, "A = [2, 0; 0, 3]");
    expect_contains(interp, "bidiag(A)", "B =");
    expect_contains(interp, "qr(A)", "Q =");
    expect_contains(interp, "lu(A)", "L =");
    expect_contains(interp, "svd(A)", "singular values");
    expect_contains(interp, "eig(A)", "eigenvalues");
    expect_contains(interp, "eig_sym(A)", "eigenvalues");
    expect_contains(interp, "hess(A)", "H =");
    expect_contains(interp, "schur(A)", "T =");
    expect_contains(interp, "chol(A)", "L =");
    expect_ok(interp, "U2, B2 = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("U2"), 0u);
    ASSERT_GT(interp.state().matrices.count("B2"), 0u);
    expect_error(interp, "bidiag(no_such_matrix)");
    expect_error(interp, "chol([-1, 0; 0, -1])");
}

TEST(ReplCommandsTest, unary_display_transforms) {
    Interpreter interp;
    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_contains(interp, "fftshift(M)", "shifted");
    expect_contains(interp, "ifftshift(M)", "shifted");
    expect_contains(interp, "ml_mat_transpose(M)", "At");
    expect_contains(interp, "poly_deriv([1; 2; 3])", "deriv");
    expect_contains(interp, "diag([1, 2, 3])", "D =");
    expect_contains(interp, "prewitt([1, 2, 3; 4, 5, 6; 7, 8, 9])", "edge");
    expect_contains(interp, "scharr([1, 2, 3; 4, 5, 6; 7, 8, 9])", "edge");
    expect_contains(interp, "roberts([1, 2, 3; 4, 5, 6; 7, 8, 9])", "edge");
    expect_error(interp, "fftshift(missing)");
}

TEST(ReplCommandsTest, info_permutation_entropy_errors) {
    Interpreter interp;
    expect_ok(interp, "xpe = [1; 2; 3; 4; 5; 6]");
    expect_contains(interp, "info_permutation_entropy(xpe)", "\n");
    expect_contains(interp, "info_permutation_entropy(xpe, 3, 1)", "\n");
    expect_error_contains(interp, "info_permutation_entropy()", "info_permutation_entropy");
    expect_error_contains(interp, "info_permutation_entropy(xpe, 0)", "positive integer order");
    expect_error_contains(interp, "info_permutation_entropy(xpe, 3, 0)", "positive integer delay");
    expect_error(interp, "info_permutation_entropy(xpe, not_a_number)");
}

TEST(ReplCommandsTest, signal_firwin_lms_error_paths) {
    Interpreter interp;
    expect_ok(interp, "signal_firwin(5, 0.2)");
    expect_ok(interp, "signal_firwin(5, 0.2, 0)");
    expect_error_contains(interp, "signal_firwin(1.5, 0.2)", "integer n_taps");
    expect_error(interp, "signal_firwin(foo, 0.2)");
    expect_ok(interp, "signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    expect_error(interp, "signal_lms(missing_x, [1; 2], 2, 0)");
    expect_error(interp, "signal_coherence(missing, missing, 8, 8)");
}

TEST(ReplCommandsTest, clausen_1_execute_noassign) {
    Interpreter interp;
    expect_ok(interp, "clausen(1)");
}

TEST(ReplCommandsTest, eta_dirichlet_2_execute_noassign) {
    Interpreter interp;
    expect_ok(interp, "eta_dirichlet(2)");
}

TEST(ReplCommandsTest, heun_b_noassign) {
    Interpreter interp;
    expect_ok(interp, "heun_b(0.1, 0.2, 0.3, 0.4, 0.2)");
    expect_error_contains(interp, "heun_b(0.1, 0.2, 0.3, 0.4, missing)", "numeric");
}

TEST(ReplCommandsTest, heun_d_noassign) {
    Interpreter interp;
    expect_ok(interp, "heun_d(0.1, 0.2, 0.3, 0.4, 0.2)");
    expect_error_contains(interp, "heun_d(0.1, 0.2, 0.3, 0.4, missing)", "numeric");
}

TEST(ReplCommandsTest, heun_t_noassign) {
    Interpreter interp;
    expect_ok(interp, "heun_t(0.1, 0.2, 0.3, 0.4, 0.2)");
    expect_error_contains(interp, "heun_t(0.1, 0.2, 0.3, 0.4, missing)", "numeric");
}

TEST(ReplCommandsTest, beta_dirichlet_noassign) {
    Interpreter interp;
    expect_ok(interp, "beta_dirichlet(2)");
    expect_error_contains(interp, "beta_dirichlet(missing)", "expected numeric s");
}

TEST(ReplCommandsTest, cuda_add_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = [5, 6; 7, 8]");
    expect_contains(interp, "cuda_add(A, B)", "sum =");
    expect_error_contains(interp, "cuda_add(no_such_matrix, B)", "unknown matrix");
}

TEST(ReplCommandsTest, ml_mat_mul_noassign) {
    Interpreter interp;
    expect_ok(interp, "I = eye(3)");
    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "ml_mat_mul(I, A)", "C =");
    expect_error_contains(interp, "ml_mat_mul(no_such_matrix, A)", "unknown matrix");
}

TEST(ReplCommandsTest, graph_k_core_subgraph_noassign) {
    Interpreter interp;
    expect_ok(interp, "C4 = [0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0]");
    expect_contains(interp, "graph_k_core_subgraph(C4, 2)", "subgraph =");
    expect_error_contains(interp, "graph_k_core_subgraph(C4, 1.5)", "integer k");
}

TEST(ReplCommandsTest, fem_poisson1d_noassign) {
    Interpreter interp;
    expect_contains(interp, "fem_poisson1d(4, 0)", "u =");
    expect_error_contains(interp, "fem_poisson1d(missing, 0)", "expected fem_poisson1d");
}

TEST(ReplCommandsTest, fem_poisson2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "fem_poisson2d(2, 2)", "u =");
    expect_error_contains(interp, "fem_poisson2d(2, missing)", "expected fem_poisson2d");
}

TEST(ReplCommandsTest, fem_poisson3d_noassign) {
    Interpreter interp;
    expect_contains(interp, "fem_poisson3d(2, 2, 2)", "u =");
    expect_error_contains(interp, "fem_poisson3d(2, 2, missing)", "expected fem_poisson3d");
}

TEST(ReplCommandsTest, sym_laplace_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_laplace(\"1\", \"t\", \"s\")");
    expect_error_contains(interp, "sym_laplace(\"1\", t, \"s\")", "expected sym_laplace");
}

TEST(ReplCommandsTest, sym_ilaplace_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_ilaplace(\"1/s\", \"s\", \"t\")");
    expect_error_contains(interp, "sym_ilaplace(\"1/s\", s, \"t\")", "expected sym_ilaplace");
}

TEST(ReplCommandsTest, sym_mellin_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_mellin(\"1\", \"t\", \"s\")");
    expect_error_contains(interp, "sym_mellin(\"1\", t, \"s\")", "expected sym_mellin");
}

TEST(ReplCommandsTest, sym_imellin_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_imellin(\"1\", \"s\", \"t\")");
    expect_error_contains(interp, "sym_imellin(\"1\", s, \"t\")", "expected sym_imellin");
}

TEST(ReplCommandsTest, sym_hankel_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_hankel(\"1\", \"r\", \"k\")");
    expect_error_contains(interp, "sym_hankel(\"1\", r, \"k\")", "expected sym_hankel");
}

TEST(ReplCommandsTest, sym_ihankel_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_ihankel(\"1\", \"k\", \"r\")");
    expect_error_contains(interp, "sym_ihankel(\"1\", k, \"r\")", "expected sym_ihankel");
}

TEST(ReplCommandsTest, sym_fourier_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_fourier(\"1\", \"t\", \"w\")");
    expect_error_contains(interp, "sym_fourier(\"1\", t, \"w\")", "expected sym_fourier");
}

TEST(ReplCommandsTest, sym_ifourier_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_ifourier(\"1\", \"w\", \"t\")");
    expect_error_contains(interp, "sym_ifourier(\"1\", w, \"t\")", "expected sym_ifourier");
}

TEST(ReplCommandsTest, sym_ztransform_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_ztransform(\"1\", \"n\", \"z\")");
    expect_error_contains(interp, "sym_ztransform(\"1\", n, \"z\")", "expected sym_ztransform");
}

TEST(ReplCommandsTest, sym_iztransform_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_iztransform(\"1\", \"z\", \"n\")");
    expect_error_contains(interp, "sym_iztransform(\"1\", z, \"n\")", "expected sym_iztransform");
}

TEST(ReplCommandsTest, sym_dsolve_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_dsolve(\"1\", \"x\", \"y\")");
    expect_error_contains(interp, "sym_dsolve(\"1\", x, \"y\")", "expected sym_dsolve");
}

TEST(ReplCommandsTest, sym_substitute_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_substitute(\"x+1\", \"x\", \"2\")");
    expect_error_contains(interp, "sym_substitute(\"x+1\", x, \"2\")", "expected sym_substitute");
}

TEST(ReplCommandsTest, sym_limit_noassign) {
    Interpreter interp;
    expect_contains(interp, "sym_limit(\"x+1\", \"x\", 2)", "3");
    expect_error_contains(interp, "sym_limit(\"x+1\", x, 2)", "expected sym_limit");
}

TEST(ReplCommandsTest, sym_series_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_series(\"exp(x)\", \"x\", 0, 2)");
    expect_error_contains(interp, "sym_series(\"exp(x)\", x, 0, 2)", "expected sym_series");
}

TEST(ReplCommandsTest, sym_simplify_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_simplify(\"x+x\")");
    expect_error_contains(interp, "sym_simplify(x+x)", "expected");
}

TEST(ReplCommandsTest, sym_expand_noassign) {
    Interpreter interp;
    expect_ok(interp, "sym_expand(\"x+x\")");
    expect_error_contains(interp, "sym_expand(x+x)", "expected");
}

TEST(ReplCommandsTest, hypergeo_0f1_noassign) {
    Interpreter interp;
    expect_ok(interp, "hypergeo_0f1(2, 1)");
    expect_error_contains(interp, "hypergeo_0f1(2, missing)", "expected hypergeo_0f1(b,z)");
}

TEST(ReplCommandsTest, weierstrass_zeta_noassign) {
    Interpreter interp;
    expect_ok(interp, "weierstrass_zeta(0.5, 1, 0)");
    expect_error_contains(interp, "weierstrass_zeta(0.5, 1, missing)",
                          "expected weierstrass_zeta(z,g2,g3)");
}

TEST(ReplCommandsTest, weierstrass_p_noassign) {
    Interpreter interp;
    expect_ok(interp, "weierstrass_p(0.5, 1, 0)");
    expect_error_contains(interp, "weierstrass_p(0.5, 1, missing)",
                          "expected weierstrass_p(z,g2,g3)");
}

TEST(ReplCommandsTest, weierstrass_pprime_noassign) {
    Interpreter interp;
    expect_ok(interp, "weierstrass_pprime(0.5, 1, 0)");
    expect_error_contains(interp, "weierstrass_pprime(0.5, 1, missing)",
                          "expected weierstrass_pprime(z,g2,g3)");
}

TEST(ReplCommandsTest, beta_noassign) {
    Interpreter interp;
    expect_ok(interp, "beta(2, 3)");
    expect_error_contains(interp, "beta(2, missing)", "expected numeric arguments beta(a,b)");
}

TEST(ReplCommandsTest, zeta_hurwitz_noassign) {
    Interpreter interp;
    expect_ok(interp, "zeta_hurwitz(2, 0.3)");
    expect_error_contains(interp, "zeta_hurwitz(2, missing)", "expected zeta_hurwitz(s,a)");
}

TEST(ReplCommandsTest, chebyshev_t_noassign) {
    Interpreter interp;
    expect_ok(interp, "chebyshev_t(2, 0.5)");
    expect_error_contains(interp, "chebyshev_t(2, missing)", "expected chebyshev_t(n,x)");
}

TEST(ReplCommandsTest, jacobi_sc_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_sc(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_sc(0.5, missing)", "expected jacobi_sc(u,k)");
}

TEST(ReplCommandsTest, jacobi_cn_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_cn(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_cn(0.5, missing)", "expected jacobi_cn(u,k)");
}

TEST(ReplCommandsTest, ellip_pi_noassign) {
    Interpreter interp;
    expect_ok(interp, "ellip_pi(0.5, 0.5)");
    expect_error_contains(interp, "ellip_pi(0.5, missing)", "expected ellip_pi(n,k)");
}

TEST(ReplCommandsTest, fft_goertzel_noassign) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 0; -1; 0]");
    expect_contains(interp, "fft_goertzel(x, 0.25, 1.0)", "goertzel =");
    expect_error_contains(interp, "fft_goertzel(missing, 0.25, 1.0)", "unknown matrix");
    expect_error_contains(interp, "fft_goertzel(x, notnum, 1.0)",
                          "expected fft_goertzel(x, f, fs)");
}

TEST(ReplCommandsTest, bessel_j_noassign) {
    Interpreter interp;
    expect_ok(interp, "bessel_j(0, 1)");
    expect_error_contains(interp, "bessel_j(0, missing)", "expected bessel_j(nu,x)");
}

TEST(ReplCommandsTest, struve_h_noassign) {
    Interpreter interp;
    expect_ok(interp, "struve_h(1, 0.5)");
    expect_error_contains(interp, "struve_h(1, missing)", "expected struve_h(nu,x)");
}

TEST(ReplCommandsTest, theta1_noassign) {
    Interpreter interp;
    expect_ok(interp, "theta1(0.5, 0.3)");
    expect_error_contains(interp, "theta1(0.5, missing)", "expected theta1(z,q)");
}

TEST(ReplCommandsTest, jacobi_sn_noassign) {
    Interpreter interp;
    expect_ok(interp, "jacobi_sn(0.5, 0.5)");
    expect_error_contains(interp, "jacobi_sn(0.5, missing)", "expected jacobi_sn(u,k)");
}

TEST(ReplCommandsTest, bfgs_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(bfgs("(x0-3)*(x0-3)", [0]))cmd");
    expect_error_contains(interp, "bfgs()", "expected bfgs(\"formula\", x0[, tol[, max_iter]])");
}

TEST(ReplCommandsTest, nelder_mead_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(nelder_mead("(x0-3)*(x0-3)", [8]))cmd");
    expect_error_contains(interp, "nelder_mead()",
                          "expected nelder_mead(\"formula\", x0[, tol[, max_iter]])");
}

TEST(ReplCommandsTest, lbfgs_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(lbfgs("(x0-3)*(x0-3)", [0]))cmd");
    expect_error_contains(interp, "lbfgs()",
                          "expected lbfgs(\"formula\", x0[, m[, tol[, max_iter]]])");
}

TEST(ReplCommandsTest, cmaes_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(cmaes("x0*x0+x1*x1", [2, 3], 0.5, 50, 42))cmd");
    expect_error_contains(interp, "cmaes()",
                          "expected cmaes(\"formula\", x0, sigma0, max_iter[, seed])");
}

TEST(ReplCommandsTest, golden_section_noassign) {
    Interpreter interp;
    expect_contains(interp, R"cmd(golden_section("(x0-3)*(x0-3)", 0, 10))cmd", "x_opt");
    expect_error_contains(interp, "golden_section()",
                          "expected golden_section(\"formula\", a, b[, tol])");
}

TEST(ReplCommandsTest, levenberg_marquardt_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(levenberg_marquardt("x0-3", [0]))cmd");
    expect_error_contains(interp, "levenberg_marquardt()",
                          "expected levenberg_marquardt(\"r0;r1;...\", x0[, max_iter[, tol]])");
}

TEST(ReplCommandsTest, adam_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(adam("(x0-3)*(x0-3)", [0]))cmd");
    expect_error_contains(interp, "adam()", "expected adam(\"formula\", x0[, alpha[, max_iter]])");
}

TEST(ReplCommandsTest, adadelta_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(adadelta("(x0-3)*(x0-3)", [0]))cmd");
    expect_error_contains(interp, "adadelta()",
                          "expected adadelta(\"formula\", x0[, lr[, max_iter]])");
}

TEST(ReplCommandsTest, rmsprop_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(rmsprop("(x0-3)*(x0-3)", [0]))cmd");
    expect_error_contains(interp, "rmsprop()",
                          "expected rmsprop(\"formula\", x0[, alpha[, max_iter]])");
}

TEST(ReplCommandsTest, conjugate_gradient_noassign) {
    Interpreter interp;
    expect_ok(interp, R"cmd(conjugate_gradient("(x0-3)*(x0-3)", [0]))cmd");
    expect_error_contains(interp, "conjugate_gradient()",
                          "expected conjugate_gradient(\"formula\", x0[, tol[, max_iter]])");
}

TEST(ReplCommandsTest, sin_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "s = sin(0)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, cos_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "c = cos(0)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, tan_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "t = tan(0)");
    EXPECT_NEAR(interp.state().scalars.at("t"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, asin_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "as = asin(0)");
    EXPECT_NEAR(interp.state().scalars.at("as"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, acos_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "ac = acos(1)");
    EXPECT_NEAR(interp.state().scalars.at("ac"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, atan_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "at = atan(0)");
    EXPECT_NEAR(interp.state().scalars.at("at"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, sinh_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "sh = sinh(0)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, cosh_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "ch = cosh(0)");
    EXPECT_NEAR(interp.state().scalars.at("ch"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, tanh_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "th = tanh(0)");
    EXPECT_NEAR(interp.state().scalars.at("th"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, sqrt_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "sq = sqrt(4)");
    EXPECT_NEAR(interp.state().scalars.at("sq"), 2.0, 1e-12);
}

TEST(ReplCommandsTest, abs_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "ab = abs(-2)");
    EXPECT_NEAR(interp.state().scalars.at("ab"), 2.0, 1e-12);
}

TEST(ReplCommandsTest, exp_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "e = exp(0)");
    EXPECT_NEAR(interp.state().scalars.at("e"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, log_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "lg = log(1)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, log10_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "l10 = log10(10)");
    EXPECT_NEAR(interp.state().scalars.at("l10"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, floor_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "fl = floor(1.5)");
    EXPECT_NEAR(interp.state().scalars.at("fl"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, ceil_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cl = ceil(1.5)");
    EXPECT_NEAR(interp.state().scalars.at("cl"), 2.0, 1e-12);
}

TEST(ReplCommandsTest, pow_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "pw = pow(2,3)");
    EXPECT_NEAR(interp.state().scalars.at("pw"), 8.0, 1e-12);
}

TEST(ReplCommandsTest, min_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "mn = min(1,2)");
    EXPECT_NEAR(interp.state().scalars.at("mn"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, max_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "mx = max(1,2)");
    EXPECT_NEAR(interp.state().scalars.at("mx"), 2.0, 1e-12);
}

TEST(ReplCommandsTest, atan2_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "a2 = atan2(0,1)");
    EXPECT_NEAR(interp.state().scalars.at("a2"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, bigint_is_even_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "ev = bigint_is_even(4)");
    EXPECT_NEAR(interp.state().scalars.at("ev"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, bigint_is_odd_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "od = bigint_is_odd(3)");
    EXPECT_NEAR(interp.state().scalars.at("od"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, bigint_isqrt_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "iq = bigint_isqrt(9)");
    EXPECT_NEAR(interp.state().scalars.at("iq"), 3.0, 1e-12);
}

TEST(ReplCommandsTest, bigint_is_prime_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "pr = bigint_is_prime(7)");
    EXPECT_NEAR(interp.state().scalars.at("pr"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, bigint_lcm_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "lc = bigint_lcm(4,6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-12);
}

TEST(ReplCommandsTest, bigint_pow_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "bp = bigint_pow(2,5)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 32.0, 1e-12);
}

TEST(ReplCommandsTest, mpi_allreduce_sum_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "ms = mpi_allreduce_sum(1)");
    EXPECT_NEAR(interp.state().scalars.at("ms"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, mpi_allreduce_max_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "mm = mpi_allreduce_max(2)");
    EXPECT_NEAR(interp.state().scalars.at("mm"), 2.0, 1e-12);
}

TEST(ReplCommandsTest, mpi_allreduce_min_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "mi = mpi_allreduce_min(3)");
    EXPECT_NEAR(interp.state().scalars.at("mi"), 3.0, 1e-12);
}

TEST(ReplCommandsTest, mpi_bcast_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "mb = mpi_bcast(1)");
    EXPECT_NEAR(interp.state().scalars.at("mb"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_allreduce_sum_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cs = cuda_allreduce_sum(1)");
    EXPECT_NEAR(interp.state().scalars.at("cs"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_allreduce_max_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cm = cuda_allreduce_max(2)");
    EXPECT_NEAR(interp.state().scalars.at("cm"), 2.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_allreduce_min_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cn = cuda_allreduce_min(0)");
    EXPECT_NEAR(interp.state().scalars.at("cn"), 0.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_allreduce_prod_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cp = cuda_allreduce_prod(1)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_allreduce_avg_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "ca = cuda_allreduce_avg(1)");
    EXPECT_NEAR(interp.state().scalars.at("ca"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_broadcast_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cb = cuda_broadcast(1)");
    EXPECT_NEAR(interp.state().scalars.at("cb"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_reduce_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cr = cuda_reduce(1)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), 1.0, 1e-12);
}

TEST(ReplCommandsTest, cuda_allgather_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "cg = cuda_allgather(1)");
    EXPECT_NEAR(interp.state().scalars.at("cg"), 1.0, 1e-12);
}

