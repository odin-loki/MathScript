// MathScript Integration Tests: REPL tensorops decomposition session-object pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <optional>
#include <sstream>
#include <string>

#include "ms/core/matrix.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;
using ms::Matrix;

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

std::string trim_output(const std::string& text) {
    return Interpreter::trim(text);
}

std::optional<Matrix<double>> parse_matrix_output(const std::string& text) {
    auto parse_row_values = [](const std::string& row_text) -> std::optional<std::vector<double>> {
        const auto open = row_text.find('[');
        const auto close = row_text.find(']');
        if (open == std::string::npos || close == std::string::npos || close <= open) {
            return std::nullopt;
        }
        std::stringstream values(row_text.substr(open + 1, close - open - 1));
        std::string cell;
        std::vector<double> row;
        while (std::getline(values, cell, ',')) {
            cell = Interpreter::trim(cell);
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }
        if (row.empty()) {
            return std::nullopt;
        }
        return row;
    };

    std::vector<std::vector<double>> rows;
    std::stringstream line_stream(text);
    std::string line;
    while (std::getline(line_stream, line)) {
        if (line.find('[') == std::string::npos) {
            continue;
        }
        const auto open = line.find('[');
        const auto close = line.find(']');
        if (close > open &&
            line.substr(open + 1, close - open - 1).find(';') != std::string::npos) {
            continue;
        }
        auto row = parse_row_values(line);
        if (row) {
            rows.push_back(std::move(*row));
        }
    }

    if (rows.empty()) {
        const auto open = text.find('[');
        const auto close = text.rfind(']');
        if (open == std::string::npos || close == std::string::npos || close <= open) {
            return std::nullopt;
        }
        std::stringstream row_stream(text.substr(open + 1, close - open - 1));
        std::string row_text;
        while (std::getline(row_stream, row_text, ';')) {
            row_text = Interpreter::trim(row_text);
            if (row_text.empty()) {
                continue;
            }
            std::stringstream values(row_text);
            std::string cell;
            std::vector<double> row;
            while (std::getline(values, cell, ',')) {
                cell = Interpreter::trim(cell);
                if (!cell.empty()) {
                    row.push_back(std::stod(cell));
                }
            }
            if (row.empty()) {
                return std::nullopt;
            }
            rows.push_back(std::move(row));
        }
    }

    if (rows.empty()) {
        return std::nullopt;
    }
    const size_t cols = rows.front().size();
    Matrix<double> m(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != cols) {
            return std::nullopt;
        }
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = rows[i][j];
        }
    }
    return m;
}

double frobenius_error(const Matrix<double>& a, const Matrix<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            const double diff = a(i, j) - b(i, j);
            sum += diff * diff;
        }
    }
    return std::sqrt(sum);
}

std::optional<double> parse_residual_from_create(const std::string& text) {
    const auto pos = text.find("residual=");
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    return std::stod(text.substr(pos + 9));
}

} // namespace

TEST(TensoropsDecomposeReplPipeline, CpDecomposeCreatesHandle) {
    Interpreter interp;
    const auto created =
        run(interp, "tensorops_decompose_cp(cp1, [1, 2; 3, 4], 2)");
    ASSERT_TRUE(created.has_value());
    EXPECT_NE(created->find("CPDecomposition"), std::string::npos);
    EXPECT_NE(created->find("residual="), std::string::npos);
    const auto residual = parse_residual_from_create(*created);
    ASSERT_TRUE(residual.has_value());
    EXPECT_LT(*residual, 1e-3);
}

TEST(TensoropsDecomposeReplPipeline, CpRoundTripLowError) {
    Interpreter interp;
    const auto created = run(interp, "tensorops_decompose_cp(cp1, [1, 2; 3, 4], 2)");
    ASSERT_TRUE(created.has_value());
    const auto residual = parse_residual_from_create(*created);
    ASSERT_TRUE(residual.has_value());
    EXPECT_LT(*residual, 1e-2);

    const auto reconstructed = run(interp, "tensorops_reconstruct_cp(cp1)");
    ASSERT_TRUE(reconstructed.has_value()) << *reconstructed;
    const auto approx_opt = parse_matrix_output(*reconstructed);
    const auto truth_opt = parse_matrix_output("[1, 2; 3, 4]");
    ASSERT_TRUE(approx_opt.has_value()) << *reconstructed;
    ASSERT_TRUE(truth_opt.has_value());
    ASSERT_EQ(approx_opt->rows(), truth_opt->rows());
    ASSERT_EQ(approx_opt->cols(), truth_opt->cols());
    const double error = frobenius_error(*truth_opt, *approx_opt);
    EXPECT_NEAR(error, *residual, 1e-4);
    EXPECT_LT(error, 1e-2);
}

TEST(TensoropsDecomposeReplPipeline, TuckerDecomposeCreatesHandle) {
    Interpreter interp;
    const auto created =
        run(interp, "tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [2, 2])");
    ASSERT_TRUE(created.has_value());
    EXPECT_NE(created->find("TuckerDecomposition"), std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, TuckerRoundTripLowError) {
    Interpreter interp;
    expect_ok(interp, "tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [2, 2])");
    const auto reconstructed = run(interp, "tensorops_reconstruct_tucker(tk1)");
    ASSERT_TRUE(reconstructed.has_value());
    const auto approx_opt = parse_matrix_output(*reconstructed);
    const auto truth_opt = parse_matrix_output("[1, 2; 3, 4]");
    ASSERT_TRUE(approx_opt.has_value());
    ASSERT_TRUE(truth_opt.has_value());
    ASSERT_EQ(approx_opt->rows(), truth_opt->rows());
    ASSERT_EQ(approx_opt->cols(), truth_opt->cols());
    const double error = frobenius_error(*truth_opt, *approx_opt);
    EXPECT_LT(error, 1e-2);
}

TEST(TensoropsDecomposeReplPipeline, HosvdDecomposeCreatesHandle) {
    Interpreter interp;
    const auto created =
        run(interp, "tensorops_decompose_hosvd(tk2, [1, 0; 0, 1], [2, 2])");
    ASSERT_TRUE(created.has_value());
    EXPECT_NE(created->find("TuckerDecomposition"), std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, HosvdRoundTripLowError) {
    Interpreter interp;
    expect_ok(interp, "tensorops_decompose_hosvd(tk2, [1, 0; 0, 1], [2, 2])");
    const auto reconstructed = run(interp, "tensorops_reconstruct_tucker(tk2)");
    ASSERT_TRUE(reconstructed.has_value());
    const auto approx_opt = parse_matrix_output(*reconstructed);
    const auto truth_opt = parse_matrix_output("[1, 0; 0, 1]");
    ASSERT_TRUE(approx_opt.has_value());
    ASSERT_TRUE(truth_opt.has_value());
    ASSERT_EQ(approx_opt->rows(), truth_opt->rows());
    ASSERT_EQ(approx_opt->cols(), truth_opt->cols());
    const double error = frobenius_error(*truth_opt, *approx_opt);
    EXPECT_LT(error, 1e-6);
}

TEST(TensoropsDecomposeReplPipeline, ReconstructMissingHandleError) {
    Interpreter interp;
    expect_error(interp, "tensorops_reconstruct_cp(missing)");
    const auto missing = run(interp, "tensorops_reconstruct_cp(missing)");
    ASSERT_FALSE(missing.has_value());
    EXPECT_NE(ms::format_error(missing.error()).find("not found"), std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, ReconstructWrongTypeError) {
    Interpreter interp;
    expect_ok(interp, "tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [2, 2])");
    expect_error(interp, "tensorops_reconstruct_cp(tk1)");
    const auto wrong_type = run(interp, "tensorops_reconstruct_cp(tk1)");
    ASSERT_FALSE(wrong_type.has_value());
    EXPECT_NE(ms::format_error(wrong_type.error()).find("not a CPDecomposition"),
              std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, DecomposeInvalidRankError) {
    Interpreter interp;
    expect_error(interp, "tensorops_decompose_cp(cp_bad, [1, 2; 3, 4], 0)");
    const auto invalid = run(interp, "tensorops_decompose_cp(cp_bad, [1, 2; 3, 4], 0)");
    ASSERT_FALSE(invalid.has_value());
    EXPECT_NE(ms::format_error(invalid.error()).find("positive integer rank"),
              std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, SessionObjectsListsHandles) {
    Interpreter interp;
    expect_ok(interp, "tensorops_decompose_cp(cp1, [1, 2; 3, 4], 2)");
    expect_ok(interp, "tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [2, 2])");
    expect_ok(interp, "tensorops_decompose_hosvd(tk2, [1, 0; 0, 1], [2, 2])");
    const auto listed = run(interp, "session_objects()");
    ASSERT_TRUE(listed.has_value());
    EXPECT_NE(listed->find("cp1 cp"), std::string::npos);
    EXPECT_NE(listed->find("tk1 tucker"), std::string::npos);
    EXPECT_NE(listed->find("tk2 tucker"), std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, SessionObjectClearRemovesHandles) {
    Interpreter interp;
    expect_ok(interp, "tensorops_decompose_cp(cp1, [1, 2; 3, 4], 2)");
    expect_ok(interp, "tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [2, 2])");
    expect_ok(interp, "session_object_clear(cp1)");
    const auto listed = run(interp, "session_objects()");
    ASSERT_TRUE(listed.has_value());
    EXPECT_EQ(listed->find("cp1 cp"), std::string::npos);
    EXPECT_NE(listed->find("tk1 tucker"), std::string::npos);

    expect_error(interp, "tensorops_reconstruct_cp(cp1)");
    const auto missing = run(interp, "tensorops_reconstruct_cp(cp1)");
    ASSERT_FALSE(missing.has_value());
    EXPECT_NE(ms::format_error(missing.error()).find("not found"), std::string::npos);
}

TEST(TensoropsDecomposeReplPipeline, CpDecomposeOptionalIterTol) {
    Interpreter interp;
    const auto created = run(
        interp, "tensorops_decompose_cp(cp_opt, [2, 1; 1, 2], 2, 300, 1e-8)");
    ASSERT_TRUE(created.has_value());
    EXPECT_NE(created->find("CPDecomposition"), std::string::npos);
    const auto residual = parse_residual_from_create(*created);
    ASSERT_TRUE(residual.has_value());
    EXPECT_LT(*residual, 1e-3);
}
