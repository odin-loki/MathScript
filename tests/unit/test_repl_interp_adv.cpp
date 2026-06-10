// MathScript Interpreter Advanced Tests
// Covers: assign_scalar, assign_scalar_binary, assign_scalar_matrix_call,
//         eval_scalar_call, PlotSeries/has_plot/plot, print_matrix,
//         SessionState, try_parse_scalar_assignment,
//         try_parse_scalar_binary_assignment, try_parse_scalar_expr_assignment,
//         try_parse_matrix_call_assignment

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;
using ms::Matrix;

// ---------------------------------------------------------------------------
// assign_scalar
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_AssignScalar, BasicValue_StoresInState) {
    Interpreter interp;
    auto r = interp.assign_scalar("x", 3.14);
    ASSERT_TRUE(r.has_value());
    ASSERT_TRUE(interp.state().scalars.count("x") > 0);
    EXPECT_NEAR(interp.state().scalars.at("x"), 3.14, 1e-12);
}

TEST(ReplInterpAdv_AssignScalar, ResultStringContainsName) {
    Interpreter interp;
    auto r = interp.assign_scalar("myvar", 7.0);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->find("myvar"), std::string::npos);
}

TEST(ReplInterpAdv_AssignScalar, ZeroValue) {
    Interpreter interp;
    auto r = interp.assign_scalar("zero", 0.0);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(interp.state().scalars.at("zero"), 0.0);
}

TEST(ReplInterpAdv_AssignScalar, NegativeValue) {
    Interpreter interp;
    auto r = interp.assign_scalar("neg", -42.5);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("neg"), -42.5, 1e-12);
}

TEST(ReplInterpAdv_AssignScalar, OverwriteExistingScalar) {
    Interpreter interp;
    interp.assign_scalar("a", 1.0);
    interp.assign_scalar("a", 99.0);
    EXPECT_NEAR(interp.state().scalars.at("a"), 99.0, 1e-12);
}

TEST(ReplInterpAdv_AssignScalar, MultipleDistinctScalars) {
    Interpreter interp;
    interp.assign_scalar("p", 10.0);
    interp.assign_scalar("q", 20.0);
    interp.assign_scalar("r", 30.0);
    EXPECT_EQ(interp.state().scalars.size(), 3u);
}

// ---------------------------------------------------------------------------
// assign_scalar_binary
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_AssignScalarBinary, TwoLiterals_Add) {
    Interpreter interp;
    ScalarBinaryAssign sa;
    sa.target = "result";
    sa.op = '+';
    sa.left.is_literal = true;  sa.left.literal = 3.0;
    sa.right.is_literal = true; sa.right.literal = 4.0;
    auto r = interp.assign_scalar_binary(sa);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("result"), 7.0, 1e-12);
}

TEST(ReplInterpAdv_AssignScalarBinary, TwoLiterals_Subtract) {
    Interpreter interp;
    ScalarBinaryAssign sa;
    sa.target = "diff";
    sa.op = '-';
    sa.left.is_literal = true;  sa.left.literal = 10.0;
    sa.right.is_literal = true; sa.right.literal = 3.0;
    auto r = interp.assign_scalar_binary(sa);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("diff"), 7.0, 1e-12);
}

TEST(ReplInterpAdv_AssignScalarBinary, TwoLiterals_Multiply) {
    Interpreter interp;
    ScalarBinaryAssign sa;
    sa.target = "prod";
    sa.op = '*';
    sa.left.is_literal = true;  sa.left.literal = 5.0;
    sa.right.is_literal = true; sa.right.literal = 6.0;
    auto r = interp.assign_scalar_binary(sa);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("prod"), 30.0, 1e-12);
}

TEST(ReplInterpAdv_AssignScalarBinary, TwoLiterals_Divide) {
    Interpreter interp;
    ScalarBinaryAssign sa;
    sa.target = "quot";
    sa.op = '/';
    sa.left.is_literal = true;  sa.left.literal = 9.0;
    sa.right.is_literal = true; sa.right.literal = 3.0;
    auto r = interp.assign_scalar_binary(sa);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("quot"), 3.0, 1e-12);
}

TEST(ReplInterpAdv_AssignScalarBinary, VarOperand_UsesStateScalar) {
    Interpreter interp;
    interp.assign_scalar("x", 10.0);
    ScalarBinaryAssign sa;
    sa.target = "y";
    sa.op = '-';
    sa.left.is_literal = false; sa.left.name = "x";
    sa.right.is_literal = true; sa.right.literal = 3.0;
    auto r = interp.assign_scalar_binary(sa);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("y"), 7.0, 1e-12);
}

TEST(ReplInterpAdv_AssignScalarBinary, UnknownVar_ReturnsError) {
    Interpreter interp;
    ScalarBinaryAssign sa;
    sa.target = "y";
    sa.op = '+';
    sa.left.is_literal = false; sa.left.name = "nonexistent_var";
    sa.right.is_literal = true; sa.right.literal = 1.0;
    auto r = interp.assign_scalar_binary(sa);
    EXPECT_FALSE(r.has_value());
}

TEST(ReplInterpAdv_AssignScalarBinary, DivisionByZeroLiteral_ReturnsError) {
    Interpreter interp;
    ScalarBinaryAssign sa;
    sa.target = "bad";
    sa.op = '/';
    sa.left.is_literal = true;  sa.left.literal = 5.0;
    sa.right.is_literal = true; sa.right.literal = 0.0;
    auto r = interp.assign_scalar_binary(sa);
    EXPECT_FALSE(r.has_value());
}

TEST(ReplInterpAdv_AssignScalarBinary, BothVarOperands) {
    Interpreter interp;
    interp.assign_scalar("a", 8.0);
    interp.assign_scalar("b", 2.0);
    ScalarBinaryAssign sa;
    sa.target = "c";
    sa.op = '*';
    sa.left.is_literal = false; sa.left.name = "a";
    sa.right.is_literal = false; sa.right.name = "b";
    auto r = interp.assign_scalar_binary(sa);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("c"), 16.0, 1e-12);
}

// ---------------------------------------------------------------------------
// assign_scalar_matrix_call
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_AssignScalarMatrixCall, Trace_IdentityMatrix) {
    Interpreter interp;
    ScalarMatrixCallAssign sma;
    sma.target = "t";
    sma.callee = "trace";
    sma.arg = "[1, 0; 0, 1]";
    auto r = interp.assign_scalar_matrix_call(sma);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("t"), 2.0, 1e-10);
}

TEST(ReplInterpAdv_AssignScalarMatrixCall, Trace_DiagonalMatrix) {
    Interpreter interp;
    ScalarMatrixCallAssign sma;
    sma.target = "tr";
    sma.callee = "trace";
    sma.arg = "[1, 2; 3, 4]";
    auto r = interp.assign_scalar_matrix_call(sma);
    ASSERT_TRUE(r.has_value());
    // trace = 1 + 4 = 5
    EXPECT_NEAR(interp.state().scalars.at("tr"), 5.0, 1e-10);
}

TEST(ReplInterpAdv_AssignScalarMatrixCall, Det_DiagonalMatrix) {
    Interpreter interp;
    ScalarMatrixCallAssign sma;
    sma.target = "d";
    sma.callee = "det";
    sma.arg = "[2, 0; 0, 3]";
    auto r = interp.assign_scalar_matrix_call(sma);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("d"), 6.0, 1e-10);
}

TEST(ReplInterpAdv_AssignScalarMatrixCall, Norm_StoresInState) {
    Interpreter interp;
    ScalarMatrixCallAssign sma;
    sma.target = "nrm";
    sma.callee = "norm";
    sma.arg = "[1, 0; 0, 1]";
    auto r = interp.assign_scalar_matrix_call(sma);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(interp.state().scalars.count("nrm") > 0);
    EXPECT_GT(interp.state().scalars.at("nrm"), 0.0);
}

TEST(ReplInterpAdv_AssignScalarMatrixCall, UnknownMatrixArg_ReturnsError) {
    Interpreter interp;
    ScalarMatrixCallAssign sma;
    sma.target = "t";
    sma.callee = "trace";
    sma.arg = "NoSuchMatrix";
    auto r = interp.assign_scalar_matrix_call(sma);
    EXPECT_FALSE(r.has_value());
}

TEST(ReplInterpAdv_AssignScalarMatrixCall, MatrixFromState_ViaExecute) {
    Interpreter interp;
    interp.execute("A = [1, 2; 3, 4]");
    ScalarMatrixCallAssign sma;
    sma.target = "tr";
    sma.callee = "trace";
    sma.arg = "A";
    auto r = interp.assign_scalar_matrix_call(sma);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("tr"), 5.0, 1e-10); // 1 + 4
}

TEST(ReplInterpAdv_AssignScalarMatrixCall, Rank_IdentityMatrix) {
    Interpreter interp;
    ScalarMatrixCallAssign sma;
    sma.target = "rk";
    sma.callee = "rank";
    sma.arg = "[1, 0; 0, 1]";
    auto r = interp.assign_scalar_matrix_call(sma);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("rk"), 2.0, 1e-10);
}

// ---------------------------------------------------------------------------
// eval_scalar_call (static)
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_EvalScalarCall, Sin_Zero) {
    auto r = Interpreter::eval_scalar_call("sin", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Cos_Zero) {
    auto r = Interpreter::eval_scalar_call("cos", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Tan_Zero) {
    auto r = Interpreter::eval_scalar_call("tan", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Sqrt_Four) {
    auto r = Interpreter::eval_scalar_call("sqrt", {4.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 2.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Sqrt_Nine) {
    auto r = Interpreter::eval_scalar_call("sqrt", {9.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 3.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Abs_Negative) {
    auto r = Interpreter::eval_scalar_call("abs", {-7.5});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 7.5, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Abs_Positive) {
    auto r = Interpreter::eval_scalar_call("abs", {3.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 3.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Exp_Zero) {
    auto r = Interpreter::eval_scalar_call("exp", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Log_One) {
    auto r = Interpreter::eval_scalar_call("log", {1.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Log10_Hundred) {
    auto r = Interpreter::eval_scalar_call("log10", {100.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 2.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Floor_Down) {
    auto r = Interpreter::eval_scalar_call("floor", {3.9});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 3.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Ceil_Up) {
    auto r = Interpreter::eval_scalar_call("ceil", {3.1});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 4.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Asin_Zero) {
    auto r = Interpreter::eval_scalar_call("asin", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Acos_One) {
    auto r = Interpreter::eval_scalar_call("acos", {1.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Atan_Zero) {
    auto r = Interpreter::eval_scalar_call("atan", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Pow_TwoArgs) {
    auto r = Interpreter::eval_scalar_call("pow", {2.0, 10.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1024.0, 1e-8);
}

TEST(ReplInterpAdv_EvalScalarCall, Min_PicksSmaller) {
    auto r = Interpreter::eval_scalar_call("min", {5.0, 3.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 3.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Max_PicksLarger) {
    auto r = Interpreter::eval_scalar_call("max", {5.0, 3.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 5.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Atan2_TwoArgs) {
    auto r = Interpreter::eval_scalar_call("atan2", {1.0, 1.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, std::atan2(1.0, 1.0), 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, UnknownFunction_ReturnsError) {
    auto r = Interpreter::eval_scalar_call("nonexistent_fn", {1.0});
    EXPECT_FALSE(r.has_value());
}

TEST(ReplInterpAdv_EvalScalarCall, CaseInsensitive_SIN) {
    auto r = Interpreter::eval_scalar_call("SIN", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, CaseInsensitive_SQRT) {
    auto r = Interpreter::eval_scalar_call("SQRT", {16.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 4.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Sinh_Zero) {
    auto r = Interpreter::eval_scalar_call("sinh", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Cosh_Zero) {
    auto r = Interpreter::eval_scalar_call("cosh", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1.0, 1e-12);
}

TEST(ReplInterpAdv_EvalScalarCall, Tanh_Zero) {
    auto r = Interpreter::eval_scalar_call("tanh", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// PlotSeries struct members and has_plot / plot methods
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_PlotSeries, DefaultState_NoPlot) {
    Interpreter interp;
    EXPECT_FALSE(interp.has_plot());
    EXPECT_FALSE(interp.plot().valid);
}

TEST(ReplInterpAdv_PlotSeries, DefaultPlotSeries_ZeroFields) {
    PlotSeries ps;
    EXPECT_FALSE(ps.valid);
    EXPECT_EQ(ps.kind, PlotSeries::Kind::Line);
    EXPECT_EQ(ps.matrix_rows, 0u);
    EXPECT_EQ(ps.matrix_cols, 0u);
    EXPECT_EQ(ps.nnz, 0u);
    EXPECT_TRUE(ps.x.empty());
    EXPECT_TRUE(ps.y.empty());
}

TEST(ReplInterpAdv_PlotSeries, KindEnumValuesDistinct) {
    EXPECT_NE(PlotSeries::Kind::Line, PlotSeries::Kind::Bar);
    EXPECT_NE(PlotSeries::Kind::Bar, PlotSeries::Kind::Scatter);
    EXPECT_NE(PlotSeries::Kind::Scatter, PlotSeries::Kind::Heatmap);
    EXPECT_NE(PlotSeries::Kind::Heatmap, PlotSeries::Kind::Spy);
    EXPECT_NE(PlotSeries::Kind::Spy, PlotSeries::Kind::Surface3D);
}

TEST(ReplInterpAdv_PlotSeries, AfterPlotCommand_HasPlotTrue) {
    Interpreter interp;
    auto r = interp.execute("plot([1, 2, 3])");
    if (r.has_value()) {
        EXPECT_TRUE(interp.has_plot());
        EXPECT_TRUE(interp.plot().valid);
        EXPECT_EQ(interp.plot().kind, PlotSeries::Kind::Line);
        EXPECT_EQ(interp.plot().y.size(), 3u);
    }
    SUCCEED();
}

TEST(ReplInterpAdv_PlotSeries, Reset_ClearsPlot) {
    Interpreter interp;
    interp.execute("plot([1, 2, 3])");
    interp.reset();
    EXPECT_FALSE(interp.has_plot());
    EXPECT_FALSE(interp.plot().valid);
}

TEST(ReplInterpAdv_PlotSeries, PlotConstRef_ReturnsSameData) {
    Interpreter interp;
    auto r = interp.execute("plot([10, 20, 30])");
    if (r.has_value() && interp.has_plot()) {
        const PlotSeries& ps = interp.plot();
        EXPECT_EQ(ps.y.size(), 3u);
        EXPECT_NEAR(ps.y[0], 10.0, 1e-10);
        EXPECT_NEAR(ps.y[1], 20.0, 1e-10);
        EXPECT_NEAR(ps.y[2], 30.0, 1e-10);
    }
    SUCCEED();
}

TEST(ReplInterpAdv_PlotSeries, PlotSeries_CanSetFields) {
    PlotSeries ps;
    ps.valid = true;
    ps.kind = PlotSeries::Kind::Bar;
    ps.x = {1.0, 2.0, 3.0};
    ps.y = {4.0, 5.0, 6.0};
    ps.matrix_rows = 10;
    ps.matrix_cols = 20;
    ps.nnz = 5;
    EXPECT_TRUE(ps.valid);
    EXPECT_EQ(ps.kind, PlotSeries::Kind::Bar);
    EXPECT_EQ(ps.x.size(), 3u);
    EXPECT_EQ(ps.y.size(), 3u);
    EXPECT_EQ(ps.matrix_rows, 10u);
    EXPECT_EQ(ps.matrix_cols, 20u);
    EXPECT_EQ(ps.nnz, 5u);
}

// ---------------------------------------------------------------------------
// print_matrix
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_PrintMatrix, SingleElement_ContainsValue) {
    Matrix<double> m(1, 1);
    m(0, 0) = 3.14;
    std::ostringstream oss;
    print_matrix(oss, m);
    const std::string out = oss.str();
    EXPECT_FALSE(out.empty());
    EXPECT_NE(out.find("3.14"), std::string::npos);
}

TEST(ReplInterpAdv_PrintMatrix, TwoByTwo_ContainsBrackets) {
    Matrix<double> m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;
    std::ostringstream oss;
    print_matrix(oss, m);
    const std::string out = oss.str();
    EXPECT_FALSE(out.empty());
    EXPECT_NE(out.find("["), std::string::npos);
}

TEST(ReplInterpAdv_PrintMatrix, EmptyMatrix_NoOutput) {
    Matrix<double> m(0, 0);
    std::ostringstream oss;
    print_matrix(oss, m);
    EXPECT_TRUE(oss.str().empty());
}

TEST(ReplInterpAdv_PrintMatrix, RowVector_ContainsValues) {
    Matrix<double> m(1, 3);
    m(0, 0) = 10.0; m(0, 1) = 20.0; m(0, 2) = 30.0;
    std::ostringstream oss;
    print_matrix(oss, m);
    const std::string out = oss.str();
    EXPECT_NE(out.find("10."), std::string::npos);
    EXPECT_NE(out.find("20."), std::string::npos);
    EXPECT_NE(out.find("30."), std::string::npos);
}

TEST(ReplInterpAdv_PrintMatrix, ThreeByOne_HasThreeRows) {
    Matrix<double> m(3, 1);
    m(0, 0) = 1.0;
    m(1, 0) = 2.0;
    m(2, 0) = 3.0;
    std::ostringstream oss;
    print_matrix(oss, m);
    const std::string out = oss.str();
    // Each row should produce a line with '[' and ']'
    size_t open_count = 0;
    for (char c : out) {
        if (c == '[') {
            ++open_count;
        }
    }
    EXPECT_EQ(open_count, 3u);
}

// ---------------------------------------------------------------------------
// SessionState struct fields
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_SessionState, InitialState_AllEmpty) {
    Interpreter interp;
    const auto& s = interp.state();
    EXPECT_TRUE(s.scalars.empty());
    EXPECT_TRUE(s.matrices.empty());
    EXPECT_TRUE(s.history.empty());
    EXPECT_FALSE(s.plot.valid);
}

TEST(ReplInterpAdv_SessionState, Scalars_PopulatedAfterAssign) {
    Interpreter interp;
    interp.assign_scalar("pi_approx", 3.14159);
    const auto& s = interp.state();
    EXPECT_EQ(s.scalars.size(), 1u);
    ASSERT_TRUE(s.scalars.count("pi_approx") > 0);
    EXPECT_NEAR(s.scalars.at("pi_approx"), 3.14159, 1e-10);
}

TEST(ReplInterpAdv_SessionState, Matrices_PopulatedAfterExecute) {
    Interpreter interp;
    interp.execute("M = [1, 2; 3, 4]");
    const auto& s = interp.state();
    ASSERT_TRUE(s.matrices.count("M") > 0);
    EXPECT_EQ(s.matrices.at("M").rows(), 2u);
    EXPECT_EQ(s.matrices.at("M").cols(), 2u);
}

TEST(ReplInterpAdv_SessionState, History_RecordsExecutedLines) {
    Interpreter interp;
    interp.execute("x = 5");
    interp.execute("y = 6");
    const auto& s = interp.state();
    ASSERT_EQ(s.history.size(), 2u);
    EXPECT_EQ(s.history[0], "x = 5");
    EXPECT_EQ(s.history[1], "y = 6");
}

TEST(ReplInterpAdv_SessionState, History_EmptyLineNotRecorded) {
    Interpreter interp;
    interp.execute("");
    interp.execute("   ");
    const auto& s = interp.state();
    EXPECT_TRUE(s.history.empty());
}

TEST(ReplInterpAdv_SessionState, Reset_ClearsAll) {
    Interpreter interp;
    interp.assign_scalar("a", 1.0);
    interp.execute("M = [1, 2; 3, 4]");
    interp.execute("b = 3");
    interp.reset();
    const auto& s = interp.state();
    EXPECT_TRUE(s.scalars.empty());
    EXPECT_TRUE(s.matrices.empty());
    EXPECT_TRUE(s.history.empty());
    EXPECT_FALSE(s.plot.valid);
}

TEST(ReplInterpAdv_SessionState, Plot_FieldInSessionState) {
    Interpreter interp;
    interp.execute("plot([5, 10, 15])");
    const auto& s = interp.state();
    // state().plot should match has_plot() and plot()
    EXPECT_EQ(s.plot.valid, interp.has_plot());
}

// ---------------------------------------------------------------------------
// try_parse_scalar_assignment (static)
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_TryParseScalarAssignment, SimpleInteger) {
    std::string name;
    double value = 0.0;
    EXPECT_TRUE(Interpreter::try_parse_scalar_assignment("x = 42", name, value));
    EXPECT_EQ(name, "x");
    EXPECT_NEAR(value, 42.0, 1e-12);
}

TEST(ReplInterpAdv_TryParseScalarAssignment, FloatingPoint) {
    std::string name;
    double value = 0.0;
    EXPECT_TRUE(Interpreter::try_parse_scalar_assignment("pi = 3.14159", name, value));
    EXPECT_EQ(name, "pi");
    EXPECT_NEAR(value, 3.14159, 1e-10);
}

TEST(ReplInterpAdv_TryParseScalarAssignment, WithWhitespace) {
    std::string name;
    double value = 0.0;
    EXPECT_TRUE(Interpreter::try_parse_scalar_assignment("  y  =  7.5  ", name, value));
    EXPECT_EQ(name, "y");
    EXPECT_NEAR(value, 7.5, 1e-12);
}

TEST(ReplInterpAdv_TryParseScalarAssignment, NoEquals_ReturnsFalse) {
    std::string name;
    double value = 0.0;
    EXPECT_FALSE(Interpreter::try_parse_scalar_assignment("x 42", name, value));
}

TEST(ReplInterpAdv_TryParseScalarAssignment, RhsIsExpression_ReturnsFalse) {
    std::string name;
    double value = 0.0;
    EXPECT_FALSE(Interpreter::try_parse_scalar_assignment("x = a + b", name, value));
}

TEST(ReplInterpAdv_TryParseScalarAssignment, RhsIsMatrix_ReturnsFalse) {
    std::string name;
    double value = 0.0;
    EXPECT_FALSE(Interpreter::try_parse_scalar_assignment("x = [1, 2]", name, value));
}

TEST(ReplInterpAdv_TryParseScalarAssignment, Zero) {
    std::string name;
    double value = 99.0;
    EXPECT_TRUE(Interpreter::try_parse_scalar_assignment("z = 0", name, value));
    EXPECT_EQ(name, "z");
    EXPECT_NEAR(value, 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// try_parse_scalar_binary_assignment (static)
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, TwoLiterals_Add) {
    ScalarBinaryAssign sa;
    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("z = 3 + 4", sa));
    EXPECT_EQ(sa.target, "z");
    EXPECT_EQ(sa.op, '+');
    EXPECT_TRUE(sa.left.is_literal);
    EXPECT_NEAR(sa.left.literal, 3.0, 1e-12);
    EXPECT_TRUE(sa.right.is_literal);
    EXPECT_NEAR(sa.right.literal, 4.0, 1e-12);
}

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, VarAndLiteral_Multiply) {
    ScalarBinaryAssign sa;
    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("r = x * 2", sa));
    EXPECT_EQ(sa.target, "r");
    EXPECT_EQ(sa.op, '*');
    EXPECT_FALSE(sa.left.is_literal);
    EXPECT_EQ(sa.left.name, "x");
    EXPECT_TRUE(sa.right.is_literal);
    EXPECT_NEAR(sa.right.literal, 2.0, 1e-12);
}

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, Division) {
    ScalarBinaryAssign sa;
    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("q = 10 / 2", sa));
    EXPECT_EQ(sa.op, '/');
}

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, Subtraction) {
    ScalarBinaryAssign sa;
    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("d = a - b", sa));
    EXPECT_EQ(sa.op, '-');
    EXPECT_EQ(sa.target, "d");
}

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, NoEquals_ReturnsFalse) {
    ScalarBinaryAssign sa;
    EXPECT_FALSE(Interpreter::try_parse_scalar_binary_assignment("x 3 + 4", sa));
}

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, PureScalar_NoOp_ReturnsFalse) {
    ScalarBinaryAssign sa;
    EXPECT_FALSE(Interpreter::try_parse_scalar_binary_assignment("x = 5", sa));
}

TEST(ReplInterpAdv_TryParseScalarBinaryAssignment, BothVarOperands) {
    ScalarBinaryAssign sa;
    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("c = a + b", sa));
    EXPECT_EQ(sa.target, "c");
    EXPECT_FALSE(sa.left.is_literal);
    EXPECT_EQ(sa.left.name, "a");
    EXPECT_FALSE(sa.right.is_literal);
    EXPECT_EQ(sa.right.name, "b");
}

// ---------------------------------------------------------------------------
// try_parse_scalar_expr_assignment (static)
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_TryParseScalarExprAssignment, SinCall_IsExpr) {
    std::string name, expr;
    bool ok = Interpreter::try_parse_scalar_expr_assignment("y = sin(x)", name, expr);
    if (ok) {
        EXPECT_EQ(name, "y");
        EXPECT_EQ(expr, "sin(x)");
    }
    SUCCEED();
}

TEST(ReplInterpAdv_TryParseScalarExprAssignment, PureLiteral_ReturnsFalse) {
    std::string name, expr;
    EXPECT_FALSE(Interpreter::try_parse_scalar_expr_assignment("x = 42", name, expr));
}

TEST(ReplInterpAdv_TryParseScalarExprAssignment, NoEquals_ReturnsFalse) {
    std::string name, expr;
    EXPECT_FALSE(Interpreter::try_parse_scalar_expr_assignment("x sin(y)", name, expr));
}

TEST(ReplInterpAdv_TryParseScalarExprAssignment, TwoVarBinaryExpr) {
    std::string name, expr;
    bool ok = Interpreter::try_parse_scalar_expr_assignment("r = a + b", name, expr);
    if (ok) {
        EXPECT_EQ(name, "r");
        EXPECT_EQ(expr, "a + b");
    }
    SUCCEED();
}

TEST(ReplInterpAdv_TryParseScalarExprAssignment, MatrixCallee_ReturnsFalse) {
    std::string name, expr;
    // "matmul(A, B)" is a matrix call, not a scalar expression
    EXPECT_FALSE(Interpreter::try_parse_scalar_expr_assignment("C = matmul(A, B)", name, expr));
}

TEST(ReplInterpAdv_TryParseScalarExprAssignment, EmptyRhs_ReturnsFalse) {
    std::string name, expr;
    EXPECT_FALSE(Interpreter::try_parse_scalar_expr_assignment("x = ", name, expr));
}

// ---------------------------------------------------------------------------
// try_parse_matrix_call_assignment (static)
// ---------------------------------------------------------------------------

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Matmul_TwoArgs) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("C = matmul(A, B)", ma));
    EXPECT_EQ(ma.target, "C");
    EXPECT_EQ(ma.callee, "matmul");
    ASSERT_EQ(ma.args.size(), 2u);
    EXPECT_EQ(ma.args[0], "A");
    EXPECT_EQ(ma.args[1], "B");
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Zeros_OneArg) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("Z = zeros(3)", ma));
    EXPECT_EQ(ma.callee, "zeros");
    ASSERT_EQ(ma.args.size(), 1u);
    EXPECT_EQ(ma.args[0], "3");
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Zeros_TwoArgs) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("Z = zeros(3, 4)", ma));
    EXPECT_EQ(ma.callee, "zeros");
    ASSERT_EQ(ma.args.size(), 2u);
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Eye_OneArg) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("I = eye(4)", ma));
    EXPECT_EQ(ma.callee, "eye");
    ASSERT_EQ(ma.args.size(), 1u);
    EXPECT_EQ(ma.args[0], "4");
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Ones_OneArg) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("O = ones(2)", ma));
    EXPECT_EQ(ma.callee, "ones");
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Transpose_OneArg) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("T = transpose(M)", ma));
    EXPECT_EQ(ma.callee, "transpose");
    ASSERT_EQ(ma.args.size(), 1u);
    EXPECT_EQ(ma.args[0], "M");
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Solve_TwoArgs) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("x = solve(A, b)", ma));
    EXPECT_EQ(ma.callee, "solve");
    ASSERT_EQ(ma.args.size(), 2u);
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, UnknownCallee_ReturnsFalse) {
    MatrixCallAssign ma;
    EXPECT_FALSE(Interpreter::try_parse_matrix_call_assignment("y = sin(x)", ma));
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, NoEquals_ReturnsFalse) {
    MatrixCallAssign ma;
    EXPECT_FALSE(Interpreter::try_parse_matrix_call_assignment("matmul(A, B)", ma));
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, MatmulWrongArity_ReturnsFalse) {
    MatrixCallAssign ma;
    // matmul requires exactly 2 args
    EXPECT_FALSE(Interpreter::try_parse_matrix_call_assignment("C = matmul(A)", ma));
}

TEST(ReplInterpAdv_TryParseMatrixCallAssignment, Inv_OneArg) {
    MatrixCallAssign ma;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("B = inv(A)", ma));
    EXPECT_EQ(ma.callee, "inv");
}
