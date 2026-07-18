// MathScript Integration Tests: REPL Interpreter – Wave 234 Bindings Pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

void expect_near_matrix_entry(Interpreter& interp, const std::string& assign_cmd, size_t row,
                              double expected, double tol) {
    expect_ok(interp, assign_cmd);
    const std::string name = Interpreter::trim(assign_cmd.substr(0, assign_cmd.find('=')));
    const auto& m = interp.state().matrices.at(name);
    ASSERT_GT(m.rows(), row);
    EXPECT_NEAR(m(row, 0), expected, tol) << assign_cmd;
}

} // namespace

TEST(ReplWave234Pipeline, ImageMorphologyBindings) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "D = imdilate(M, 3)");
    expect_ok(interp, "E = imerode(M, 3)");
    expect_ok(interp, "O = imopen(M, 3)");
    expect_ok(interp, "C = imclose(M, 3)");

    const auto& dil = interp.state().matrices.at("D");
    const auto& ero = interp.state().matrices.at("E");
    EXPECT_GE(dil(1, 2), ero(1, 2));

    expect_contains(interp, "help", "imdilate(M,k)");
    expect_contains(interp, "help", "imerode(M,k)");
    expect_contains(interp, "help", "imopen(M,k)");
    expect_contains(interp, "help", "imclose(M,k)");
}

TEST(ReplWave234Pipeline, Rgb2HsvBinding) {
    Interpreter interp;

    expect_ok(interp, "RGB = [1,0,0; 0,1,0; 0,0,1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB)");
    const auto& hsv = interp.state().matrices.at("HSV");
    ASSERT_EQ(hsv.rows(), 3u);
    ASSERT_EQ(hsv.cols(), 3u);
    EXPECT_NEAR(hsv(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(hsv(1, 0), 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(hsv(2, 0), 2.0 / 3.0, 1e-6);

    expect_contains(interp, "help", "rgb2hsv(M)");
}

TEST(ReplWave234Pipeline, LinearRegressionBindings) {
    Interpreter interp;

    expect_ok(interp, "X = [1;2;3;4;5]");
    expect_ok(interp, "y = [3;5;7;9;11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_near_matrix_entry(interp, "pred = ml_linear_predict([6], model)", 0, 13.0, 0.2);

    expect_contains(interp, "help", "ml_linear_fit(X,y)");
    expect_contains(interp, "help", "ml_linear_predict(X,model)");
}

TEST(ReplWave234Pipeline, RidgeRegressionBindings) {
    Interpreter interp;

    expect_ok(interp, "X = [0;1;2;3]");
    expect_ok(interp, "y = [1;3;5;7]");
    expect_ok(interp, "model = ml_ridge_fit(X, y, 0.001)");
    expect_near_matrix_entry(interp, "pred = ml_ridge_predict([4], model)", 0, 9.0, 0.5);

    expect_contains(interp, "help", "ml_ridge_fit(X,y,alpha)");
    expect_contains(interp, "help", "ml_ridge_predict(X,model)");
}

TEST(ReplWave234Pipeline, LogisticRegressionBindings) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    expect_ok(interp, "pred_pos = ml_logistic_predict([2], model)");
    expect_ok(interp, "pred_neg = ml_logistic_predict([-2], model)");

    const auto& pos = interp.state().matrices.at("pred_pos");
    const auto& neg = interp.state().matrices.at("pred_neg");
    EXPECT_GT(pos(0, 0), neg(0, 0));

    expect_contains(interp, "help", "ml_logistic_fit(X,y)");
    expect_contains(interp, "help", "ml_logistic_predict(X,model)");
}
