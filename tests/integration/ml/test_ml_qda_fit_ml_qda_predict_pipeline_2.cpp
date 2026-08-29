
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationMl,  MlQdaSvmTail18) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_qda_fit(X,y)");
    expect_contains(interp, "help", "ml_svm_fit(X,y");

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0,0; 3,1], qda_m)");
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2,0; -1,0; 1,0; 2,0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5,0; 1.5,0], svm_m)");
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(IntegrationMl,  JacobiDsScalar) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.5, 0.5), 1e-8);
}
