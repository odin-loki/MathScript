
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

TEST(IntegrationMl,  MlKmeansQdaSvmTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_kmeans_fit(X,k)");
    expect_contains(interp, "help", "ml_qda_fit(X,y)");
    expect_contains(interp, "help", "ml_svm_fit(X,y");

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
    const double l0 = interp.state().matrices.at("labels")(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("labels")(i, 0), l0, 1e-9);
    }
    const double l1 = interp.state().matrices.at("labels")(3, 0);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("labels")(i, 0), l1, 1e-9);
    }
    EXPECT_NE(l0, l1);

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

TEST(IntegrationMl,  MathieuThreeArgScalar) {
    Interpreter interp;

    expect_ok(interp, "ce = mathieu_ce(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ce"), ms::mathieu_ce(1, 0.1, 0.5), 1e-8);

    expect_ok(interp, "se = mathieu_se(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("se"), ms::mathieu_se(1, 0.1, 0.5), 1e-8);

    expect_ok(interp, "mc = mathieu_mc(0, 0.1, 0.0)");
    EXPECT_NEAR(interp.state().scalars.at("mc"), ms::mathieu_mc(0, 0.1, 0.0), 1e-6);

    expect_ok(interp, "ms_ = mathieu_ms(1, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ms_"), ms::mathieu_ms(1, 0.1, 0.5), 1e-8);
}
