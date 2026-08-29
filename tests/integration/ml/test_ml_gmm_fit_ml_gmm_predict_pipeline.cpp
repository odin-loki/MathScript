
#include <gtest/gtest.h>
#include <cmath>
#include <set>
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

TEST(IntegrationMl,  MlGmmDbscanSpectralTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_gmm_fit");
    expect_contains(interp, "help", "ml_dbscan_fit");
    expect_contains(interp, "help", "ml_spectral_clustering");

    expect_ok(interp, "X = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(X, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(X, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);
    const double g0 = interp.state().matrices.at("gmm_l")(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("gmm_l")(i, 0), g0, 1e-9);
    }
    const double g1 = interp.state().matrices.at("gmm_l")(3, 0);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("gmm_l")(i, 0), g1, 1e-9);
    }
    EXPECT_NE(g0, g1);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(X, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
    EXPECT_GE(interp.state().matrices.at("gmm_p").cols(), 2u);
    for (size_t i = 0; i < 6; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < interp.state().matrices.at("gmm_p").cols(); ++j) {
            row_sum += interp.state().matrices.at("gmm_p")(i, j);
        }
        EXPECT_NEAR(row_sum, 1.0, 1e-6);
    }

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);
    std::set<int> db_clusters;
    for (size_t i = 0; i < 10; ++i) {
        const int label = static_cast<int>(interp.state().matrices.at("db_l")(i, 0));
        if (label >= 0) {
            db_clusters.insert(label);
        }
    }
    EXPECT_EQ(db_clusters.size(), 2u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);
    std::set<int> sp_labels;
    for (size_t i = 0; i < 8; ++i) {
        sp_labels.insert(static_cast<int>(interp.state().matrices.at("sp_l")(i, 0)));
    }
    EXPECT_GE(sp_labels.size(), 2u);
}

TEST(IntegrationMl,  SpheroidalS1Scalar) {
    Interpreter interp;

    expect_ok(interp, "s1 = spheroidal_s1(1, 0, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), ms::spheroidal_s1(1, 0, 0.1, 0.5), 1e-6);
}
