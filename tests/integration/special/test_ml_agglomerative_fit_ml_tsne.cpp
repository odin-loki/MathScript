
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

TEST(IntegrationSpecial,  MlAggloTsneGolombTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_agglomerative_fit");
    expect_contains(interp, "help", "ml_tsne_fit");
    expect_contains(interp, "help", "golomb_rice_encode_vec");

    expect_ok(interp, "A = [0,0; 1,0; 2,0; 3,0; 4,0; 100,0; 101,0; 102,0; 103,0; 104,0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
    const double ac0 = interp.state().matrices.at("ac_l")(0, 0);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ac_l")(i, 0), ac0, 1e-9);
    }
    const double ac1 = interp.state().matrices.at("ac_l")(5, 0);
    for (size_t i = 5; i < 10; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ac_l")(i, 0), ac1, 1e-9);
    }
    EXPECT_NE(ac0, ac1);

    expect_ok(interp, "T = [0,0; 0.1,0; 0.2,0; 10,10; 10.1,10; 10.2,10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0;1;2;5;10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("GR")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("GR")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("GR")(2, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("GR")(3, 0), 5.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("GR")(4, 0), 10.0, 1e-9);
}

TEST(IntegrationSpecial,  BesselJScalar) {
    Interpreter interp;

    expect_ok(interp, "j = bessel_j(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("j"), ms::bessel_j(1, 1.0), 1e-8);
}
