
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

TEST(IntegrationSpecial,  NaiveBayesFitPredict) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_naive_bayes_fit");
    expect_contains(interp, "help", "ml_naive_bayes_predict");

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0; 0; 0; 1; 1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(IntegrationSpecial,  ChebyshevUScalar) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}
