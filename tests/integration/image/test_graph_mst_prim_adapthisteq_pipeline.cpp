
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"
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

TEST(IntegrationImage,  MstPrimAdapthisteqTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_mst_prim(A)");
    expect_contains(interp, "help", "adapthisteq(M)");

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(IntegrationImage,  ProbPoisPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}
