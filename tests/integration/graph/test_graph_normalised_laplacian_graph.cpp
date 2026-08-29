
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

TEST(IntegrationGraph,  NormLapEccTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_normalised_laplacian(A)");
    expect_contains(interp, "help", "graph_eccentricity(A)");

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(IntegrationGraph,  NumthyLcmScalar) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}
