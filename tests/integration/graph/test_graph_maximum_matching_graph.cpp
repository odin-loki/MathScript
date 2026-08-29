
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/cplx/cplx.hpp"
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

TEST(IntegrationGraph,  MatchingClosureTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_maximum_matching");
    expect_contains(interp, "help", "graph_transitive_closure(A)");

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(IntegrationGraph,  JoukowskiScalar) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}
