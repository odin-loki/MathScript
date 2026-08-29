
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

TEST(IntegrationCore,  SparseSpmvDenseAddTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "sparse_spmv");
    expect_contains(interp, "help", "sparse_to_dense");
    expect_contains(interp, "help", "sparse_add");

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-9);

    expect_ok(interp, "D = sparse_to_dense(A)");
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-9);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(IntegrationCore,  LegendrePScalar) {
    Interpreter interp;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, 0.5), 1e-8);
}
