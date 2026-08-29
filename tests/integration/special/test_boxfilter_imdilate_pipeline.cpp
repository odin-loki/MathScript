
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

TEST(IntegrationSpecial,  BoxfilterMorphTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "boxfilter(M,k)");
    expect_contains(interp, "help", "imdilate(M,k)");
    expect_contains(interp, "help", "imerode(M,k)");
    expect_contains(interp, "help", "imopen(M,k)");
    expect_contains(interp, "help", "imclose(M,k)");

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "D = imdilate(M, 3)");
    expect_ok(interp, "E = imerode(M, 3)");
    const auto& dil = interp.state().matrices.at("D");
    const auto& ero = interp.state().matrices.at("E");
    double dmax = 0.0;
    double emax = 0.0;
    bool d_nonzero = false;
    for (std::size_t i = 0; i < dil.rows(); ++i) {
        for (std::size_t j = 0; j < dil.cols(); ++j) {
            if (dil(i, j) > dmax) {
                dmax = dil(i, j);
            }
            if (dil(i, j) != 0.0) {
                d_nonzero = true;
            }
        }
    }
    for (std::size_t i = 0; i < ero.rows(); ++i) {
        for (std::size_t j = 0; j < ero.cols(); ++j) {
            if (ero(i, j) > emax) {
                emax = ero(i, j);
            }
        }
    }
    EXPECT_TRUE(dmax > emax || (dmax >= emax && d_nonzero));

    expect_ok(interp, "O = imopen(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 5u);
    expect_ok(interp, "C = imclose(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 5u);
}

TEST(IntegrationSpecial,  JacobiSdScalar) {
    Interpreter interp;
    expect_ok(interp, "jsd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}
