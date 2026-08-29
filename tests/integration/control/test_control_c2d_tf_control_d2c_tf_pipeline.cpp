
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

TEST(IntegrationControl,  C2dD2cTfTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "control_c2d_tf(num,den,Ts)");
    expect_contains(interp, "help", "control_d2c_tf(num,den,Ts)");

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    EXPECT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(IntegrationControl,  JacobiSnScalar) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}
