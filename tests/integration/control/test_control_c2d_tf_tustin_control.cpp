
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

TEST(IntegrationControl,  ControlC2dD2cTfTustin) {
    Interpreter interp;
    expect_contains(interp, "help", "control_c2d_tf_tustin(num,den,Ts)");

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    const auto& dt = interp.state().matrices.at("Dt");
    EXPECT_EQ(dt.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    EXPECT_GT(interp.state().matrices.at("Ct").rows(), 0u);
}

TEST(IntegrationControl,  StruveKScalar) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}
