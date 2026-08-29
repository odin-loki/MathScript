
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

TEST(IntegrationQuantum,  KetBasisFockStateTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_ket_basis");
    expect_contains(interp, "help", "quantum_fock_state");

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    EXPECT_EQ(interp.state().matrices.at("kb").rows(), 2u);

    expect_ok(interp, "fs = quantum_fock_state(1, 3)");
    EXPECT_EQ(interp.state().matrices.at("fs").rows(), 4u);
}

TEST(IntegrationQuantum,  PolylogScalar) {
    Interpreter interp;
    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), ms::polylog(2, 0.25), 1e-8);
}
