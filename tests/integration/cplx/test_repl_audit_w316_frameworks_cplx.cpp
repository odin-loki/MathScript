// MathScript Integration Tests: REPL Interpreter – Audit Wave 316 Frameworks/Cplx Pipeline
//
// Inventory: leftover cplx contour/residue/argument_principle need CFunc + contour
// (same skip as audit Wave 314). Leftover izaac/cellai/cypha/axiom session APIs need
// session-object design. Already bound: cplx_mobius_re, cplx_cauchy_integral, izaac_*,
// gria_*, cellai_*.

#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(ReplAuditW316FrameworksCplxPipeline, MobiusIdentityFixesZ) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_mobius_re(a,b,c,d,zre,zim)");

    // Identity Möbius (az+b)/(cz+d) with a=d=1, b=c=0 leaves z unchanged.
    expect_ok(interp, "w = cplx_mobius_re(1, 0, 0, 1, 2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("w"), 2.0, 1e-8);
}
