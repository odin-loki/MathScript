#include <cmath>
#include <gtest/gtest.h>
#include <string>

#include "ms/interp/jit_backend.hpp"

using namespace ms::interp;

namespace {

void expect_scalar_parity(JitBackend& repl, JitBackend& jit, const std::string& name, double tol) {
    ASSERT_TRUE(repl.state().scalars.count(name) > 0);
    ASSERT_TRUE(jit.state().scalars.count(name) > 0);
    EXPECT_NEAR(repl.state().scalars.at(name), jit.state().scalars.at(name), tol);
}

} // namespace

TEST(JitReplParity, scalar_expr_parity) {
    auto repl = create_backend(Backend::Repl);
    auto jit = create_backend(Backend::OrcJit);

    for (const auto& cmd : {"x = sin(1.0)", "y = cos(2.0)", "z = x + y"}) {
        ASSERT_TRUE(repl->execute(cmd).has_value());
        ASSERT_TRUE(jit->execute(cmd).has_value());
    }

    expect_scalar_parity(*repl, *jit, "z", 1e-12);
}

TEST(JitReplParity, matrix_call_parity) {
    auto repl = create_backend(Backend::Repl);
    auto jit = create_backend(Backend::OrcJit);

    const std::string mat_cmd = "M = [3, 1; 1, 2]";
    const std::string det_cmd = "d = det(M)";

    ASSERT_TRUE(repl->execute(mat_cmd).has_value());
    ASSERT_TRUE(repl->execute(det_cmd).has_value());
    ASSERT_TRUE(jit->execute(mat_cmd).has_value());
    ASSERT_TRUE(jit->execute(det_cmd).has_value());

    expect_scalar_parity(*repl, *jit, "d", 1e-12);
}
