#include <gtest/gtest.h>
#include "ms/interp/repl_engine.hpp"

using namespace ms;
using namespace ms::interp;

// Division by zero in REPL: IEEE 754 may yield inf, or the engine may return an
// error — both are acceptable. Just ensure no crash.
TEST(ReplErrorPathTest, divide_by_zero) {
    Interpreter interp;
    const auto r = interp.execute("x = 1.0 / 0.0");
    (void)r;
}

// Referencing an undefined variable must produce an error result.
TEST(ReplErrorPathTest, undefined_variable) {
    Interpreter interp;
    const auto r = interp.execute("y = undefined_var + 1");
    EXPECT_FALSE(r.has_value());
}

// matmul of incompatible dimensions must fail.
TEST(ReplErrorPathTest, matmul_dimension_mismatch) {
    Interpreter interp;
    interp.execute("A = [1,2;3,4]");           // 2x2
    interp.execute("B = [1,2,3;4,5,6;7,8,9]"); // 3x3
    const auto r = interp.execute("C = matmul(A, B)");
    EXPECT_FALSE(r.has_value());
}

// Completely unknown command should not crash; result may be error or success.
TEST(ReplErrorPathTest, unknown_command) {
    Interpreter interp;
    const auto r = interp.execute("nonexistent_command foo bar");
    (void)r;
}

// Empty input should not crash.
TEST(ReplErrorPathTest, empty_input) {
    Interpreter interp;
    const auto r = interp.execute("");
    (void)r;
}

// Whitespace-only input should not crash.
TEST(ReplErrorPathTest, whitespace_input) {
    Interpreter interp;
    const auto r = interp.execute("   ");
    (void)r;
}

// Successive scalar assignments must each be independently stored.
TEST(ReplErrorPathTest, multiple_assignments_independent) {
    Interpreter interp;
    interp.execute("a = 1.0");
    interp.execute("b = 2.0");
    interp.execute("c = 3.0");
    EXPECT_NEAR(interp.state().scalars.at("a"), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().scalars.at("b"), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().scalars.at("c"), 3.0, 1e-12);
}

// Overwriting a scalar must update the stored value.
TEST(ReplErrorPathTest, scalar_overwrite) {
    Interpreter interp;
    interp.execute("x = 1.0");
    interp.execute("x = 42.0");
    EXPECT_NEAR(interp.state().scalars.at("x"), 42.0, 1e-12);
}
