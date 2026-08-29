#include <gtest/gtest.h>
#include "ms/interp/jit_backend.hpp"

#include <cmath>

using namespace ms::interp;

TEST(JitBackendTest, default_backend_is_repl) {
    auto backend = create_backend();
    EXPECT_EQ(backend->backend_name(), "repl");
}

TEST(JitBackendTest, orc_jit_stub_backend_name) {
    auto backend = create_backend(Backend::OrcJit);
    const auto caps = backend->capabilities();
    if (caps.llvm_linked) {
        EXPECT_TRUE(backend->backend_name() == "orc-jit" ||
                    backend->backend_name() == "orc-jit-llvm-fallback");
    } else {
        EXPECT_EQ(backend->backend_name(), "orc-jit-stub");
    }
}

TEST(JitBackendTest, orc_jit_stub_executes_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    auto result = backend->execute("x = 42");
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("42"), std::string::npos);
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("x"), 42.0);
}

TEST(JitBackendTest, try_parse_scalar_assignment) {
    std::string name;
    double value = 0.0;
    EXPECT_TRUE(Interpreter::try_parse_scalar_assignment("alpha = 3.5", name, value));
    EXPECT_EQ(name, "alpha");
    EXPECT_DOUBLE_EQ(value, 3.5);
    EXPECT_FALSE(Interpreter::try_parse_scalar_assignment("plot([1, 2])", name, value));
    EXPECT_FALSE(Interpreter::try_parse_scalar_assignment("= 1", name, value));
}

TEST(JitBackendTest, try_parse_scalar_binary_assignment) {
    ScalarBinaryAssign assign;
    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("z = 2 + 3", assign));
    EXPECT_EQ(assign.target, "z");
    EXPECT_EQ(assign.op, '+');
    EXPECT_TRUE(assign.left.is_literal);
    EXPECT_DOUBLE_EQ(assign.left.literal, 2.0);
    EXPECT_TRUE(assign.right.is_literal);
    EXPECT_DOUBLE_EQ(assign.right.literal, 3.0);

    EXPECT_TRUE(Interpreter::try_parse_scalar_binary_assignment("y = x * 4", assign));
    EXPECT_FALSE(assign.left.is_literal);
    EXPECT_EQ(assign.left.name, "x");
    EXPECT_EQ(assign.op, '*');
}

TEST(JitBackendTest, scalar_binary_assignment) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("a = 2").has_value());
    auto result = interp.execute("b = a * 3");
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("b"), 6.0);
}

TEST(JitBackendTest, scalar_expr_precedence) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("c = 1 + 2 * 3").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("c"), 7.0);
    ASSERT_TRUE(interp.execute("d = 10 - 2 - 3").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("d"), 5.0);
    ASSERT_TRUE(interp.execute("e = (1 + 2) * 3").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("e"), 9.0);
}

TEST(JitBackendTest, scalar_expr_divide_by_zero) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("z = 1 / 0").has_value());
}

TEST(JitBackendTest, scalar_unary_call) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("y = sin(0)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 0.0);
    ASSERT_TRUE(interp.execute("z = sqrt(9)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("z"), 3.0);
    ASSERT_TRUE(interp.execute("w = 1 + cos(0)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("w"), 2.0);
    ASSERT_TRUE(interp.execute("a = 1.5").has_value());
    ASSERT_TRUE(interp.execute("b = sin(a)").has_value());
    EXPECT_NEAR(interp.state().scalars.at("b"), std::sin(1.5), 1e-12);
    EXPECT_FALSE(interp.execute("bad = foo(1)").has_value());
    ASSERT_TRUE(interp.execute("p = pow(2, 10)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("p"), 1024.0);
    ASSERT_TRUE(interp.execute("m = min(3, 7)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("m"), 3.0);
}

TEST(JitBackendTest, orc_jit_binary_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("a = 2").has_value());
    auto result = backend->execute("b = a + 1.5");
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("b"), 3.5);
}

TEST(JitBackendTest, orc_jit_scalar_expr_precedence) {
    auto backend = create_backend(Backend::OrcJit);
    const auto caps = backend->capabilities();
    if (!caps.can_compile) {
        GTEST_SKIP() << "LLVM JIT compile unavailable";
    }
    ASSERT_TRUE(backend->execute("c = 1 + 2 * 3").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("c"), 7.0);
    ASSERT_TRUE(backend->execute("a = 10").has_value());
    ASSERT_TRUE(backend->execute("d = a - 2 - 3").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("d"), 5.0);
}

TEST(JitBackendTest, try_parse_scalar_expr_assignment) {
    std::string name;
    std::string expr;
    EXPECT_TRUE(Interpreter::try_parse_scalar_expr_assignment("z = 1 + 2 * 3", name, expr));
    EXPECT_EQ(name, "z");
    EXPECT_EQ(expr, "1 + 2 * 3");
    EXPECT_FALSE(Interpreter::try_parse_scalar_expr_assignment("x = 42", name, expr));
    EXPECT_EQ(Interpreter::list_scalar_expr_variables("a + b * 2"),
              (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(Interpreter::list_scalar_expr_variables("sin(x) + 1"), (std::vector<std::string>{"x"}));
    EXPECT_EQ(Interpreter::list_scalar_expr_variables("pow(a, b)"), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(Interpreter::list_scalar_expr_variables("-a + b"), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(Interpreter::list_scalar_expr_variables("+a - b"), (std::vector<std::string>{"a", "b"}));
}

TEST(JitBackendTest, orc_jit_unary_minus_scalar_expr) {
    auto backend = create_backend(Backend::OrcJit);
    const auto caps = backend->capabilities();
    if (!caps.can_compile) {
        GTEST_SKIP() << "LLVM JIT compile unavailable";
    }
    ASSERT_TRUE(backend->execute("pi = 3.14159265358979323846").has_value());
    ASSERT_TRUE(backend->execute("x = -sin(pi/2)").has_value());
    EXPECT_NEAR(backend->state().scalars.at("x"), -1.0, 1e-12);
    ASSERT_TRUE(backend->execute("y = 2 * -3").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("y"), -6.0);
    ASSERT_TRUE(backend->execute("a = 5").has_value());
    ASSERT_TRUE(backend->execute("b = 0").has_value());
    ASSERT_TRUE(backend->execute("z = -a + b").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("z"), -5.0);
}

TEST(JitBackendTest, orc_jit_scalar_call) {
    auto backend = create_backend(Backend::OrcJit);
    const auto caps = backend->capabilities();
    if (!caps.can_compile) {
        GTEST_SKIP() << "LLVM JIT compile unavailable";
    }
    ASSERT_TRUE(backend->execute("y = sqrt(16)").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("y"), 4.0);
    ASSERT_TRUE(backend->execute("z = 1 + sin(0)").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("z"), 1.0);
    ASSERT_TRUE(backend->execute("p = pow(2, 8)").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("p"), 256.0);
}

TEST(JitBackendTest, try_parse_matrix_call_assignment) {
    MatrixCallAssign assign;
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("C = matmul(A, B)", assign));
    EXPECT_EQ(assign.target, "C");
    EXPECT_EQ(assign.callee, "matmul");
    EXPECT_EQ(assign.args.size(), 2u);
    EXPECT_EQ(assign.args[0], "A");
    EXPECT_EQ(assign.args[1], "B");
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("x = solve(A, b)", assign));
    EXPECT_EQ(assign.target, "x");
    EXPECT_EQ(assign.callee, "solve");
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("T = transpose(A)", assign));
    EXPECT_EQ(assign.callee, "transpose");
    EXPECT_EQ(assign.args, (std::vector<std::string>{"A"}));
    EXPECT_TRUE(Interpreter::try_parse_matrix_call_assignment("L = chol(S)", assign));
    EXPECT_EQ(assign.callee, "chol");
    EXPECT_FALSE(Interpreter::try_parse_matrix_call_assignment("C = A * B", assign));
}

TEST(JitBackendTest, orc_jit_matmul_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("A = [1, 2; 3, 4]").has_value());
    auto result = backend->execute("C = matmul(A, A)");
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(backend->state().matrices.at("C")(0, 0), 7.0);
}

TEST(JitBackendTest, orc_jit_solve_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("A = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(backend->execute("b = [1; 1]").has_value());
    auto result = backend->execute("x = solve(A, b)");
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(backend->state().matrices.at("x")(0, 0), 0.2, 1e-9);
    EXPECT_NEAR(backend->state().matrices.at("x")(1, 0), 0.4, 1e-9);
}

TEST(JitBackendTest, orc_jit_unary_matrix_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("A = [1, 2; 3, 4]").has_value());
    ASSERT_TRUE(backend->execute("T = transpose(A)").has_value());
    EXPECT_DOUBLE_EQ(backend->state().matrices.at("T")(0, 1), 3.0);
    ASSERT_TRUE(backend->execute("S = [4, 2; 2, 3]").has_value());
    ASSERT_TRUE(backend->execute("L = chol(S)").has_value());
    EXPECT_NEAR(backend->state().matrices.at("L")(0, 0), 2.0, 1e-9);
}

TEST(JitBackendTest, scalar_matrix_call_assignment) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(interp.execute("d = det(A)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("d"), 5.0);
    ASSERT_TRUE(interp.execute("t = trace(A)").has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("t"), 5.0);
    ScalarMatrixCallAssign sm;
    EXPECT_TRUE(Interpreter::try_parse_scalar_matrix_call_assignment("n = norm(A)", sm));
    EXPECT_EQ(sm.callee, "norm");
}

TEST(JitBackendTest, orc_jit_scalar_matrix_call) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("A = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(backend->execute("d = det(A)").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("d"), 5.0);
}

TEST(JitBackendTest, try_parse_multi_matrix_call_assignment) {
    MultiMatrixCallAssign assign;
    EXPECT_TRUE(Interpreter::try_parse_multi_matrix_call_assignment("L, U, P = lu(A)", assign));
    EXPECT_EQ(assign.targets.size(), 3u);
    EXPECT_EQ(assign.callee, "lu");
    EXPECT_TRUE(Interpreter::try_parse_multi_matrix_call_assignment("Q, R = qr(A)", assign));
    EXPECT_EQ(assign.targets.size(), 2u);
    EXPECT_EQ(assign.callee, "qr");
    EXPECT_TRUE(Interpreter::try_parse_multi_matrix_call_assignment("U, S, V = svd(A)", assign));
    EXPECT_EQ(assign.callee, "svd");
    EXPECT_TRUE(Interpreter::try_parse_multi_matrix_call_assignment("D, V = eig_sym(S)", assign));
    EXPECT_EQ(assign.callee, "eig_sym");
    EXPECT_FALSE(Interpreter::try_parse_multi_matrix_call_assignment("L = lu(A)", assign));
}

TEST(JitBackendTest, orc_jit_multi_matrix_call) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("M = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(backend->execute("L, U = lu(M)").has_value());
    EXPECT_TRUE(backend->state().matrices.count("L") > 0);
    EXPECT_TRUE(backend->state().matrices.count("U") > 0);
}

TEST(JitBackendTest, orc_jit_svd_eig_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("M = [3, 1; 1, 2]").has_value());
    ASSERT_TRUE(backend->execute("U, S = svd(M)").has_value());
    EXPECT_TRUE(backend->state().matrices.count("S") > 0);
    ASSERT_TRUE(backend->execute("S = [4, 1; 1, 3]").has_value());
    ASSERT_TRUE(backend->execute("Evals, Evecs = eig_sym(S)").has_value());
    EXPECT_TRUE(backend->state().matrices.count("Evals") > 0);
    EXPECT_TRUE(backend->state().matrices.count("Evecs") > 0);
}

TEST(JitBackendTest, repl_backend_capabilities) {
    auto backend = create_backend();
    auto caps = backend->capabilities();
    EXPECT_FALSE(caps.llvm_linked);
    EXPECT_FALSE(caps.can_compile);
    EXPECT_EQ(caps.notes, "interpreted REPL");
}

TEST(JitBackendTest, orc_jit_stub_capabilities) {
    auto backend = create_backend(Backend::OrcJit);
    auto caps = backend->capabilities();
    if (caps.llvm_linked) {
        EXPECT_TRUE(backend->backend_name() == "orc-jit" ||
                    backend->backend_name() == "orc-jit-llvm-fallback");
        if (caps.can_compile) {
            EXPECT_NE(caps.notes.find("scalar exprs: LLVM IR"), std::string::npos);
            EXPECT_NE(caps.notes.find("native C++ dispatch"), std::string::npos);
        } else {
            EXPECT_FALSE(caps.can_compile);
            EXPECT_NE(caps.notes.find("LLVM"), std::string::npos);
        }
    } else {
        EXPECT_EQ(backend->backend_name(), "orc-jit-stub");
        EXPECT_FALSE(caps.can_compile);
        EXPECT_EQ(caps.notes, "ORC JIT stub — delegates to REPL until LLVM ORC v2 is wired");
    }
}

TEST(JitBackendTest, reset_clears_orc_state) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("x = 42").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("x"), 42.0);

    backend->reset();

    EXPECT_TRUE(backend->state().scalars.empty());
    EXPECT_TRUE(backend->state().matrices.empty());
}

TEST(JitBackendTest, execute_after_reset) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("x = 42").has_value());
    backend->reset();
    ASSERT_TRUE(backend->execute("y = 7").has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("y"), 7.0);
}

TEST(JitBackendTest, capabilities_not_empty) {
    auto backend = create_backend(Backend::OrcJit);
    const auto caps = backend->capabilities();
    EXPECT_FALSE(caps.notes.empty());
}

TEST(JitBackendTest, orc_jit_negative_parse) {
    auto backend = create_backend(Backend::OrcJit);
    const auto result = backend->execute("not valid @@@ input");
    EXPECT_FALSE(result.has_value());
}
