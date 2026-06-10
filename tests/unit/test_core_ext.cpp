#include <cmath>

#include <gtest/gtest.h>

#include "ms/core/operations.hpp"
#include "ms/core/rng.hpp"
#include "ms/core/scalar.hpp"
#include "ms/core/sparse.hpp"
#include "ms/core/sym.hpp"
#include "ms/core/tensor.hpp"
#include "ms/domain/domain.hpp"
#include "ms/error/error_types.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/version.hpp"

using namespace ms;

[[MS_UNSAFE("test: MS_UNSAFE macro compile smoke")]]
static int ms_unsafe_macro_smoke() {
    return 42;
}

namespace {

int g_seed = 99;

double test_uniform() {
    g_seed = (g_seed * 1103515245 + 12345) & 0x7fffffff;
    return static_cast<double>(g_seed) / static_cast<double>(0x7fffffff);
}

} // namespace

TEST(CoreExtTest, sym_arithmetic) {
    Sym a("x");
    Sym b("y");
    const auto sum = a + b;
    const auto prod = a * b;
    EXPECT_NE(sum.to_string().find("x"), std::string::npos);
    EXPECT_NE(prod.to_string().find("*"), std::string::npos);
}

TEST(CoreExtTest, sym_compound_operators) {
    Sym a("x");
    Sym b("y");
    Sym c = a;
    c += b;
    c -= Sym("1");
    c *= Sym("2");
    c /= Sym("z");
    EXPECT_NE(c.to_string().find("/"), std::string::npos);
    const Sym diff = a - b;
    const Sym quot = a / b;
    EXPECT_NE(diff.to_string().find("-"), std::string::npos);
    EXPECT_NE(quot.to_string().find("/"), std::string::npos);
    Sym uneval("t");
    EXPECT_TRUE(std::isnan(uneval.eval()));
}

TEST(CoreExtTest, session_rng_uniform) {
    set_session_rng(test_uniform, nullptr);
    for (int i = 0; i < 32; ++i) {
        const double v = session_uniform();
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
    clear_session_rng();
}

TEST(CoreExtTest, sym_eval_numeric) {
    Sym value(6.0);
    EXPECT_DOUBLE_EQ(value.eval(), 6.0);
    EXPECT_DOUBLE_EQ(Sym("3.5").eval(), 3.5);
    EXPECT_TRUE(std::isnan(Sym("x").eval()));
    EXPECT_TRUE(std::isnan(Sym("x+y").eval()));
}

TEST(CoreExtTest, version_header) {
    EXPECT_EQ(VERSION_MAJOR, 1);
    EXPECT_EQ(VERSION_MINOR, 0);
    EXPECT_EQ(VERSION_PATCH, 0);
    EXPECT_EQ(VERSION_STRING, "1.0.0");
}

TEST(CoreExtTest, sparse_rectangular_spmv) {
    Sparse<double> S(2, 3, {0, 1}, {0, 2}, {1.0, 2.5});
    EXPECT_EQ(S.rows(), 2u);
    EXPECT_EQ(S.cols(), 3u);
    EXPECT_EQ(S.nnz(), 2u);
    ColMatrix<double> x(3, 1, 1.0);
    const auto y = S.spmv(x);
    EXPECT_DOUBLE_EQ(y(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(y(1, 0), 2.5);
}

TEST(CoreExtTest, domain_combinatorics) {
    EXPECT_EQ(factorial(5), 120u);
    EXPECT_EQ(nchoosek(6, 0), 1u);
    Graph g;
    g.n = 3;
    g.adj = {{1}, {0, 2}, {1}};
    EXPECT_EQ(graph_num_edges(g), 2u);
}

TEST(CoreExtTest, format_error_variants) {
    EXPECT_NE(format_error(DimensionMismatch{2, 3, 4, 5}).find("expected 4x5"), std::string::npos);
    EXPECT_EQ(format_error(SingularMatrix{}), "singular matrix");
    EXPECT_EQ(format_error(DomainError{"fn", "bad input"}), "fn: bad input");
    EXPECT_EQ(format_error(ConvergenceFail{10, 1e-3}), "convergence failed");
    EXPECT_EQ(format_error(ParseError{1, 2, "token"}), "token");
}

TEST(CoreExtTest, scalar_arithmetic) {
    const Scalar meters(3.0, "m");
    const Scalar seconds(2.0, "s");
    EXPECT_DOUBLE_EQ((meters + meters).value(), 6.0);
    EXPECT_DOUBLE_EQ((meters - seconds).value(), 1.0);
    EXPECT_DOUBLE_EQ((meters * seconds).value(), 6.0);
    EXPECT_EQ((meters * seconds).unit(), "ms");
    EXPECT_DOUBLE_EQ((meters / seconds).value(), 1.5);
}

TEST(CoreExtTest, session_rng_normal) {
    set_session_rng(test_uniform, []() { return 0.25; });
    EXPECT_TRUE(session_rng_active());
    EXPECT_DOUBLE_EQ(session_normal(), 0.25);
    clear_session_rng();
    EXPECT_FALSE(session_rng_active());
    EXPECT_DOUBLE_EQ(session_uniform(), 0.0);
}

TEST(CoreExtTest, tensor_shape) {
    Tensor<double> t2(std::vector<size_t>{2, 3});
    EXPECT_EQ(t2.dims(), 2u);
    EXPECT_EQ(t2.size(0), 2u);
    EXPECT_EQ(t2.size(1), 3u);
    EXPECT_EQ(t2.size(2), 0u);
    EXPECT_EQ(t2.total_size(), 6u);

    Tensor<double> t3(2, 3, 4);
    EXPECT_EQ(t3.dims(), 3u);
    EXPECT_EQ(t3.total_size(), 24u);
    t3.at(1, 2) = 7.0;
    EXPECT_DOUBLE_EQ(t3.at(1, 2), 7.0);
}

TEST(CoreExtTest, dispatch_execute_smoke) {
    const auto cpu = decide(256, ExecPolicy::CPU);
    execute(cpu);
    EXPECT_EQ(cpu.backend, Backend::CPU);
    EXPECT_EQ(get_policy_from_error(), ExecPolicy::CPU);
}

TEST(CoreExtTest, ms_unsafe_macro_compiles) {
    EXPECT_EQ(ms_unsafe_macro_smoke(), 42);
}

TEST(ExpmTest, NilpotentMatrix) {
    ColMatrix<double> A{{0, 1}, {0, 0}};
    const auto E = expm(A);
    ASSERT_TRUE(E.has_value());
    EXPECT_NEAR((*E)(0, 0), 1.0, 1e-9);
    EXPECT_NEAR((*E)(0, 1), 1.0, 1e-9);
    EXPECT_NEAR((*E)(1, 0), 0.0, 1e-9);
    EXPECT_NEAR((*E)(1, 1), 1.0, 1e-9);
}

TEST(ExpmTest, DiagonalMatchesExp) {
    ColMatrix<double> A{{1, 0}, {0, 2}};
    const auto E = expm(A);
    ASSERT_TRUE(E.has_value());
    EXPECT_NEAR((*E)(0, 0), std::exp(1.0), 2e-6);
    EXPECT_NEAR((*E)(1, 1), std::exp(2.0), 2e-6);
    EXPECT_NEAR((*E)(0, 1), 0.0, 1e-9);
    EXPECT_NEAR((*E)(1, 0), 0.0, 1e-9);
}

TEST(DetTest, DirectApi_2x2) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    const auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, -2.0, 1e-9);
}

TEST(DetTest, SingularReturnsError) {
    ColMatrix<double> A{{1, 2}, {2, 4}};
    EXPECT_FALSE(det(A).has_value());
}

TEST(TraceTest, DirectApi) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    const auto t = trace(A);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*t, 5.0, 1e-9);
}

TEST(TraceTest, NonSquareError) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(trace(A).has_value());
}

TEST(OnesTest, ShapeAndValues) {
    const ColMatrix<double> O = ones<double>(2, 3);
    EXPECT_EQ(O.rows(), 2u);
    EXPECT_EQ(O.cols(), 3u);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(O(i, j), 1.0);
        }
    }
}
