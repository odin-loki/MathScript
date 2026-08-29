#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, poly_deriv) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_deriv(coeffs)");

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pd").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pd")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pd")(1, 0), 6.0, 1e-9);
}

TEST(ReplCommandsTest, poly_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_eval(coeffs,x)");

    expect_ok(interp, "pe = poly_eval([1; -1; 1], 2)");
    EXPECT_NEAR(interp.state().scalars.at("pe"), 3.0, 1e-9);

    expect_contains(interp, "poly_eval([1; -1; 1], 2)", "3");
}

TEST(ReplCommandsTest, poly_integ) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_integ(coeffs,c)");

    expect_ok(interp, "pi = poly_integ([3], 0)");
    ASSERT_GT(interp.state().matrices.count("pi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pi").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pi")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pi")(1, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, poly_add) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_add(a,b)");

    expect_ok(interp, "s = poly_add([1; 2], [3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("s")(0, 0), 4.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s")(1, 0), 6.0, 1e-9);
}

TEST(ReplCommandsTest, poly_mul) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_mul(a,b)");

    expect_ok(interp, "p = poly_mul([1; 2], [3; 4])");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("p")(1, 0), 10.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("p")(2, 0), 8.0, 1e-9);
}

TEST(ReplCommandsTest, poly_sub) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_sub(a,b)");

    expect_ok(interp, "d = poly_sub([5; 3], [2; 1])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("d")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, poly_compose) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_compose(p,q)");

    expect_ok(interp, "c = poly_compose([1; 1], [0; 2])");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, poly_bernstein) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_bernstein(n,i,x)");

    // B_{2,1}(0.5) = C(2,1)*0.5*0.5 = 0.5.
    expect_ok(interp, "bn = poly_bernstein(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bn"), 0.5, 1e-9);

    expect_contains(interp, "poly_bernstein(2, 1, 0.5)", "0.5");
}

TEST(ReplCommandsTest, poly_resultant) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_resultant(p,q)");

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "q = [10; -7; 1]");
    expect_ok(interp, "r = poly_resultant(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("r"), 0.0, 1e-6);

    expect_contains(interp, "poly_resultant([6; -5; 1], [10; -7; 1])", "0");
}

TEST(ReplCommandsTest, poly_discriminant) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_discriminant(p)");

    expect_ok(interp, "sq = [1; -2; 1]");
    expect_ok(interp, "d = poly_discriminant(sq)");
    EXPECT_NEAR(interp.state().scalars.at("d"), 0.0, 1e-6);

    expect_contains(interp, "poly_discriminant([1; -2; 1])", "0");
}

TEST(ReplCommandsTest, poly_lagrange_newton) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_lagrange(xs,ys)");
    expect_contains(interp, "help", "poly_interp_newton(xs,ys)");

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "p = poly_lagrange(xs, ys)");
    expect_ok(interp, "v = poly_eval(p, 1)");
    EXPECT_NEAR(interp.state().scalars.at("v"), 2.0, 1e-6);

    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    expect_ok(interp, "vn = poly_eval(pn, 1)");
    EXPECT_NEAR(interp.state().scalars.at("vn"), 2.0, 1e-6);
}

TEST(ReplCommandsTest, poly_roots_fit_gcd) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_roots(p)");
    expect_contains(interp, "help", "poly_fit(xs,ys,degree)");
    expect_contains(interp, "help", "poly_interp_hermite(xs,ys,dys)");
    expect_contains(interp, "help", "poly_gcd(a,b)");
    expect_contains(interp, "help", "poly_squarefree(p)");

    // (x-2)(x-3) = x^2 - 5x + 6
    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rts").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rts").cols(), 2u);
    {
        std::vector<double> reals = {interp.state().matrices.at("rts")(0, 0),
                                     interp.state().matrices.at("rts")(1, 0)};
        std::sort(reals.begin(), reals.end());
        EXPECT_NEAR(reals[0], 2.0, 1e-5);
        EXPECT_NEAR(reals[1], 3.0, 1e-5);
        EXPECT_NEAR(interp.state().matrices.at("rts")(0, 1), 0.0, 1e-5);
        EXPECT_NEAR(interp.state().matrices.at("rts")(1, 1), 0.0, 1e-5);
    }

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_EQ(interp.state().matrices.at("c").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("c")(1, 0), 2.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    expect_ok(interp, "vh = poly_eval(ph, 2)");
    EXPECT_NEAR(interp.state().scalars.at("vh"), 5.0, 1e-6);

    // gcd((x-2)(x-3), (x-2)(x-5)) ~ (x-2)
    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
    ASSERT_GE(interp.state().matrices.at("g").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("g")(0, 0), -2.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    // square-free part of (x-2)^2(x-3)
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    expect_ok(interp, "v2 = poly_eval(sf, 2)");
    expect_ok(interp, "v3 = poly_eval(sf, 3)");
    EXPECT_NEAR(interp.state().scalars.at("v2"), 0.0, 1e-4);
    EXPECT_NEAR(interp.state().scalars.at("v3"), 0.0, 1e-4);
}

TEST(ReplCommandsTest, poly_factor) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_factor(p)");
    expect_contains(interp, "help", "poly_rational_roots(p)");
    expect_contains(interp, "help", "poly_factor_rational(p)");
    expect_contains(interp, "help", "poly_partial_fractions(num,den)");
    expect_contains(interp, "help", "poly_root_count(p,a,b)");
    expect_contains(interp, "help", "poly_cheb_eval(cheb_coeffs,x)");

    // (x-2)(x-3) = x^2 - 5x + 6
    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("fac").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("fac")(0, 2), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("fac")(1, 2), 1.0, 1e-6);

    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rr").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rr").cols(), 2u);
    {
        std::vector<double> nums = {interp.state().matrices.at("rr")(0, 0),
                                    interp.state().matrices.at("rr")(1, 0)};
        std::vector<double> dens = {interp.state().matrices.at("rr")(0, 1),
                                    interp.state().matrices.at("rr")(1, 1)};
        std::sort(nums.begin(), nums.end());
        EXPECT_NEAR(nums[0], 2.0, 1e-6);
        EXPECT_NEAR(nums[1], 3.0, 1e-6);
        EXPECT_NEAR(dens[0], 1.0, 1e-6);
        EXPECT_NEAR(dens[1], 1.0, 1e-6);
    }

    expect_ok(interp, "fr = poly_factor_rational(p)");
    ASSERT_GT(interp.state().matrices.count("fr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("fr").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("fr")(2, 0), 1.0, 1e-6);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    ASSERT_GT(interp.state().matrices.count("pf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pf").cols(), 9u);
    EXPECT_NEAR(interp.state().matrices.at("pf")(0, 8), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("pf")(1, 8), 0.0, 1e-6);

    expect_ok(interp, "rc = poly_root_count(p, 1, 4)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), 2.0, 1e-6);

    expect_ok(interp, "cheb = [0; 1]");
    expect_ok(interp, "cv = poly_cheb_eval(cheb, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), 0.5, 1e-6);
    expect_contains(interp, "poly_cheb_eval([0; 1], 0.5)", "0.5");
}

TEST(ReplCommandsTest, poly_cheb_expand) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_cheb_expand(p,n)");
    expect_contains(interp, "help", "poly_cheb_expand(p,n,a,b)");

    // f(x) = x^3 - 2x + 1 on [-1,1]
    expect_ok(interp, "p = [1; -2; 0; 1]");
    expect_ok(interp, "cheb = poly_cheb_expand(p, 3)");
    ASSERT_GT(interp.state().matrices.count("cheb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cheb").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("cheb").cols(), 1u);

    expect_ok(interp, "cv = poly_cheb_eval(cheb, 0.5)");
    expect_ok(interp, "pv = poly_eval(p, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv"), interp.state().scalars.at("pv"), 1e-6);

    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    expect_ok(interp, "cv2 = poly_cheb_eval(cheb2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cv2"), interp.state().scalars.at("pv"), 1e-6);
}

TEST(ReplCommandsTest, poly_transforms) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_shift(p,a)");
    expect_contains(interp, "help", "poly_scale(p,a)");
    expect_contains(interp, "help", "poly_monic(p)");
    expect_contains(interp, "help", "poly_reverse(p)");
    expect_contains(interp, "help", "poly_pow(p,n)");
    expect_contains(interp, "help", "poly_lcm(a,b)");
    expect_contains(interp, "help", "poly_div_quot(a,b)");
    expect_contains(interp, "help", "poly_mod(a,b)");
    expect_contains(interp, "help", "poly_eval_at(coeffs,xs)");
    expect_contains(interp, "help", "poly_sylvester(p,q)");

    // x^2 shifted by 1 -> (x-1)^2
    expect_ok(interp, "s = poly_shift([0; 0; 1], 1)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("s")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("s")(1, 0), -2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("s")(2, 0), 1.0, 1e-6);

    // p(x)=x^2, p(2x)=4x^2
    expect_ok(interp, "sc = poly_scale([0; 0; 1], 2)");
    ASSERT_GT(interp.state().matrices.count("sc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("sc")(2, 0), 4.0, 1e-6);

    // monic normalize 2x^2 - 5x + 6
    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("m")(0, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("m")(1, 0), -2.5, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("m")(2, 0), 1.0, 1e-6);

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("r")(0, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("r")(1, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("r")(2, 0), 1.0, 1e-6);

    // (1+x)^2
    expect_ok(interp, "pw = poly_pow([1; 1], 2)");
    ASSERT_GT(interp.state().matrices.count("pw"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pw")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("pw")(1, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("pw")(2, 0), 1.0, 1e-6);

    // lcm(x-1, x+1) ~ x^2-1
    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("l")(0, 0), -1.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("l")(2, 0), 1.0, 1e-5);

    // (x^2+1) / (x+1) -> quotient x-1, remainder 2
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("q")(0, 0), -1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("q")(1, 0), 1.0, 1e-6);

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("rm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("vals"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("vals")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("vals")(1, 0), 6.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("vals")(2, 0), 17.0, 1e-6);

    // p(x)=x^2+2x+3, q(x)=x+5
    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 1), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 2), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 1), 5.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(2, 1), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(2, 2), 5.0, 1e-6);
}

TEST(ReplCommandsTest, graph_image_poly) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian(M)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "p = [1; 2; 3]");
    expect_ok(interp, "dp = poly_deriv(p)");
    EXPECT_EQ(interp.state().matrices.at("dp").rows(), 2u);
}

TEST(ReplCommandsTest, combo_poly) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 2u);

    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);

    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    EXPECT_GT(interp.state().matrices.at("sf").rows(), 0u);

    expect_ok(interp, "p = [1; 2; 3]");
    expect_ok(interp, "rev = poly_reverse(p)");
    EXPECT_NEAR(interp.state().matrices.at("rev")(0, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_poly) {
    Interpreter interp;

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_GT(interp.state().matrices.at("fe").rows(), 0u);

    expect_ok(interp, "sb = numthy_stern_brocot(3)");
    EXPECT_GT(interp.state().matrices.at("sb").rows(), 0u);

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    EXPECT_GT(interp.state().matrices.at("l").rows(), 0u);

    expect_ok(interp, "p = [1; 2]");
    expect_ok(interp, "q = [1; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_GT(interp.state().matrices.at("S").rows(), 0u);
}

TEST(ReplCommandsTest, numthy_poly_graph) {
    Interpreter interp;

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    ASSERT_GT(interp.state().matrices.count("lu"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lu").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("lu").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 1), 11.0, 1e-9);

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_EQ(interp.state().matrices.at("c").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("c")(1, 0), 2.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    ASSERT_GT(interp.state().matrices.count("pin"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rr").rows(), 2u);

    expect_ok(interp, "fr = poly_factor_rational(p)");
    ASSERT_GT(interp.state().matrices.count("fr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_poly_2) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pp").rows(), 3u);

    expect_ok(interp, "num = [1; 0]");
    expect_ok(interp, "den = [1; -1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    ASSERT_GT(interp.state().matrices.count("pf"), 0u);
    EXPECT_GT(interp.state().matrices.at("pf").rows(), 0u);

    expect_ok(interp, "p = [0; 0; 1]");
    expect_ok(interp, "cheb = poly_cheb_expand(p, 3)");
    ASSERT_GT(interp.state().matrices.count("cheb"), 0u);
    EXPECT_GT(interp.state().matrices.at("cheb").rows(), 0u);
}

TEST(ReplCommandsTest, poly_sph) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 3; 5]");
    expect_ok(interp, "p = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_GT(interp.state().matrices.at("p").rows(), 0u);

    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);
    EXPECT_GT(interp.state().matrices.at("pn").rows(), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rts").rows(), 2u);

    expect_ok(interp, "fac = poly_factor(p2)");
    ASSERT_GT(interp.state().matrices.count("fac"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplCommandsTest, restricted_squarefree) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_monic) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_div) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_evalat) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, sylvester_lucas) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_hermite) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, rational_factor) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, prevperm_partial) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, cheb_lagrange) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, newton_roots) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, restricted_squarefree_2) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_monic_2) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_div_2) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_evalat_2) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, sylvester_lucas_2) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_hermite_2) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, rational_factor_2) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, prevperm_partial_2) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, cheb_lagrange_2) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, newton_roots_2) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_2) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_2) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_2) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_2) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_2) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_2) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_2) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_2) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_2) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_2) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_3) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_3) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_3) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_3) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_3) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_3) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_3) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_3) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_3) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_3) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_4) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_4) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_4) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_4) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_4) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_4) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_4) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_4) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_4) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_4) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_5) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_5) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_5) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_5) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_5) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_5) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_5) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_5) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_5) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_5) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_6) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_6) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_6) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_6) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_6) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_6) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_6) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_6) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_6) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_6) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_7) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_7) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_7) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_7) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_7) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_7) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_7) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_7) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_7) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_7) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_8) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_8) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_8) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_8) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_8) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_8) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_8) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_8) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_8) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_8) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_9) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_9) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_9) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_9) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_9) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_9) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_9) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_9) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_9) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_9) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_10) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_10) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_10) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_10) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_10) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_10) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_10) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_10) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_10) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_10) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_11) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_11) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_11) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_11) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_11) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_11) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_11) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_11) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_11) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_11) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_12) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_12) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_12) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_12) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_12) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_12) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_12) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_12) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_12) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_12) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_13) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_13) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_13) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_13) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_13) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_13) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_13) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_13) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_13) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_13) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_14) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_14) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_14) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_14) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_14) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_14) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_14) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_14) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_14) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_14) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_15) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_15) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_15) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_15) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_15) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_15) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_15) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_15) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_15) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_15) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_16) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_16) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_16) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_16) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_16) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_16) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_16) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_16) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_16) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_16) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_17) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_17) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_17) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_17) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_17) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_17) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_17) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_17) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_17) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_17) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_18) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_18) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_18) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_18) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_18) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_18) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_18) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_18) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_18) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_18) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_19) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_19) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_19) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_19) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_19) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_19) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_19) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_19) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_19) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_19) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_20) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_20) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_20) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_20) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_20) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_20) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_20) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_20) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_20) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_20) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_21) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_21) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_21) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_21) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_21) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_21) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_21) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_21) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_21) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_21) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_22) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_22) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_22) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_22) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_22) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_22) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_22) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_22) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_22) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_22) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_23) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_23) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_23) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_23) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_23) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_23) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_23) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_23) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_23) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_23) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_24) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_24) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_24) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_24) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_24) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_24) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_24) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_24) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_24) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_24) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_25) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_25) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_25) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_25) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_25) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_25) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_25) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_25) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_25) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_25) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_26) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_26) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_26) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_26) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_26) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_26) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_26) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_26) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_26) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_26) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_27) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_27) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_27) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_27) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_27) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_27) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_27) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_27) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_27) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_27) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_28) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_28) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_28) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_28) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_28) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_28) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_28) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_28) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_28) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_28) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_29) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_29) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_29) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_29) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_29) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_29) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_29) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_29) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_29) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_29) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_30) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_30) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_30) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_30) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_30) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_30) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_30) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_30) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_30) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_30) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, dijkstra_poly_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
}

TEST(ReplCommandsTest, combo_restricted_partitions_poly_squarefree_31) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(ReplCommandsTest, poly_gcd_poly_monic_31) {
    Interpreter interp;

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(ReplCommandsTest, poly_lcm_poly_div_quot_31) {
    Interpreter interp;

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(ReplCommandsTest, poly_mod_poly_eval_at_31) {
    Interpreter interp;

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(ReplCommandsTest, poly_sylvester_numthy_lucas_sequence_31) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "S = poly_sylvester(p, q)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    EXPECT_NEAR(interp.state().matrices.at("lu")(0, 0), 5.0, 1e-9);
}

TEST(ReplCommandsTest, poly_fit_poly_interp_hermite_31) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(ReplCommandsTest, poly_rational_roots_poly_factor_rational_31) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(ReplCommandsTest, combo_prev_perm_poly_partial_fractions_31) {
    Interpreter interp;

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [2; -3; 1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_EQ(interp.state().matrices.at("pf").rows(), 2u);
}

TEST(ReplCommandsTest, poly_cheb_expand_poly_lagrange_31) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "cheb2 = poly_cheb_expand(p, 3, -1, 1)");
    ASSERT_GT(interp.state().matrices.count("cheb2"), 0u);

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
}

TEST(ReplCommandsTest, poly_interp_newton_poly_roots_31) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    ASSERT_GT(interp.state().matrices.count("rts"), 0u);
}

TEST(ReplCommandsTest, poly_scale_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "poly_scale([0; 0; 1], 2)", "scale =");
    expect_error_contains(interp, "poly_scale(missing, 2)", "unknown matrix");
}

TEST(ReplCommandsTest, poly_discriminant_noassign) {
    Interpreter interp;
    expect_contains(interp, "poly_discriminant([1; -2; 1])", "0");
    expect_error_contains(interp, "poly_discriminant(no_such_matrix)", "unknown matrix");
}
