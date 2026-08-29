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

TEST(ReplCommandsTest, fft_rfft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_rfft(x)");

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-6);
}

TEST(ReplCommandsTest, fft_irfft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_irfft(spectrum,n)");

    expect_ok(interp, "S = fft_rfft([1; 2; 3; 4])");
    expect_ok(interp, "x = fft_irfft(S, 4)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(3, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, fft_dct2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dct2(x)");

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d").rows(), 4u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("d")(0, 0)));
}

TEST(ReplCommandsTest, fft_idct2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_idct2(x)");

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "x = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("x")(3, 0), 4.0, 1e-5);
}

TEST(ReplCommandsTest, fft_dst2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dst2(x)");

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s").rows(), 4u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("s")(0, 0)));
}

TEST(ReplCommandsTest, fft_dft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dft(x)");

    expect_ok(interp, "S = fft_dft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 1.0, 1e-9);

    expect_ok(interp, "fft_dft([1; 0; 0; 0])");
}

TEST(ReplCommandsTest, fft_goertzel) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_goertzel(x,f,fs)");

    // DC bin of a constant length-4 signal: sum of samples = 4.
    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("G").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 4.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 1), 0.0, 1e-9);

    expect_ok(interp, "fft_goertzel([1; 1; 1; 1], 0, 4)");
}

TEST(ReplCommandsTest, fft_ifft) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_ifft(spectrum)");

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);

    expect_ok(interp, "fft_ifft(S)");
}

TEST(ReplCommandsTest, fft_fft2) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_fft2(S)");

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "F = fft_fft2(S)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("F").cols(), 2u);

    expect_ok(interp, "fft_fft2(S)");
}

TEST(ReplCommandsTest, fftshift) {
    Interpreter interp;
    expect_contains(interp, "help", "fftshift(S)");

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, fftfreq) {
    Interpreter interp;
    expect_contains(interp, "help", "fftfreq(n[,d])");

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    ASSERT_EQ(freqs.rows(), 8u);
    ASSERT_EQ(freqs.cols(), 1u);
    const std::vector<double> expected{0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    const auto& scaled = interp.state().matrices.at("f2");
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(scaled(i, 0), expected[i] / 0.5, 1e-12);
    }
}

TEST(ReplCommandsTest, rfftfreq) {
    Interpreter interp;
    expect_contains(interp, "help", "rfftfreq(n[,d])");

    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
    const auto& freqs = interp.state().matrices.at("rf");
    ASSERT_EQ(freqs.rows(), 5u);
    ASSERT_EQ(freqs.cols(), 1u);
    const std::vector<double> expected{0.0, 0.125, 0.25, 0.375, 0.5};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs(i, 0), expected[i], 1e-12);
    }

    expect_ok(interp, "rf2 = rfftfreq(5)");
    const auto& odd = interp.state().matrices.at("rf2");
    ASSERT_EQ(odd.rows(), 3u);
    EXPECT_NEAR(odd(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(odd(1, 0), 0.2, 1e-12);
    EXPECT_NEAR(odd(2, 0), 0.4, 1e-12);
}

TEST(ReplCommandsTest, ifftshift) {
    Interpreter interp;
    expect_contains(interp, "help", "ifftshift(S)");

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
    const auto& restored = interp.state().matrices.at("Sr");
    EXPECT_NEAR(restored(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(restored(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(restored(2, 0), 2.0, 1e-9);
    EXPECT_NEAR(restored(3, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, fft_control) {
    Interpreter interp;

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, rfft_dft_ifft) {
    Interpreter interp;

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_idst2) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);
}

TEST(ReplCommandsTest, dct_idct_dst) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    EXPECT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv_dft) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, dft_ifft) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_2) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_2) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_2) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv_dft_2) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, dft_ifft_2) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_2) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_2) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_3) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_3) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_3) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_3) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_3) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_3) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_3) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_4) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_4) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_4) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_2) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_4) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_4) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_4) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_4) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_5) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_5) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_5) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_3) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_5) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_5) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_5) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_5) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_6) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_6) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_6) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_4) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_6) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_6) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_6) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_6) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_7) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_7) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_7) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_5) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_7) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_7) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_7) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_7) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_8) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_8) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_8) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_6) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_8) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_8) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_8) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_8) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_9) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_9) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_9) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_7) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_9) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_9) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_9) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_9) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_10) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_10) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_10) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_8) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_10) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_10) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_10) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_10) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_11) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_11) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_11) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_9) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_11) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_11) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_11) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_11) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_12) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_12) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_12) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_10) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_12) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_12) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_12) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_12) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_13) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_13) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_13) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_11) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_13) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_13) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_13) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_13) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_14) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_14) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_14) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_12) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_14) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_14) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_14) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_14) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_15) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_15) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_15) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_13) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_15) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_15) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_15) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_15) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_16) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_16) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_16) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_14) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_16) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_16) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_16) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_16) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_17) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_17) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_17) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_15) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_17) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_17) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_17) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_17) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_18) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_18) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_18) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_16) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_18) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_18) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_18) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_18) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_19) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_19) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_19) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_17) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_19) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_19) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_19) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_19) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_20) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_20) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_20) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_18) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_20) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_20) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_20) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_20) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_21) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_21) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_21) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_19) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_21) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_21) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_21) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_21) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_22) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_22) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_22) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_20) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_22) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_22) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_22) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_22) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_23) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_23) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_23) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_21) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_23) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_23) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_23) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_23) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_24) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_24) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_24) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_22) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_24) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_24) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_24) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_24) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_25) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_25) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_25) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_23) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_25) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_25) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_25) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_25) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_26) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_26) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_26) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_24) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_26) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_26) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_26) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_26) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_27) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_27) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_27) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_25) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_27) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_27) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_27) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_27) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_28) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_28) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_28) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_26) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_28) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_28) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_28) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_28) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_29) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_29) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_29) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_27) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_29) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_29) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_29) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_29) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_30) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_30) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_30) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_28) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_30) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_30) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_30) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_30) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_31) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_31) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_31) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_29) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_31) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_31) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_31) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_31) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_32) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_32) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_32) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(ReplCommandsTest, hsv2rgb_dft_magnitude_30) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 0, 1]");
    expect_ok(interp, "R = hsv2rgb(H)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 3u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
}

TEST(ReplCommandsTest, dft_ifft_32) {
    Interpreter interp;

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(ReplCommandsTest, fft2_ifft2_32) {
    Interpreter interp;

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    ASSERT_GT(interp.state().matrices.count("back"), 0u);
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(ReplCommandsTest, idst2_dct2_32) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    ASSERT_GT(interp.state().matrices.count("ib"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    ASSERT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(ReplCommandsTest, idct2_dst2_32) {
    Interpreter interp;

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    ASSERT_GT(interp.state().matrices.count("xr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    ASSERT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(ReplCommandsTest, fftshift_ifftshift_33) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    ASSERT_GT(interp.state().matrices.count("Sh"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Sh")(0, 0), 2.0, 1e-9);
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
}

TEST(ReplCommandsTest, fftfreq_rfftfreq_33) {
    Interpreter interp;

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(ReplCommandsTest, goertzel_bode_33) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}
