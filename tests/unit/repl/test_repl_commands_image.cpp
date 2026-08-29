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

TEST(ReplCommandsTest, image_filters_and_resize) {
    Interpreter interp;
    expect_contains(interp, "help", "imgaussfilt(M,s)");
    expect_contains(interp, "help", "laplacian(M)");
    expect_contains(interp, "help", "imresize(M,r,c)");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    expect_ok(interp, "L = laplacian(G)");
    expect_ok(interp, "H = histeq(G)");
    expect_ok(interp, "S = sharpen(G)");
    expect_ok(interp, "T = threshold_otsu(G)");
    expect_ok(interp, "R = imresize(G, 2, 2)");

    ASSERT_TRUE(interp.state().matrices.count("B") > 0);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 2u);
}

TEST(ReplCommandsTest, bilateral) {
    Interpreter interp;
    expect_contains(interp, "help", "bilateral(M,sigma_s,sigma_r)");

    expect_ok(interp, "M = ones(5, 5)");
    expect_ok(interp, "B = bilateral(M, 1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 5u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("B")(2, 2)));
}

TEST(ReplCommandsTest, canny) {
    Interpreter interp;
    expect_contains(interp, "help", "canny(M,low,high)");

    expect_ok(interp,
              "M = [0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; "
              "0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; "
              "0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; 0,0,0,0,0,1,1,1,1,1; "
              "0,0,0,0,0,1,1,1,1,1]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 10u);
    EXPECT_EQ(interp.state().matrices.at("E").cols(), 10u);
}

TEST(ReplCommandsTest, boxfilter) {
    Interpreter interp;
    expect_contains(interp, "help", "boxfilter(M,k)");

    expect_ok(interp, "M = ones(5, 5)");
    expect_ok(interp, "B = boxfilter(M, 3)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "boxfilter(M, 3)");
}

TEST(ReplCommandsTest, prewitt) {
    Interpreter interp;
    expect_contains(interp, "help", "prewitt(M)");

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
}

TEST(ReplCommandsTest, image_transform) {
    Interpreter interp;
    expect_contains(interp, "help", "imflip(M,horizontal)");
    expect_contains(interp, "help", "imrotate90(M)");
    expect_contains(interp, "help", "threshold_binary(M,t)");
    expect_contains(interp, "help", "adapthisteq(M)");

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("F").cols(), 2u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "R = imrotate90(M)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 2u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("R")(1, 0), 0.0);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(1, 1), 1.0);

    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("A").cols(), 2u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("A")(0, 0)));
}

TEST(ReplCommandsTest, image_segment) {
    Interpreter interp;
    expect_contains(interp, "help", "label_components(B)");
    expect_contains(interp, "help", "watershed(G,M)");
    expect_contains(interp, "help", "slic(M,K");

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("L")(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("L")(0, 1), interp.state().matrices.at("L")(1, 2));
    EXPECT_NE(interp.state().matrices.at("L")(0, 1), interp.state().matrices.at("L")(3, 0));

    expect_ok(interp, "G = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; "
                      "0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G, Mk)");
    ASSERT_GT(interp.state().matrices.count("W"), 0u);
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("W").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("W")(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("W")(0, 0), 1.0);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 4u);
    EXPECT_GE(interp.state().matrices.at("S")(0, 0), 1.0);

    expect_ok(interp, "S2 = slic(RGB, 4, 10)");
    ASSERT_GT(interp.state().matrices.count("S2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S2").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("S2").cols(), 4u);
}

TEST(ReplCommandsTest, image_hough) {
    Interpreter interp;
    expect_contains(interp, "help", "hough_lines(M[,edge])");
    expect_contains(interp, "help", "hough_circles(M[,r_min,r_max])");
    expect_contains(interp, "help", "harris(M[,k[,thr]])");
    expect_contains(interp, "help", "shi_tomasi(M,n[,q])");

    // 10x10 horizontal edge at row 5; vote_threshold=5 so short lines still peak.
    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("L").rows(), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 1), 1.5707963267948966, 0.05);  // ~pi/2
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 5.0, 1.5);

    // Blank image: no circles.
    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 0u);

    // Bright quadrant corner -> Harris / Shi-Tomasi keypoints.
    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("H").rows(), 0u);
    EXPECT_GT(interp.state().matrices.at("H")(0, 2), 0.0);

    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("S").rows(), 0u);
    EXPECT_LE(interp.state().matrices.at("S").rows(), 5u);
}

TEST(ReplCommandsTest, gray2rgb_and_impad) {
    Interpreter interp;
    expect_contains(interp, "help", "gray2rgb(M)");
    expect_contains(interp, "help", "impad(M,pad");

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    expect_ok(interp, "RGB2 = gray2rgb(G)");
    ASSERT_GT(interp.state().matrices.count("RGB2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("RGB2").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("RGB2").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("RGB2")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("RGB2")(0, 1), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("RGB2")(0, 2), 0.299, 1e-6);

    // Use 0..1 intensities so matrixâ†”Image round-trip preserves values.
    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("P").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("P")(0, 0), 0.0);
    EXPECT_NEAR(interp.state().matrices.at("P")(1, 1), 0.1, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("P")(2, 2), 0.4, 1e-6);
}

TEST(ReplCommandsTest, radon_iradon) {
    Interpreter interp;
    expect_contains(interp, "help", "radon(M,theta)");
    expect_contains(interp, "help", "iradon(S,theta)");

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 8u);
    for (size_t r = 0; r < interp.state().matrices.at("S").rows(); ++r) {
        for (size_t c = 0; c < interp.state().matrices.at("S").cols(); ++c) {
            EXPECT_TRUE(std::isfinite(interp.state().matrices.at("S")(r, c)));
        }
    }

    expect_ok(interp, "R = iradon(S, Th)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 8u);
    for (size_t r = 0; r < interp.state().matrices.at("R").rows(); ++r) {
        for (size_t c = 0; c < interp.state().matrices.at("R").cols(); ++c) {
            EXPECT_TRUE(std::isfinite(interp.state().matrices.at("R")(r, c)));
        }
    }
}

TEST(ReplCommandsTest, image_imfilter_sobel) {
    Interpreter interp;
    expect_contains(interp, "help", "imfilter(M,K)");
    expect_contains(interp, "help", "sobel_x(M)");
    expect_contains(interp, "help", "laplacian_of_gaussian(M,sigma)");

    expect_ok(interp, "G = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "K = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("F")(1, 1), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("F")(0, 0), 0.0, 1e-6);

    expect_ok(interp, "Sx = sobel_x(G)");
    expect_ok(interp, "Sy = sobel_y(G)");
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Sx").cols(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Sy").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Sy").cols(), 3u);

    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "D = dft_magnitude(G)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 3u);
}

TEST(ReplCommandsTest, image) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);

    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);

    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);

    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "D = dft_magnitude(G)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, imopen) {
    Interpreter interp;
    expect_contains(interp, "help", "imopen(M,k)");

    expect_ok(interp, "M = [0,0,0,0,0; 0,1,1,1,0; 0,1,1,1,0; 0,1,1,1,0; 0,0,0,0,0]");
    expect_ok(interp, "O = imopen(M, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 5u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("O")(2, 2)));

    expect_ok(interp, "O1 = imopen(M)");
    ASSERT_GT(interp.state().matrices.count("O1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O1").rows(), 5u);

    expect_error_contains(interp, "bad = imopen(missing, 3)", "unknown matrix");
    expect_error_contains(interp, "bad = imopen(M, 2)", "odd integer ksize");
}

TEST(ReplCommandsTest, imclose) {
    Interpreter interp;
    expect_contains(interp, "help", "imclose(M,k)");

    expect_ok(interp, "M = [0,0,0,0,0; 0,1,1,1,0; 0,1,0,1,0; 0,1,1,1,0; 0,0,0,0,0]");
    expect_ok(interp, "C = imclose(M, 3)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 5u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("C")(2, 2)));

    expect_ok(interp, "C1 = imclose(M)");
    ASSERT_GT(interp.state().matrices.count("C1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("C1").rows(), 5u);

    expect_error_contains(interp, "bad = imclose(missing, 3)", "unknown matrix");
    expect_error_contains(interp, "bad = imclose(M, 4)", "odd integer ksize");
}

TEST(ReplCommandsTest, poly_prewitt_scharr) {
    Interpreter interp;

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, adapthisteq_imflip) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);
}

TEST(ReplCommandsTest, otsu_rotate) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
}

TEST(ReplCommandsTest, binary_label) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);
}

TEST(ReplCommandsTest, medfilt_boxfilter) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
}

TEST(ReplCommandsTest, canny_resize) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
}

TEST(ReplCommandsTest, harris_shitomasi) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, impad_radon) {
    Interpreter interp;

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, iradon_femmesh) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, sobel) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, medfilt_boxfilter_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
}

TEST(ReplCommandsTest, canny_resize_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
}

TEST(ReplCommandsTest, harris_shitomasi_2) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, impad_radon_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, iradon_femmesh_2) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, sobel_2) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_3) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_3) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_2) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_2) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_3) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_2) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_3) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_4) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_4) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_3) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_3) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_3) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_2) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_4) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_3) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_2) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_4) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_5) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_5) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_4) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_4) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_4) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_3) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_3) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_3) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_5) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_4) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_3) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_5) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_6) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_6) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_5) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_5) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_5) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_4) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_4) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_4) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_6) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_5) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_4) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_6) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_7) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_7) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_6) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_6) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_6) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_5) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_5) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_5) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_7) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_6) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_5) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_7) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_8) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_8) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_7) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_7) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_7) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_6) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_6) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_6) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_8) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_7) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_6) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_8) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_9) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_9) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_8) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_8) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_8) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_7) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_7) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_7) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_9) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_8) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_7) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_9) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_10) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_10) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_9) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_9) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_9) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_8) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_8) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_8) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_10) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_9) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_8) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_10) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_11) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_11) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_10) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_10) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_10) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_9) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_9) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_9) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_11) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_10) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_9) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_11) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_12) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_12) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_11) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_11) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_11) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_10) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_10) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_10) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_12) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_11) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_10) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_12) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_13) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_13) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_12) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_12) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_12) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_11) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_11) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_11) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_13) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_12) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_11) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_13) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_14) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_14) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_13) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_13) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_13) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_12) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_12) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_12) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_14) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_13) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_12) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_14) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_15) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_15) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_14) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_14) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_14) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_13) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_13) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_13) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_15) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_14) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_13) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_15) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_16) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_16) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_15) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_15) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_15) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_14) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_14) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_14) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_16) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_15) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_14) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_16) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_17) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_17) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_16) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_16) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_16) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_15) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_15) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_15) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_17) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_16) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_15) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_17) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_18) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_18) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_17) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_17) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_17) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_16) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_16) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_16) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_18) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_17) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_16) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_18) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_19) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_19) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_18) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_18) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_18) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_17) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_17) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_17) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_19) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_18) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_17) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_19) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_20) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_20) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_19) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_19) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_19) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_18) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_18) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_18) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_20) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_19) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_18) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_20) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_21) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_21) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_20) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_20) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_20) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_19) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_19) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_19) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_21) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_20) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_19) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_21) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_22) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_22) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_21) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_21) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_21) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_20) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_20) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_20) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_22) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_21) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_20) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_22) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_23) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_23) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_22) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_22) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_22) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_21) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_21) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_21) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_23) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_22) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_21) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_23) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_24) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_24) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_23) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_23) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_23) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_22) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_22) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_22) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_24) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_23) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_22) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_24) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_25) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_25) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_24) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_24) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_24) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_23) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_23) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_23) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_25) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_24) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_23) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_25) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_26) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_26) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_25) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_25) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_25) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_24) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_24) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_24) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_26) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_25) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_24) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_26) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_27) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_27) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_26) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_26) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_26) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_25) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_25) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_25) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_27) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_26) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_25) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_27) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_28) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_28) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_27) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_27) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_27) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_26) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_26) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_26) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_28) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_27) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_26) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_28) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_29) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_29) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_28) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_28) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_28) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_27) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_27) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_27) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_29) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_28) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_27) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_29) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_30) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_30) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_29) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_29) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_29) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_28) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_28) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_28) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_30) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_29) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_28) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_30) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_31) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_31) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_30) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_30) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_30) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_29) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_29) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_29) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_31) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_30) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_29) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_31) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_32) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_32) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_31) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_31) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_31) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_30) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_30) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_30) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_32) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_31) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, sobel_x_sobel_y_30) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    ASSERT_GT(interp.state().matrices.count("Sx"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
    expect_ok(interp, "Sy = sobel_y(G)");
    ASSERT_GT(interp.state().matrices.count("Sy"), 0u);
}

TEST(ReplCommandsTest, prewitt_scharr_32) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(ReplCommandsTest, roberts_laplacian_33) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(ReplCommandsTest, histeq_sharpen_33) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, prim_adapthisteq_32) {
    Interpreter interp;

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
}

TEST(ReplCommandsTest, hull_otsu_32) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(ReplCommandsTest, rotate_binary_32) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);
}

TEST(ReplCommandsTest, boxfilter_dilate_erode_31) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, bilateral_canny_31) {
    Interpreter interp;

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
}

TEST(ReplCommandsTest, imresize_watershed_31) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);

    expect_ok(interp, "G2 = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G2, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);
}

TEST(ReplCommandsTest, harris_shitomasi_33) {
    Interpreter interp;

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(ReplCommandsTest, radon_iradon_32) {
    Interpreter interp;

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(ReplCommandsTest, count_components_noassign) {
    Interpreter interp;
    expect_ok(interp, "B = [1, 1, 0; 1, 1, 0; 0, 0, 1]");
    expect_contains(interp, "count_components(B)", "2");
    expect_error_contains(interp, "count_components(no_such_matrix)", "unknown matrix");
}
