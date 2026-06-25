// MathScript Integration Test: Image → ML → Compress → Bignum pipeline (Wave 60)

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/combo/combo.hpp"
#include "ms/compress/compress.hpp"
#include "ms/image/image.hpp"
#include "ms/ml/ml.hpp"

using namespace ms;

namespace {

image::Image make_checkerboard_rgb(int rows, int cols, int block = 4) {
    image::Image img(rows, cols, 3);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const bool light = ((r / block) + (c / block)) % 2 == 0;
            const float v = light ? 0.9f : 0.1f;
            img.at(r, c, 0) = v;
            img.at(r, c, 1) = light ? 0.2f : 0.8f;
            img.at(r, c, 2) = light ? 0.2f : 0.2f;
        }
    }
    return img;
}

ml::Mat extract_patches(const image::Image& gray, int patch, int stride) {
    ml::Mat patches;
    for (int r = 0; r + patch <= gray.rows; r += stride) {
        for (int c = 0; c + patch <= gray.cols; c += stride) {
            ml::Vec feat;
            feat.reserve(static_cast<size_t>(patch * patch));
            for (int dr = 0; dr < patch; ++dr) {
                for (int dc = 0; dc < patch; ++dc) {
                    feat.push_back(static_cast<double>(gray.at(r + dr, c + dc, 0)));
                }
            }
            patches.push_back(std::move(feat));
        }
    }
    return patches;
}

compress::Bytes features_to_bytes(const ml::Mat& Z) {
    compress::Bytes out;
    for (const auto& row : Z) {
        for (double v : row) {
            const int q = static_cast<int>(std::lround(v * 1000.0));
            out.push_back(static_cast<uint8_t>((q >> 8) & 0xFF));
            out.push_back(static_cast<uint8_t>(q & 0xFF));
        }
    }
    return out;
}

bignum::BigInt bigint_binomial(int n, int k) {
    if (k < 0 || k > n) return bignum::BigInt(0LL);
    if (k == 0 || k == n) return bignum::BigInt(1LL);
    if (k > n - k) k = n - k;
    bignum::BigInt num = bignum::bigint_factorial(n);
    bignum::BigInt den = bignum::bigint_factorial(k) * bignum::bigint_factorial(n - k);
    return num / den;
}

} // namespace

// ---------------------------------------------------------------------------
// Image processing: synthetic RGB → grayscale → edge maps
// ---------------------------------------------------------------------------
TEST(Wave60Pipeline, Image_RGB2Gray_Sobel_Canny) {
    const auto rgb = make_checkerboard_rgb(32, 32, 4);
    const auto gray = image::rgb2gray(rgb);
    EXPECT_EQ(gray.channels, 1);
    EXPECT_EQ(gray.rows, 32);
    EXPECT_EQ(gray.cols, 32);

    const auto grad = image::sobel(gray);
    EXPECT_EQ(grad.channels, 1);
    EXPECT_FALSE(grad.empty());

    float max_grad = 0.f;
    for (float v : grad.data) max_grad = std::max(max_grad, v);
    EXPECT_GT(max_grad, 0.f);

    const auto edges = image::canny(gray, 0.1f, 0.3f);
    EXPECT_EQ(edges.rows, gray.rows);
    EXPECT_EQ(edges.cols, gray.cols);
}

// ---------------------------------------------------------------------------
// ML: PCA on flattened image patches reduces dimension
// ---------------------------------------------------------------------------
TEST(Wave60Pipeline, ML_PCA_OnImagePatches) {
    const auto gray = image::rgb2gray(make_checkerboard_rgb(24, 24, 3));
    const auto patches = extract_patches(gray, 4, 4);
    ASSERT_GE(patches.size(), 4u);

    ml::PCA pca(2);
    const auto Z = pca.fit_transform(patches);
    ASSERT_EQ(Z.size(), patches.size());
    ASSERT_EQ(Z[0].size(), 2u);
    EXPECT_GT(std::abs(Z.back()[0]), 0.0);
}

// ---------------------------------------------------------------------------
// ML: KMeans clusters patch features from edge-enhanced image
// ---------------------------------------------------------------------------
TEST(Wave60Pipeline, ML_KMeans_OnSobelPatches) {
    const auto gray = image::rgb2gray(make_checkerboard_rgb(20, 20, 2));
    const auto grad = image::sobel(gray);
    const auto patches = extract_patches(grad, 3, 3);
    ASSERT_GE(patches.size(), 3u);

    ml::KMeans km(2, 50, 1e-3);
    km.fit(patches);
    const auto labels = km.predict(patches);
    ASSERT_EQ(labels.size(), patches.size());
    EXPECT_TRUE(labels[0] == 0.0 || labels[0] == 1.0);
    EXPECT_LT(km.inertia(patches), 1e6);
}

// ---------------------------------------------------------------------------
// Compress: RLE and Huffman roundtrip on PCA feature bytes
// ---------------------------------------------------------------------------
TEST(Wave60Pipeline, Compress_RLE_Huffman_FeatureRoundtrip) {
    const auto gray = image::rgb2gray(make_checkerboard_rgb(16, 16, 2));
    const auto patches = extract_patches(gray, 4, 4);
    ml::PCA pca(2);
    const auto Z = pca.fit_transform(patches);
    const auto raw = features_to_bytes(Z);
    ASSERT_FALSE(raw.empty());

    const auto rle_back = compress::rle_decode(compress::rle_encode(raw));
    EXPECT_EQ(rle_back, raw);

    const auto hr = compress::huffman_encode(raw);
    const auto huff_back = compress::huffman_decode(hr, raw.size());
    EXPECT_EQ(huff_back, raw);
}

// ---------------------------------------------------------------------------
// Bignum: exact binomial count matches combo::binomial for patch selection
// ---------------------------------------------------------------------------
TEST(Wave60Pipeline, Bignum_BinomialMatchesCombo) {
    const int n = 12;  // patches available
    const int k = 4;   // patches to select
    const auto exact = bigint_binomial(n, k);
    EXPECT_EQ(exact.to_string(), "495");
    EXPECT_EQ(exact.to_ll(), static_cast<long long>(combo::binomial(
        static_cast<uint32_t>(n), static_cast<uint32_t>(k))));

    const auto fact = bignum::bigint_factorial(n);
    EXPECT_EQ(fact.to_string(), "479001600");
}
