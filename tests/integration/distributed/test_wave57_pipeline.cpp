// MathScript Integration Test: Waves 57–60 cross-module pipeline

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/combo/combo.hpp"
#include "ms/compress/compress.hpp"
#include "ms/finance/finance.hpp"
#include "ms/geo/geo.hpp"
#include "ms/image/image.hpp"
#include "ms/info/info.hpp"
#include "ms/ml/ml.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/stats/stats.hpp"
#include "ms/tensorops/tensorops.hpp"
#include "ms/topo/topo.hpp"

using namespace ms;

namespace {

bool is_finite(double v) { return std::isfinite(v); }

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

std::vector<std::vector<double>> point_cloud_from_geo(const std::vector<geo::Point2D>& pts) {
    std::vector<std::vector<double>> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        out.push_back({p.x, p.y});
    }
    return out;
}

ml::Mat dist_matrix_to_mat(const std::vector<std::vector<double>>& dist_mat) {
    ml::Mat out;
    out.reserve(dist_mat.size());
    for (const auto& row : dist_mat) {
        out.push_back(row);
    }
    return out;
}

bignum::BigInt bigint_binomial(int n, int k) {
    if (k < 0 || k > n) return bignum::BigInt(0LL);
    if (k == 0 || k == n) return bignum::BigInt(1LL);
    if (k > n - k) k = n - k;
    const bignum::BigInt num = bignum::bigint_factorial(n);
    const bignum::BigInt den =
        bignum::bigint_factorial(k) * bignum::bigint_factorial(n - k);
    return num / den;
}

compress::Bytes probs_to_bytes(const std::vector<double>& probs) {
    compress::Bytes out;
    for (double p : probs) {
        const int q = static_cast<int>(std::lround(p * 10000.0));
        out.push_back(static_cast<uint8_t>((q >> 8) & 0xFF));
        out.push_back(static_cast<uint8_t>(q & 0xFF));
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Numthy + Combo + Finance: partition/binomial counts → bond NPV
// ---------------------------------------------------------------------------
TEST(Wave57_60Pipeline, Numthy_Combo_Finance_PartitionBondNPV) {
    // partition(6) = 11 coupon periods; binomial(11,4) scales cashflow magnitude
    const int periods = static_cast<int>(numthy::partition(6));
    ASSERT_EQ(periods, 11);

    const uint32_t mid = static_cast<uint32_t>(periods / 2);
    const auto path_weight = combo::binomial(static_cast<uint32_t>(periods), mid);
    EXPECT_EQ(path_weight, 462u);

    const double coupon_rate = static_cast<double>(path_weight) / 10000.0;
    const double yield = 0.05;
    const double face = 100.0;
    const double price = finance::bond_price(coupon_rate, yield, periods, face);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(is_finite(price));

    const double coupon_pay = coupon_rate * face;
    std::vector<double> cashflows(static_cast<size_t>(periods + 1), coupon_pay);
    cashflows.back() += face;
    cashflows.front() = -price;
    const double npv_val = finance::npv(yield, std::span<const double>(cashflows));
    EXPECT_NEAR(npv_val, 0.0, 0.05);
}

// ---------------------------------------------------------------------------
// Info + Compress: Shannon entropy → Huffman/RLE on encoded distribution bytes
// ---------------------------------------------------------------------------
TEST(Wave57_60Pipeline, Info_Compress_EntropyRoundtrip) {
    const std::vector<double> probs = {0.5, 0.25, 0.125, 0.125};
    const double H_bits = info::entropy(probs, 2.0);
    EXPECT_NEAR(H_bits, 1.75, 1e-10);
    EXPECT_GT(info::efficiency(probs), 0.85);

    const auto raw = probs_to_bytes(probs);
    ASSERT_EQ(raw.size(), 8u);

    const auto rle_back = compress::rle_decode(compress::rle_encode(raw));
    EXPECT_EQ(rle_back, raw);

    const auto hr = compress::huffman_encode(raw);
    const auto huff_back = compress::huffman_decode(hr, raw.size());
    EXPECT_EQ(huff_back, raw);

    // Entropy bounds average code length for this discrete source
    EXPECT_GE(H_bits, 0.0);
    EXPECT_LE(H_bits, 2.0);
}

// ---------------------------------------------------------------------------
// Geo + Topo + ML: point cloud → distances → Vietoris-Rips → PCA on rows
// ---------------------------------------------------------------------------
TEST(Wave57_60Pipeline, Geo_Topo_ML_DistancePCA) {
    const std::vector<geo::Point2D> geo_pts = {
        {0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}, {0.5, 0.3}, {1.5, 0.5},
    };
    const auto pts = point_cloud_from_geo(geo_pts);
    const auto dist_mat = topo::pairwise_distances(pts);
    ASSERT_EQ(dist_mat.size(), geo_pts.size());
    EXPECT_NEAR(dist_mat[0][1], geo::dist(geo_pts[0], geo_pts[1]), 1e-10);

    const auto complex = topo::vietoris_rips(dist_mat, 1.1, 2);
    const int chi = complex.euler_characteristic();
    EXPECT_TRUE(is_finite(static_cast<double>(chi)));

    const auto features = dist_matrix_to_mat(dist_mat);
    ASSERT_GE(features.size(), 3u);
    ml::PCA pca(2);
    const auto Z = pca.fit_transform(features);
    ASSERT_EQ(Z.size(), features.size());
    ASSERT_EQ(Z[0].size(), 2u);
    EXPECT_TRUE(is_finite(Z.front()[0]));
    EXPECT_GT(std::abs(Z.back()[1]), 0.0);
}

// ---------------------------------------------------------------------------
// Image + Compress + ML: RGB checkerboard → gray → patches → KMeans
// ---------------------------------------------------------------------------
TEST(Wave57_60Pipeline, Image_Compress_ML_CheckerboardKMeans) {
    const auto rgb = make_checkerboard_rgb(20, 20, 2);
    const auto gray = image::rgb2gray(rgb);
    EXPECT_EQ(gray.channels, 1);

    const auto patches = extract_patches(gray, 3, 3);
    ASSERT_GE(patches.size(), 4u);

    ml::KMeans km(2, 50, 1e-3);
    km.fit(patches);
    const auto labels = km.predict(patches);
    ASSERT_EQ(labels.size(), patches.size());
    EXPECT_TRUE(labels[0] == 0.0 || labels[0] == 1.0);
    EXPECT_LT(km.inertia(patches), 1e6);

    compress::Bytes raw;
    for (const auto& row : patches) {
        for (double v : row) {
            const int q = static_cast<int>(std::lround(v * 1000.0));
            raw.push_back(static_cast<uint8_t>((q >> 8) & 0xFF));
            raw.push_back(static_cast<uint8_t>(q & 0xFF));
        }
    }
    ASSERT_FALSE(raw.empty());
    const auto rle_back = compress::rle_decode(compress::rle_encode(raw));
    EXPECT_EQ(rle_back, raw);
}

// ---------------------------------------------------------------------------
// Bignum + Combo: exact binomial matches combo::binomial for moderate n
// ---------------------------------------------------------------------------
TEST(Wave57_60Pipeline, Bignum_Combo_BinomialAgreement) {
    const int n = 18;
    const int k = 6;
    const auto exact = bigint_binomial(n, k);
    EXPECT_EQ(exact.to_string(), "18564");
    EXPECT_EQ(exact.to_ll(), static_cast<long long>(combo::binomial(
        static_cast<uint32_t>(n), static_cast<uint32_t>(k))));

    const int n2 = 14;
    const int k2 = 7;
    const auto exact2 = bigint_binomial(n2, k2);
    EXPECT_EQ(exact2.to_ll(), static_cast<long long>(combo::binomial(
        static_cast<uint32_t>(n2), static_cast<uint32_t>(k2))));
    EXPECT_EQ(exact2.to_string(), "3432");
}

// ---------------------------------------------------------------------------
// Tensorops + Stats: einsum contraction → mean/std on flattened result
// ---------------------------------------------------------------------------
TEST(Wave57_60Pipeline, Tensorops_Stats_EinsumMoments) {
    tensorops::Tensor A({3, 4}, 0.0);
    tensorops::Tensor B({4, 2}, 0.0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            A.at({i, j}) = 0.2 * i + 0.1 * j + 1.0;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            B.at({i, j}) = (i + 1) * 0.25 + 0.5 * j;
        }
    }

    const auto C = tensorops::einsum("ij,jk->ik", A, B);
    ASSERT_EQ(C.shape[0], 3);
    ASSERT_EQ(C.shape[1], 2);
    ASSERT_EQ(C.numel(), 6);

    std::vector<double> flat;
    flat.reserve(static_cast<size_t>(C.numel()));
    for (long idx = 0; idx < C.numel(); ++idx) {
        flat.push_back(C.data[static_cast<size_t>(idx)]);
    }

    const double mu = mean(std::span<const double>(flat));
    const double sd = stddev(std::span<const double>(flat));
    EXPECT_TRUE(is_finite(mu));
    EXPECT_GT(sd, 0.0);
    EXPECT_GT(mu, 1.0);
    EXPECT_LT(sd, mu);
}
