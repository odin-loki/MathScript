#include "ms/compress/compress.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <map>

using namespace ms::compress;

namespace {

double shannon_entropy_bits(const Bytes& data) {
    if (data.empty()) return 0.0;
    std::map<uint8_t, int> freq;
    for (uint8_t b : data) ++freq[b];
    double n = static_cast<double>(data.size());
    double h = 0.0;
    for (auto& [_, f] : freq) {
        double p = static_cast<double>(f) / n;
        h -= p * std::log2(p);
    }
    return h;
}

size_t arithmetic_payload_bits(const ArithmeticResult& ar) {
    return ar.encoded.size() * 8u;
}

Bytes make_skewed_text() {
    Bytes data;
    std::string s =
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    for (char c : s) data.push_back(static_cast<uint8_t>(c));
    return data;
}

} // namespace

// ---- RLE ----

TEST(CompressRLE, EmptyRoundtrip) {
    Bytes data;
    EXPECT_EQ(rle_decode(rle_encode(data)), data);
}

TEST(CompressRLE, SimpleRun) {
    Bytes data={65,65,65,65,65};
    auto enc=rle_encode(data);
    EXPECT_EQ(enc.size(), 2u);
    EXPECT_EQ(enc[0], 5);
    EXPECT_EQ(enc[1], 65);
    EXPECT_EQ(rle_decode(enc), data);
}

TEST(CompressRLE, MixedRuns) {
    Bytes data={'A','A','B','B','B','C'};
    auto enc=rle_encode(data);
    auto dec=rle_decode(enc);
    EXPECT_EQ(dec, data);
}

TEST(CompressRLE, LongRun) {
    Bytes data(300, 42);
    auto enc=rle_encode(data);
    auto dec=rle_decode(enc);
    EXPECT_EQ(dec, data);
}

TEST(CompressRLE, AlternatingBytes) {
    Bytes data={'A','B','A','B','A','B'};
    auto enc=rle_encode(data);
    auto dec=rle_decode(enc);
    EXPECT_EQ(dec, data);
}

// ---- Huffman ----

TEST(CompressHuffman, SimpleRoundtrip) {
    Bytes data={'a','b','c','a','a','b'};
    auto hr=huffman_encode(data);
    auto dec=huffman_decode(hr, data.size());
    EXPECT_EQ(dec, data);
}

TEST(CompressHuffman, SingleSymbol) {
    Bytes data={'X','X','X'};
    auto hr=huffman_encode(data);
    auto dec=huffman_decode(hr, data.size());
    EXPECT_EQ(dec, data);
}

TEST(CompressHuffman, AllBytes) {
    Bytes data;
    for (int i=0;i<256;++i) data.push_back((uint8_t)i);
    auto hr=huffman_encode(data);
    auto dec=huffman_decode(hr, data.size());
    EXPECT_EQ(dec, data);
}

TEST(CompressHuffman, CodebookAndPadding) {
    Bytes data={'a','a','b','b','b','c'};
    auto hr=huffman_encode(data);
    EXPECT_FALSE(hr.codebook.empty());
    EXPECT_GE(hr.padding_bits, 0);
    EXPECT_LT(hr.padding_bits, 8);
    EXPECT_EQ(huffman_decode(hr, data.size()), data);
}

// ---- Arithmetic (range) coding ----

TEST(CompressArithmetic, EmptyRoundtrip) {
    Bytes data;
    auto ar = arithmetic_encode(data);
    EXPECT_TRUE(ar.encoded.empty());
    EXPECT_TRUE(ar.freq_table.empty());
    EXPECT_EQ(ar.original_size, 0u);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, SkewedEnglishTextRoundtrip) {
    Bytes data = make_skewed_text();
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, SingleRepeatedByteRoundtrip) {
    Bytes data(4096, 0x42);
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, RandomLookingRoundtrip) {
    Bytes data;
    uint8_t x = 0x5A;
    for (int i = 0; i < 512; ++i) {
        x = static_cast<uint8_t>(x * 17 + 31);
        data.push_back(x);
    }
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, All256DistinctRoundtrip) {
    Bytes data;
    for (int i = 0; i < 256; ++i) data.push_back(static_cast<uint8_t>(i));
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(ar.freq_table.size(), 256u);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, SingleByteRoundtrip) {
    Bytes data = {0xAB};
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, TwoByteRoundtrip) {
    Bytes data = {0x01, 0xFF};
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, LargeInputRoundtrip) {
    Bytes data;
    data.reserve(4096);
    for (size_t i = 0; i < 4096; ++i)
        data.push_back(static_cast<uint8_t>((i * 7 + 13) % 251));
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, AlternatingPatternRoundtrip) {
    Bytes data;
    for (int i = 0; i < 1000; ++i)
        data.push_back(static_cast<uint8_t>(i % 2 ? 0xAA : 0x55));
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, FewSymbolsManyTimesRoundtrip) {
    Bytes data;
    for (int i = 0; i < 2000; ++i)
        data.push_back(static_cast<uint8_t>(i % 3));
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, NearUniformDistributionRoundtrip) {
    Bytes data;
    for (int i = 0; i < 1024; ++i)
        data.push_back(static_cast<uint8_t>(i % 16));
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, SparseHighBytesRoundtrip) {
    Bytes data = {0x00, 0x00, 0x00, 0xFF, 0xFE, 0xFD, 0x00, 0x00};
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, ResultMetadata) {
    Bytes data = {'a', 'a', 'a', 'b', 'c'};
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(ar.original_size, data.size());
    EXPECT_FALSE(ar.freq_table.empty());
    EXPECT_GE(ar.padding_bits, 0);
    EXPECT_LT(ar.padding_bits, 8);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, CompressionRatioNearEntropy) {
    Bytes data = make_skewed_text();
    auto ar = arithmetic_encode(data);
    double entropy_bits = shannon_entropy_bits(data) * static_cast<double>(data.size());
    double payload_bits = static_cast<double>(arithmetic_payload_bits(ar));
    // Model overhead is excluded; coded payload should stay reasonably close to entropy.
    EXPECT_LT(payload_bits, static_cast<double>(data.size() * 8));
    EXPECT_LT(payload_bits, entropy_bits * 1.15 + 64.0);
}

TEST(CompressArithmetic, BeatsOrMatchesHuffmanOnSkewedInput) {
    Bytes data = make_skewed_text();
    auto ar = arithmetic_encode(data);
    auto hr = huffman_encode(data);
    size_t ar_bits = arithmetic_payload_bits(ar);
    size_t hf_bits = hr.encoded.size() * 8u - static_cast<size_t>(hr.padding_bits);
    // Range coding flush adds ~5 bytes overhead; allow modest slack vs Huffman.
    EXPECT_LE(ar_bits, hf_bits + 48u);
}

TEST(CompressArithmetic, HuffmanLikeSimpleRoundtrip) {
    Bytes data = {'a', 'b', 'c', 'a', 'a', 'b'};
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, LongSingleSymbolRoundtrip) {
    Bytes data(8192, 'z');
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(ar.freq_table.size(), 1u);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

TEST(CompressArithmetic, IncrementalSequenceRoundtrip) {
    Bytes data;
    for (int i = 0; i < 300; ++i)
        data.push_back(static_cast<uint8_t>(i % 256));
    auto ar = arithmetic_encode(data);
    EXPECT_EQ(arithmetic_decode(ar), data);
}

// ---- LZ77 ----

TEST(CompressLZ77, SimpleRoundtrip) {
    Bytes data={'a','b','c','a','b','c','d'};
    auto tokens=lz77_encode(data);
    auto dec=lz77_decode(tokens);
    EXPECT_EQ(dec, data);
}

TEST(CompressLZ77, Repetitive) {
    Bytes data={'a','b','c','a','b','c','a','b','c'};
    auto tokens=lz77_encode(data);
    EXPECT_LT(tokens.size(), data.size());  // Should compress
    EXPECT_EQ(lz77_decode(tokens), data);
}

TEST(CompressLZ77, SingleByte) {
    Bytes data={'X'};
    auto tokens=lz77_encode(data);
    EXPECT_EQ(lz77_decode(tokens), data);
}

TEST(CompressLZ77, CustomWindow) {
    Bytes data={'x','y','z','x','y','z','x','y'};
    auto tokens=lz77_encode(data, 64, 8);
    EXPECT_EQ(lz77_decode(tokens), data);
}

// ---- LZW ----

TEST(CompressLZW, SimpleRoundtrip) {
    Bytes data={'a','b','c','a','b','c'};
    auto codes=lzw_encode(data);
    auto dec=lzw_decode(codes);
    EXPECT_EQ(dec, data);
}

TEST(CompressLZW, Sentence) {
    Bytes data;
    std::string s="hello world";
    for (char c:s) data.push_back((uint8_t)c);
    auto dec=lzw_decode(lzw_encode(data));
    EXPECT_EQ(dec, data);
}

// ---- BWT ----

TEST(CompressBWT, Roundtrip) {
    Bytes data={'b','a','n','a','n','a'};
    auto bwt_res=bwt(data);
    auto rec=ibwt(bwt_res);
    EXPECT_EQ(rec, data);
}

TEST(CompressBWT, Canonical) {
    // bwt uses sentinel, so output has n+1 bytes
    Bytes data={'b','a','n','a','n','a'};
    auto bwt_res=bwt(data);
    EXPECT_EQ(bwt_res.data.size(), 7u);  // n+1 (sentinel-based)
    auto rec=ibwt(bwt_res);
    EXPECT_EQ(rec, data);
}

// ---- MTF ----

TEST(CompressMTF, Roundtrip) {
    Bytes data={10,20,30,10,20};
    EXPECT_EQ(mtf_decode(mtf_encode(data)), data);
}

// ---- Delta ----

TEST(CompressDelta, Monotone) {
    Bytes data={10,11,12,13,14};
    auto enc=delta_encode(data);
    EXPECT_EQ(enc[0], 10);
    for (size_t i=1;i<enc.size();++i) EXPECT_EQ(enc[i], 1);
    EXPECT_EQ(delta_decode(enc), data);
}

TEST(CompressDelta, VariedSequence) {
    Bytes data={5,20,3,100,7};
    auto enc=delta_encode(data);
    EXPECT_EQ(delta_decode(enc), data);
}

// ---- Bit helpers ----

TEST(CompressBits, Roundtrip) {
    Bytes data={0xAB,0xCD};
    auto bits=bytes_to_bits(data);
    EXPECT_EQ(bits.size(), 16u);
    int padding=0;
    auto back=bits_to_bytes(bits,padding);
    EXPECT_EQ(back, data);
}

// ---- bzip2-like pipeline ----

TEST(CompressBzip2Like, Roundtrip) {
    Bytes data;
    std::string s="abracadabra";
    for (char c:s) data.push_back((uint8_t)c);
    auto compressed=bzip2_like_compress(data);
    auto recovered=bzip2_like_decompress(compressed,0);
    EXPECT_EQ(recovered, data);
}

TEST(CompressBzip2Like, PrimaryIndexHeader) {
    Bytes data={'m','i','s','s','i','s','s','i','p','p','i'};
    auto compressed=bzip2_like_compress(data);
    EXPECT_GE(compressed.size(), 4u);
    int pi=(compressed[0]<<24)|(compressed[1]<<16)|(compressed[2]<<8)|compressed[3];
    EXPECT_GE(pi, 0);
    EXPECT_LT(pi, (int)data.size()+1);
    EXPECT_EQ(bzip2_like_decompress(compressed, pi), data);
}

// ---- Haar Wavelet (lossy) ----

namespace {

// Slowly-varying synthetic ramp/sine, byte-valued, smooth enough that
// small-to-moderate detail coefficients dominate — exactly the regime
// where Haar thresholding is expected to zero out many coefficients.
Bytes make_smooth_ramp(size_t n) {
    Bytes data; data.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double v = 128.0 + 40.0 * std::sin(static_cast<double>(i) * 0.05);
        data.push_back(static_cast<uint8_t>(v));
    }
    return data;
}

} // namespace

TEST(CompressWavelet, EmptyRoundtripLossless) {
    Bytes data;
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_TRUE(compressed.empty());
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, SingleByteRoundtripLossless) {
    Bytes data = {0x7F};
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, TwoByteRoundtripLossless) {
    Bytes data = {10, 250};
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, OddLengthRoundtripLossless) {
    Bytes data = {1, 2, 3, 4, 5};
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, EvenLengthRoundtripLossless) {
    Bytes data = {1, 2, 3, 4, 5, 6, 7, 8};
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, AllBytesRoundtripLossless) {
    Bytes data;
    for (int i = 0; i < 256; ++i) data.push_back(static_cast<uint8_t>(i));
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, SmoothRampRoundtripLossless) {
    Bytes data = make_smooth_ramp(257);  // odd length on purpose
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, ConstantSignalRoundtripLossless) {
    Bytes data(129, 0x42);  // odd length, all-equal bytes -> all diffs are 0
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, MonotonicSizeReductionWithThreshold) {
    Bytes data = make_smooth_ramp(512);
    size_t prev_size = wavelet_compress(data, 0.0).size();
    for (double thr : {2.0, 5.0, 10.0, 20.0, 40.0}) {
        size_t sz = wavelet_compress(data, thr).size();
        EXPECT_LE(sz, prev_size)
            << "compressed size should not increase as threshold grows (thr=" << thr << ")";
        prev_size = sz;
    }
    // A clearly nonzero threshold should strictly beat threshold=0 on a smooth signal.
    size_t size_zero = wavelet_compress(data, 0.0).size();
    size_t size_large = wavelet_compress(data, 40.0).size();
    EXPECT_LT(size_large, size_zero);
}

TEST(CompressWavelet, LossyReconstructionCloseToOriginal) {
    Bytes data = make_smooth_ramp(512);
    for (double threshold : {1.0, 4.0, 8.0, 16.0}) {
        auto compressed = wavelet_compress(data, threshold);
        auto recon = wavelet_decompress(compressed);
        ASSERT_EQ(recon.size(), data.size());
        double max_abs_error = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            double err = std::abs(static_cast<double>(data[i]) - static_cast<double>(recon[i]));
            max_abs_error = std::max(max_abs_error, err);
        }
        // Any pair whose detail coefficient survives thresholding decodes
        // exactly; a zeroed coefficient contributes at most |diff| < threshold
        // of per-element error. So the aggregate error is bounded by threshold.
        EXPECT_LE(max_abs_error, threshold)
            << "reconstruction error should stay within the threshold bound";
    }
}

TEST(CompressWavelet, LossyIsSmallerThanLosslessOnSmoothRamp) {
    Bytes data = make_smooth_ramp(512);
    size_t lossless_size = wavelet_compress(data, 0.0).size();
    size_t lossy_size = wavelet_compress(data, 12.0).size();
    EXPECT_LT(lossy_size, lossless_size);
    auto recon = wavelet_decompress(wavelet_compress(data, 12.0));
    ASSERT_EQ(recon.size(), data.size());
    EXPECT_NE(recon, data);  // genuinely lossy at this threshold
}

TEST(CompressWavelet, ZeroThresholdNeverAltersData) {
    // Threshold exactly 0.0 must never zero a coefficient (strict '<' comparison),
    // so this is lossless even for data with lots of variation.
    Bytes data;
    uint8_t x = 3;
    for (int i = 0; i < 300; ++i) { x = static_cast<uint8_t>(x * 37 + 11); data.push_back(x); }
    auto compressed = wavelet_compress(data, 0.0);
    EXPECT_EQ(wavelet_decompress(compressed), data);
}

TEST(CompressWavelet, NegativeThresholdTreatedAsZero) {
    Bytes data = make_smooth_ramp(64);
    auto compressed_neg = wavelet_compress(data, -5.0);
    auto compressed_zero = wavelet_compress(data, 0.0);
    EXPECT_EQ(compressed_neg, compressed_zero);
    EXPECT_EQ(wavelet_decompress(compressed_neg), data);
}

TEST(CompressWavelet, EmptyInputWithNonzeroThreshold) {
    Bytes data;
    auto compressed = wavelet_compress(data, 25.0);
    EXPECT_TRUE(compressed.empty());
    EXPECT_TRUE(wavelet_decompress(compressed).empty());
}

TEST(CompressWavelet, MalformedShortInputDecodesToEmpty) {
    Bytes garbage = {1, 2, 3};  // shorter than the 6-byte minimum header
    EXPECT_TRUE(wavelet_decompress(garbage).empty());
}
