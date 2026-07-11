#include "ms/compress/compress.hpp"
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
