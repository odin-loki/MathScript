#include "ms/compress/compress.hpp"
#include <gtest/gtest.h>

using namespace ms::compress;

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
