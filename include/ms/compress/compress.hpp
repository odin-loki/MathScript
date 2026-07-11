#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ms {
namespace compress {

using Bytes = std::vector<uint8_t>;

// ========================== RLE ==========================
Bytes rle_encode(const Bytes& data);
Bytes rle_decode(const Bytes& data);

// ========================== Huffman ==========================
struct HuffmanResult {
    Bytes encoded;           // bit-packed bytes
    std::vector<std::pair<uint8_t, std::string>> codebook;  // symbol → bit string
    int padding_bits;        // number of padding bits at end
};
HuffmanResult huffman_encode(const Bytes& data);
Bytes          huffman_decode(const HuffmanResult& hr, size_t original_size);

// ========================== Arithmetic (range) coding ==========================
// Integer range coding — a numerically-stable variant of arithmetic coding that
// maintains a (low, range) interval with 32-bit unsigned integers (never
// floating-point), renormalizing by shifting out bytes when range falls below
// 2^24. The frequency table and original symbol count are stored alongside the
// coded bytes because range coding has no natural end-of-stream marker.
struct ArithmeticResult {
    Bytes encoded;                                      // range-coded output bytes
    std::vector<std::pair<uint8_t, uint32_t>> freq_table; // symbol → frequency (symbols present only)
    size_t original_size;                               // symbols to decode (no natural EOF marker)
    int padding_bits;                                   // always 0 (byte-aligned output)
};
ArithmeticResult arithmetic_encode(const Bytes& data);
Bytes            arithmetic_decode(const ArithmeticResult& ar);

// ========================== LZ77 ==========================
struct LZ77Token {
    uint16_t offset, length;
    uint8_t next_char;
};
std::vector<LZ77Token> lz77_encode(const Bytes& data,
                                    int window = 255, int lookahead = 15);
Bytes                   lz77_decode(const std::vector<LZ77Token>& tokens);

// ========================== LZ78 / LZW ==========================
std::vector<uint32_t> lzw_encode(const Bytes& data);
Bytes                  lzw_decode(const std::vector<uint32_t>& codes);

// ========================== Burrows-Wheeler Transform ==========================
struct BWTResult { Bytes data; int primary_index; };
BWTResult bwt(const Bytes& data);
Bytes     ibwt(const BWTResult& result);

// ========================== Move-to-Front ==========================
Bytes mtf_encode(const Bytes& data);
Bytes mtf_decode(const Bytes& data);

// ========================== Delta / Difference coding ==========================
Bytes delta_encode(const Bytes& data);
Bytes delta_decode(const Bytes& data);

// ========================== Bit manipulation helpers ==========================
std::string  bytes_to_bits(const Bytes& data);
Bytes        bits_to_bytes(const std::string& bits, int& padding);

// ========================== Combined pipelines ==========================
// BWT + MTF + RLE (like bzip2 core)
Bytes bzip2_like_compress(const Bytes& data);
Bytes bzip2_like_decompress(const Bytes& data, int bwt_primary);

} // namespace compress
} // namespace ms
