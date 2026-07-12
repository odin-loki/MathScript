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

// ========================== ANS (rANS) coding ==========================
// Range Asymmetric Numeral Systems on a byte alphabet. Frequencies are
// normalized to a fixed power-of-two scale (4096); the compact per-symbol
// frequency table and original symbol count are stored alongside the coded
// bytes because rANS has no natural end-of-stream marker.
struct AnsResult {
    Bytes encoded;                                      // rANS stream (renorm bytes + 4-byte final state)
    std::vector<std::pair<uint8_t, uint32_t>> freq_table; // symbol → normalized frequency (present symbols only)
    size_t original_size;                               // symbols to decode
    int padding_bits;                                   // always 0 (byte-aligned output)
};
AnsResult ans_encode(const Bytes& data);
Bytes     ans_decode(const AnsResult& ar);

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

// ========================== Golomb-Rice coding ==========================
// Entropy coding for non-negative integers (Rice coding: Golomb with M = 2^m_bits).
// Each value v is split into quotient q = v >> m_bits and remainder r = v & (M-1).
// The quotient is unary-coded as q ones followed by a terminating zero; the
// remainder is written directly in m_bits bits (MSB first). When m_bits == 0 this
// degenerates to pure unary coding (M = 1, remainder absent). The bitstream is
// packed into bytes via bits_to_bytes(); decoders require an explicit symbol count
// because there is no natural end-of-stream marker (same pattern as Huffman decode).
Bytes                 golomb_rice_encode(const std::vector<uint32_t>& values, int m_bits);
std::vector<uint32_t> golomb_rice_decode(const Bytes& encoded, int m_bits, size_t count);

// ========================== Combined pipelines ==========================
// BWT + MTF + RLE (like bzip2 core)
Bytes bzip2_like_compress(const Bytes& data);
Bytes bzip2_like_decompress(const Bytes& data, int bwt_primary);

// ========================== Haar Wavelet (lossy) ==========================
/// Single-level Haar Discrete Wavelet Transform compressor with hard
/// coefficient thresholding.
///
/// The input byte sequence is treated as a numeric signal and split into
/// adjacent pairs (x[2i], x[2i+1]). Each pair is decomposed via the
/// reversible integer Haar lifting scheme:
///   diff = x[2i] - x[2i+1]
///   avg  = x[2i+1] + (diff >> 1)      // == floor((x[2i]+x[2i+1]) / 2)
/// `avg` is the low-frequency/approximation coefficient (always in
/// [0,255]); `diff` is the high-frequency/detail coefficient (in
/// [-255,255]). Any detail coefficient with |diff| < @p threshold is
/// hard-thresholded to exactly 0 before serialization — this is the lossy
/// step, and the source of any size/perceptual tradeoff. A trailing odd
/// byte (when the input length is odd) is stored verbatim, untouched by
/// thresholding.
///
/// Serialized layout (all multi-byte integers big-endian):
///   [0..3]  original size n (uint32_t)
///   [4]     odd_flag (1 if n is odd, else 0)
///   [5]     odd_value (the unpaired trailing byte iff odd_flag==1, else 0)
///   [6..]   n/2 raw approximation bytes (avg, one byte each)
///   [...]   n/2 marker-coded detail coefficients (diff): a zero
///           coefficient is a single 0x00 byte; a nonzero coefficient is
///           a 0x01 marker followed by its zigzag magnitude as a 2-byte
///           big-endian value. A thresholded-to-zero coefficient always
///           costs exactly 1 byte versus 3 for a surviving one, so the
///           serialized size shrinks by exactly 2 bytes per extra
///           coefficient zeroed by a larger threshold.
///
/// @param data       Input bytes, treated as a numeric sequence.
/// @param threshold  Hard-threshold magnitude applied to detail
///                    coefficients; negative values are clamped to 0.
/// @return Serialized compressed byte stream (see layout above); empty
///         for empty input.
/// @note @p threshold == 0.0 is lossless: wavelet_decompress() reproduces
///       `data` exactly, including odd-length inputs.
/// @note Defensive: empty input returns an empty result; a negative
///       threshold is treated as 0.0 (no thresholding).
Bytes wavelet_compress(const Bytes& data, double threshold);

/// Inverse of wavelet_compress(): parses the serialized coefficient
/// stream and reconstructs a byte sequence via the inverse Haar lifting
/// step (b = avg - (diff >> 1), a = b + diff).
///
/// @param compressed  Byte stream produced by wavelet_compress().
/// @return Reconstructed bytes; exactly equal to the original input when
///         the data was compressed with threshold == 0.0, otherwise a
///         lossy approximation whose per-element error is bounded by the
///         threshold used at compression time. Empty on empty or
///         malformed/truncated input.
Bytes wavelet_decompress(const Bytes& compressed);

} // namespace compress
} // namespace ms
