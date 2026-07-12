#include "ms/compress/compress.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <string>

namespace ms {
namespace compress {

// ========================== RLE ==========================
// Format: [count][byte] pairs (max run 255)
Bytes rle_encode(const Bytes& data) {
    Bytes out; size_t i=0, n=data.size();
    while (i<n) {
        uint8_t v=data[i]; uint8_t cnt=1;
        while (i+cnt<n && data[i+cnt]==v && cnt<255) ++cnt;
        out.push_back(cnt); out.push_back(v);
        i+=cnt;
    }
    return out;
}
Bytes rle_decode(const Bytes& data) {
    Bytes out;
    for (size_t i=0;i+1<data.size();i+=2)
        for (uint8_t c=0;c<data[i];++c) out.push_back(data[i+1]);
    return out;
}

// ========================== Huffman ==========================
struct HNode {
    uint8_t sym; int freq; int left=-1,right=-1;
    bool operator>(const HNode& o) const { return freq>o.freq; }
};

HuffmanResult huffman_encode(const Bytes& data) {
    if (data.empty()) return {};
    std::map<uint8_t,int> freq;
    for (uint8_t b:data) freq[b]++;
    std::vector<HNode> nodes;
    for (auto&[s,f]:freq) nodes.push_back({s,f});
    // Build heap
    auto cmp=[](const std::pair<int,int>& a,const std::pair<int,int>& b){return a.first>b.first;};
    std::priority_queue<std::pair<int,int>,std::vector<std::pair<int,int>>,decltype(cmp)> pq(cmp);
    for (size_t i=0;i<nodes.size();++i) pq.push({nodes[i].freq,(int)i});
    while (pq.size()>1) {
        auto [f1,i1]=pq.top();pq.pop();
        auto [f2,i2]=pq.top();pq.pop();
        HNode parent; parent.sym=0; parent.freq=f1+f2; parent.left=i1; parent.right=i2;
        nodes.push_back(parent); int ni=nodes.size()-1;
        pq.push({parent.freq,ni});
    }
    int root=pq.top().second;
    // Build codebook
    std::map<uint8_t,std::string> codes;
    std::function<void(int,std::string)> build=[&](int n,std::string code){
        if (nodes[n].left==-1){codes[nodes[n].sym]=code.empty()?"0":code;return;}
        build(nodes[n].left,code+"0"); build(nodes[n].right,code+"1");
    };
    build(root,"");
    // Encode
    std::string bits;
    for (uint8_t b:data) bits+=codes[b];
    Bytes encoded; int padding=0;
    encoded=bits_to_bytes(bits,padding);
    std::vector<std::pair<uint8_t,std::string>> cb(codes.begin(),codes.end());
    return {encoded,cb,padding};
}

Bytes huffman_decode(const HuffmanResult& hr, size_t orig_size) {
    // Rebuild codebook
    std::map<std::string,uint8_t> decode_map;
    for (auto&[sym,code]:hr.codebook) decode_map[code]=sym;
    // Decode bits
    std::string bits=bytes_to_bits(hr.encoded);
    if (hr.padding_bits>0 && hr.padding_bits<8)
        bits=bits.substr(0,bits.size()-hr.padding_bits);
    Bytes out; std::string cur;
    for (char b:bits) {
        cur+=b;
        auto it=decode_map.find(cur);
        if (it!=decode_map.end()){out.push_back(it->second);cur="";}
        if (out.size()>=orig_size) break;
    }
    return out;
}

// ========================== Arithmetic (range) coding ==========================
// Russian range coder (Shelwien/Subbotin): 64-bit low, 32-bit range/code,
// renormalization with underflow avoidance via range clamping.
namespace {

constexpr uint32_t kTop = 1u << 24;

struct RangeModel {
    std::vector<uint8_t> sym;
    std::vector<uint32_t> cum; // size n+1, cum[0]=0, cum[n]=total
    uint32_t total = 0;

    static RangeModel from_data(const Bytes& data) {
        std::map<uint8_t, int> freq;
        for (uint8_t b : data) ++freq[b];
        RangeModel m;
        for (auto& [s, f] : freq) {
            m.sym.push_back(s);
            m.cum.push_back(m.total);
            m.total += static_cast<uint32_t>(f);
        }
        m.cum.push_back(m.total);
        return m;
    }

    static RangeModel from_table(const std::vector<std::pair<uint8_t, uint32_t>>& ft) {
        RangeModel m;
        for (auto& [s, f] : ft) {
            m.sym.push_back(s);
            m.cum.push_back(m.total);
            m.total += f;
        }
        m.cum.push_back(m.total);
        return m;
    }

    int index_of(uint8_t s) const {
        auto it = std::lower_bound(sym.begin(), sym.end(), s);
        if (it == sym.end() || *it != s) return -1;
        return static_cast<int>(it - sym.begin());
    }

    int symbol_for_count(uint32_t count) const {
        int lo = 0, hi = static_cast<int>(sym.size()) - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (cum[mid] <= count) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }
};

struct ByteWriter {
    Bytes out;
    void write(uint8_t b) { out.push_back(b); }
};

struct ByteReader {
    const Bytes& data;
    size_t pos = 0;
    explicit ByteReader(const Bytes& data) : data(data) {}
    uint8_t read() { return pos < data.size() ? data[pos++] : 0; }
};

struct RngCoder {
    uint64_t low = 0;
    uint32_t range = UINT32_MAX;
    uint32_t code = 0;
    ByteWriter* out = nullptr;
    ByteReader* in = nullptr;

    void renormalize_encode() {
        if (range >= kTop) return;
        do {
            if (static_cast<uint8_t>((low ^ (low + range)) >> 56))
                range = ((static_cast<uint32_t>(low) | (kTop - 1)) - static_cast<uint32_t>(low));
            out->write(static_cast<uint8_t>(low >> 56));
            range <<= 8;
            low <<= 8;
        } while (range < kTop);
    }

    void renormalize_decode() {
        while (range < kTop) {
            if (static_cast<uint8_t>((low ^ (low + range)) >> 56))
                range = ((static_cast<uint32_t>(low) | (kTop - 1)) - static_cast<uint32_t>(low));
            code = (code << 8) | in->read();
            range <<= 8;
            low <<= 8;
        }
    }

    void encode(uint32_t cum_freq, uint32_t freq, uint32_t total) {
        low += static_cast<uint64_t>(cum_freq) * (range /= total);
        range *= freq;
        renormalize_encode();
    }

    uint32_t get_freq(uint32_t total) {
        return code / (range /= total);
    }

    void decode_update(uint32_t cum_freq, uint32_t freq) {
        uint32_t temp = cum_freq * range;
        low += temp;
        code -= temp;
        range *= freq;
        renormalize_decode();
    }

    void finish_encode() {
        for (int i = 0; i < 8; ++i) {
            out->write(static_cast<uint8_t>(low >> 56));
            low <<= 8;
        }
    }

    void start_decode(ByteReader& reader) {
        in = &reader;
        for (int i = 0; i < 8; ++i)
            code = (code << 8) | in->read();
    }
};

} // namespace

ArithmeticResult arithmetic_encode(const Bytes& data) {
    ArithmeticResult result;
    result.original_size = data.size();
    result.padding_bits = 0;
    if (data.empty()) return result;

    RangeModel model = RangeModel::from_data(data);
    result.freq_table.reserve(model.sym.size());
    for (size_t i = 0; i < model.sym.size(); ++i)
        result.freq_table.emplace_back(model.sym[i], model.cum[i + 1] - model.cum[i]);

    ByteWriter writer;
    RngCoder rc;
    rc.out = &writer;
    for (uint8_t b : data) {
        int idx = model.index_of(b);
        uint32_t freq = model.cum[idx + 1] - model.cum[idx];
        rc.encode(model.cum[idx], freq, model.total);
    }
    rc.finish_encode();
    result.encoded = std::move(writer.out);
    return result;
}

Bytes arithmetic_decode(const ArithmeticResult& ar) {
    if (ar.original_size == 0) return {};
    RangeModel model = RangeModel::from_table(ar.freq_table);
    ByteReader reader(ar.encoded);
    RngCoder rc;
    rc.start_decode(reader);
    Bytes out;
    out.reserve(ar.original_size);
    for (size_t i = 0; i < ar.original_size; ++i) {
        uint32_t count = rc.get_freq(model.total);
        if (count >= model.total) count = model.total - 1;
        int idx = model.symbol_for_count(count);
        uint32_t freq = model.cum[idx + 1] - model.cum[idx];
        rc.decode_update(model.cum[idx], freq);
        out.push_back(model.sym[idx]);
    }
    return out;
}

// ========================== LZ77 ==========================
std::vector<LZ77Token> lz77_encode(const Bytes& data, int window, int lookahead) {
    std::vector<LZ77Token> tokens;
    size_t pos=0, n=data.size();
    while (pos<n) {
        int best_off=0, best_len=0;
        size_t wstart=pos>=(size_t)window?pos-window:0;
        for (size_t start=wstart;start<pos;++start) {
            int len=0;
            while (pos+len<n && data[start+len]==data[pos+len] && len<lookahead) ++len;
            if (len>best_len){best_len=len;best_off=(int)(pos-start);}
        }
        uint8_t nc=pos+best_len<n?data[pos+best_len]:0;
        tokens.push_back({(uint16_t)best_off,(uint16_t)best_len,nc});
        pos+=best_len+1;
    }
    return tokens;
}
Bytes lz77_decode(const std::vector<LZ77Token>& tokens) {
    Bytes out;
    for (auto&t:tokens) {
        if (t.offset>0&&t.length>0) {
            size_t start=out.size()-t.offset;
            for (uint16_t i=0;i<t.length;++i) out.push_back(out[start+i]);
        }
        if (t.next_char||t.length==0) out.push_back(t.next_char);
    }
    return out;
}

// ========================== LZW ==========================
std::vector<uint32_t> lzw_encode(const Bytes& data) {
    std::map<std::string,uint32_t> dict;
    for (int i=0;i<256;++i) dict[std::string(1,(char)i)]=i;
    uint32_t code=256;
    std::string w; std::vector<uint32_t> out;
    for (uint8_t c:data) {
        std::string wc=w+char(c);
        if (dict.count(wc)) w=wc;
        else {out.push_back(dict[w]); dict[wc]=code++; w=std::string(1,char(c));}
    }
    if (!w.empty()) out.push_back(dict[w]);
    return out;
}
Bytes lzw_decode(const std::vector<uint32_t>& codes) {
    std::map<uint32_t,std::string> dict;
    for (int i=0;i<256;++i) dict[i]=std::string(1,(char)i);
    uint32_t code=256;
    if (codes.empty()) return {};
    std::string w=dict[codes[0]]; Bytes out(w.begin(),w.end());
    for (size_t i=1;i<codes.size();++i) {
        std::string entry;
        if (dict.count(codes[i])) entry=dict[codes[i]];
        else entry=w+w[0];
        for (char c:entry) out.push_back((uint8_t)c);
        dict[code++]=w+entry[0]; w=entry;
    }
    return out;
}

// ========================== BWT ==========================
// Uses a sentinel (value 256) appended internally to ensure T is a single cycle.
// This is required for correct ibwt on strings with repeated characters.
BWTResult bwt(const Bytes& data) {
    int n = (int)data.size();
    int m = n + 1;  // includes sentinel at position n
    // Internal alphabet: data bytes shifted to 1..256, sentinel = 0
    auto ch = [&](int rotation, int pos) -> uint16_t {
        int p = (rotation + pos) % m;
        return p == n ? (uint16_t)0 : (uint16_t)(data[p] + 1);
    };
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        for (int i = 0; i < m; i++) {
            uint16_t ca = ch(a, i), cb = ch(b, i);
            if (ca != cb) return ca < cb;
        }
        return false;
    });
    // L: last column, using 0x00 to represent sentinel in output
    Bytes L(m);
    int primary = -1;
    for (int i = 0; i < m; i++) {
        int last = (idx[i] + m - 1) % m;
        uint16_t cv = ch(idx[i], m - 1);  // last character of sorted rotation i
        // ch(idx[i], m-1) == ch at position (idx[i]+m-1)%m = (idx[i]-1+m)%m
        // Recompute via direct access
        int lpos = (idx[i] + m - 1) % m;
        L[i] = (lpos == n) ? (uint8_t)0 : data[lpos];
        (void)cv;
        if (idx[i] == 0) primary = i;
    }
    return {L, primary};
}

Bytes ibwt(const BWTResult& res) {
    const Bytes& L = res.data;
    int m = (int)L.size();  // n+1 (includes sentinel row)
    // Sorted L = F column
    Bytes Fs = L;
    std::sort(Fs.begin(), Fs.end());
    // F_idx[c]: first position in F where c appears
    std::vector<int> F_idx(256, 0);
    {
        std::vector<int> cnt(256, 0);
        for (uint8_t b : L) cnt[b]++;
        int tot = 0;
        for (int c = 0; c < 256; c++) { F_idx[c] = tot; tot += cnt[c]; }
    }
    // T[i] = F_idx[L[i]] + #{j < i : L[j] == L[i]}
    std::vector<int> T(m);
    std::vector<int> occ(256, 0);
    for (int i = 0; i < m; i++) {
        T[i] = F_idx[L[i]] + occ[L[i]];
        occ[L[i]]++;
    }
    // T maps row i → predecessor rotation (position k → k-1).
    // T_inv maps row i → successor rotation (position k → k+1).
    // Use T_inv to visit rotations in order: pos 0, 1, ..., n-1, n(sentinel).
    std::vector<int> T_inv(m);
    for (int i = 0; i < m; i++) T_inv[T[i]] = i;

    Bytes out;
    out.reserve(m - 1);
    int r = res.primary_index;
    for (int i = 0; i < m; i++) {
        if (Fs[r] != 0) out.push_back(Fs[r]);  // skip sentinel character
        r = T_inv[r];
    }
    return out;
}

// ========================== MTF ==========================
Bytes mtf_encode(const Bytes& data) {
    std::vector<uint8_t> alphabet(256); std::iota(alphabet.begin(),alphabet.end(),0);
    Bytes out;
    for (uint8_t c:data) {
        auto it=std::find(alphabet.begin(),alphabet.end(),c);
        int pos=it-alphabet.begin();
        out.push_back((uint8_t)pos);
        alphabet.erase(it); alphabet.insert(alphabet.begin(),c);
    }
    return out;
}
Bytes mtf_decode(const Bytes& data) {
    std::vector<uint8_t> alphabet(256); std::iota(alphabet.begin(),alphabet.end(),0);
    Bytes out;
    for (uint8_t pos:data) {
        uint8_t c=alphabet[pos];
        out.push_back(c);
        alphabet.erase(alphabet.begin()+pos); alphabet.insert(alphabet.begin(),c);
    }
    return out;
}

// ========================== Delta ==========================
Bytes delta_encode(const Bytes& data) {
    if (data.empty()) return {};
    Bytes out; out.push_back(data[0]);
    for (size_t i=1;i<data.size();++i) out.push_back((uint8_t)(data[i]-data[i-1]));
    return out;
}
Bytes delta_decode(const Bytes& data) {
    if (data.empty()) return {};
    Bytes out; out.push_back(data[0]);
    for (size_t i=1;i<data.size();++i) out.push_back((uint8_t)(out.back()+data[i]));
    return out;
}

// ========================== Bit helpers ==========================
std::string bytes_to_bits(const Bytes& data) {
    std::string bits;
    for (uint8_t b:data)
        for (int i=7;i>=0;--i) bits+=(char)('0'+((b>>i)&1));
    return bits;
}
Bytes bits_to_bytes(const std::string& bits, int& padding) {
    Bytes out; std::string padded=bits;
    padding=(8-bits.size()%8)%8;
    padded+=std::string(padding,'0');
    for (size_t i=0;i<padded.size();i+=8) {
        uint8_t b=0;
        for (int j=0;j<8;++j) b=(b<<1)|(padded[i+j]=='1'?1:0);
        out.push_back(b);
    }
    return out;
}

// ========================== bzip2-like pipeline ==========================
Bytes bzip2_like_compress(const Bytes& data) {
    auto bwt_res=bwt(data);
    auto mtf=mtf_encode(bwt_res.data);
    auto rle=rle_encode(mtf);
    // Prepend primary index as 4 bytes
    Bytes out={
        (uint8_t)(bwt_res.primary_index>>24),
        (uint8_t)((bwt_res.primary_index>>16)&0xFF),
        (uint8_t)((bwt_res.primary_index>>8)&0xFF),
        (uint8_t)(bwt_res.primary_index&0xFF)
    };
    out.insert(out.end(),rle.begin(),rle.end());
    return out;
}
Bytes bzip2_like_decompress(const Bytes& data, int) {
    if (data.size()<4) return {};
    int pi=(data[0]<<24)|(data[1]<<16)|(data[2]<<8)|data[3];
    Bytes rle(data.begin()+4,data.end());
    auto mtf=rle_decode(rle);
    auto bwt_data=mtf_decode(mtf);
    return ibwt({bwt_data,pi});
}

// ========================== Haar Wavelet (lossy) ==========================
// Self-contained serialization helpers (deliberately independent of
// bytes_to_bits/bits_to_bytes, which are shared with the Huffman path).
namespace {

void wv_push_u32_be(Bytes& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

uint32_t wv_read_u32_be(const Bytes& data, size_t pos) {
    return (static_cast<uint32_t>(data[pos]) << 24) |
           (static_cast<uint32_t>(data[pos + 1]) << 16) |
           (static_cast<uint32_t>(data[pos + 2]) << 8) |
           static_cast<uint32_t>(data[pos + 3]);
}

// Zigzag maps a signed detail coefficient (range [-255,255]) to an
// unsigned value in [0,510], used only to keep the on-disk magnitude
// representation simple/unsigned; it does not affect the byte count.
uint32_t wv_zigzag_encode(int32_t v) {
    return (v >= 0) ? static_cast<uint32_t>(v) * 2u
                     : static_cast<uint32_t>(-v) * 2u - 1u;
}
int32_t wv_zigzag_decode(uint32_t z) {
    return (z % 2 == 0) ? static_cast<int32_t>(z / 2)
                          : -static_cast<int32_t>((z + 1) / 2);
}

// Detail coefficients are marker-coded rather than fixed-width so that a
// thresholded-to-zero coefficient always costs exactly 1 byte, while any
// surviving (nonzero) coefficient costs exactly 3 bytes — independent of
// its magnitude. This keeps the zero/nonzero size tradeoff unambiguous:
// serialized size shrinks by exactly 2 bytes for every extra coefficient a
// larger threshold zeroes out, regardless of the underlying data's scale.
void wv_push_diff(Bytes& out, int32_t diff) {
    if (diff == 0) { out.push_back(0); return; }
    uint32_t z = wv_zigzag_encode(diff);
    out.push_back(1);
    out.push_back(static_cast<uint8_t>((z >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(z & 0xFF));
}

int32_t wv_read_diff(const Bytes& data, size_t& pos) {
    if (pos >= data.size()) return 0;
    uint8_t marker = data[pos++];
    if (marker == 0) return 0;
    if (pos + 2 > data.size()) return 0;
    uint32_t z = (static_cast<uint32_t>(data[pos]) << 8) | static_cast<uint32_t>(data[pos + 1]);
    pos += 2;
    return wv_zigzag_decode(z);
}

} // namespace

Bytes wavelet_compress(const Bytes& data, double threshold) {
    if (data.empty()) return {};
    double thr = threshold > 0.0 ? threshold : 0.0;

    size_t n = data.size();
    size_t pairs = n / 2;
    bool odd = (n % 2) != 0;

    Bytes out;
    wv_push_u32_be(out, static_cast<uint32_t>(n));
    out.push_back(odd ? (uint8_t)1 : (uint8_t)0);
    out.push_back(odd ? data[n - 1] : (uint8_t)0);

    Bytes avg_bytes; avg_bytes.reserve(pairs);
    std::vector<int32_t> diffs; diffs.reserve(pairs);
    for (size_t i = 0; i < pairs; ++i) {
        int32_t a = data[2 * i], b = data[2 * i + 1];
        int32_t diff = a - b;               // detail coefficient
        int32_t avg = b + (diff >> 1);      // approximation == floor((a+b)/2)
        int32_t adiff = diff < 0 ? -diff : diff;
        if (static_cast<double>(adiff) < thr) diff = 0;  // hard threshold
        avg_bytes.push_back(static_cast<uint8_t>(avg));
        diffs.push_back(diff);
    }
    out.insert(out.end(), avg_bytes.begin(), avg_bytes.end());
    for (int32_t d : diffs) wv_push_diff(out, d);
    return out;
}

Bytes wavelet_decompress(const Bytes& compressed) {
    if (compressed.size() < 6) return {};
    size_t pos = 0;
    uint32_t n = wv_read_u32_be(compressed, pos); pos += 4;
    uint8_t odd = compressed[pos++];
    uint8_t odd_value = compressed[pos++];
    size_t pairs = n / 2;
    if (pos + pairs > compressed.size()) return {};

    Bytes avg_bytes(compressed.begin() + pos, compressed.begin() + pos + pairs);
    pos += pairs;

    Bytes out; out.reserve(n);
    for (size_t i = 0; i < pairs; ++i) {
        int32_t diff = wv_read_diff(compressed, pos);
        int32_t avg = avg_bytes[i];
        int32_t b = avg - (diff >> 1);
        int32_t a = b + diff;
        out.push_back(static_cast<uint8_t>(a));
        out.push_back(static_cast<uint8_t>(b));
    }
    if (odd) out.push_back(odd_value);
    return out;
}

} // namespace compress
} // namespace ms
