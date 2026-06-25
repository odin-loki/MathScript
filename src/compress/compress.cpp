#include "ms/compress/compress.hpp"
#include <algorithm>
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

} // namespace compress
} // namespace ms
