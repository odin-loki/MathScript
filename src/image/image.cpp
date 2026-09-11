// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#define _USE_MATH_DEFINES
#include "ms/image/image.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <tuple>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace image {

// ========================== Image ==========================

namespace {

/// `r * c * ch` as a count, computed in `std::size_t` rather than in `int`.
///
/// `data(r*c*ch, fill)` was `int` arithmetic, and `impad(img, 1000000)` asks for a
/// 2000002 x 2000002 image: the product is 4,000,008,000,004, which is not an `int`.
/// Signed overflow is undefined, and what it did in practice was allocate a buffer of
/// whatever the wrap produced and then let `at()` -- also `int` arithmetic, also
/// overflowing -- write far outside it. That is a segmentation fault reachable from one
/// line of a REPL session.
///
/// In `size_t` the count is exact, so an image too large to hold fails at the
/// allocation instead of succeeding at the wrong size. Failing there is still a
/// process death in a tree built without exceptions, which is why the callers bound
/// the request as well; but a death at the allocation is a resource limit, and a write
/// outside the buffer is a memory-safety defect, and only one of those two can be left
/// to a caller.
std::size_t image_element_count(int r, int c, int ch) {
    if (r <= 0 || c <= 0 || ch <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(c) *
           static_cast<std::size_t>(ch);
}

} // namespace

Image::Image(int r, int c, int ch, float fill)
    : rows(r > 0 ? r : 0), cols(c > 0 ? c : 0), channels(ch > 0 ? ch : 0),
      data(image_element_count(r, c, ch), fill) {}

// The index is computed in `size_t` for the reason above: in `int` it overflows on an
// image this type can legitimately hold -- 46341 x 46341 single-channel is past INT_MAX
// -- and an overflowed index is an out-of-bounds access rather than a wrong pixel.
float& Image::at(int r, int c, int ch) {
    return data[(static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(c)) *
                    static_cast<std::size_t>(channels) +
                static_cast<std::size_t>(ch)];
}
float Image::at(int r, int c, int ch) const {
    return data[(static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(c)) *
                    static_cast<std::size_t>(channels) +
                static_cast<std::size_t>(ch)];
}

float bilinear_sample(const Image& img, float r, float c, int ch) {
    int r0=(int)r, c0=(int)c;
    int r1=std::min(r0+1,img.rows-1), c1=std::min(c0+1,img.cols-1);
    r0=std::max(0,r0); c0=std::max(0,c0);
    float dr=r-r0, dc=c-c0;
    return img.at(r0,c0,ch)*(1-dr)*(1-dc)+img.at(r0,c1,ch)*(1-dr)*dc
          +img.at(r1,c0,ch)*dr*(1-dc)+img.at(r1,c1,ch)*dr*dc;
}

Image im_from_data(int r,int c,int ch,const std::vector<float>& d) {
    Image img(r,c,ch); img.data=d; return img;
}

// ========================== Color Conversions ==========================

Image rgb2gray(const Image& img) {
    Image out(img.rows, img.cols, 1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c)
        out.at(r,c,0) = 0.299f*img.at(r,c,0)+0.587f*img.at(r,c,1)+0.114f*img.at(r,c,2);
    return out;
}

Image gray2rgb(const Image& img) {
    Image out(img.rows, img.cols, 3);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c)
        out.at(r,c,0)=out.at(r,c,1)=out.at(r,c,2)=img.at(r,c,0);
    return out;
}

Image rgb2hsv(const Image& img) {
    Image out(img.rows, img.cols, 3);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        float R=img.at(r,c,0),G=img.at(r,c,1),B=img.at(r,c,2);
        float mx=std::max({R,G,B}), mn=std::min({R,G,B}), d=mx-mn;
        float H=0,S=mx>0?d/mx:0,V=mx;
        if (d>0) {
            if (mx==R) H=std::fmod((G-B)/d,6.f)*60.f;
            else if (mx==G) H=((B-R)/d+2)*60.f;
            else H=((R-G)/d+4)*60.f;
            if (H<0) H+=360.f;
        }
        out.at(r,c,0)=H/360.f; out.at(r,c,1)=S; out.at(r,c,2)=V;
    }
    return out;
}

Image hsv2rgb(const Image& img) {
    Image out(img.rows, img.cols, 3);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        float H=img.at(r,c,0)*360.f,S=img.at(r,c,1),V=img.at(r,c,2);
        float C=V*S, X=C*(1-std::abs(std::fmod(H/60.f,2.f)-1)), m=V-C;
        float R=0,G=0,B=0;
        int hi=(int)(H/60)%6;
        if (hi==0){R=C;G=X;}else if (hi==1){R=X;G=C;}
        else if (hi==2){G=C;B=X;}else if (hi==3){G=X;B=C;}
        else if (hi==4){R=X;B=C;}else{R=C;B=X;}
        out.at(r,c,0)=R+m; out.at(r,c,1)=G+m; out.at(r,c,2)=B+m;
    }
    return out;
}

// ========================== Geometric ==========================

Image imresize(const Image& img, int nr, int nc) {
    const int channels = img.channels;
    Image out(nr, nc, channels);
    if (img.empty() || nr <= 0 || nc <= 0) {
        return out;
    }

    const int src_rows = img.rows;
    const int src_cols = img.cols;
    const float inv_denom_r = 1.f / (static_cast<float>(nr - 1) + 1e-6f);
    const float inv_denom_c = 1.f / (static_cast<float>(nc - 1) + 1e-6f);
    const float scale_r = static_cast<float>(src_rows - 1) * inv_denom_r;
    const float scale_c = static_cast<float>(src_cols - 1) * inv_denom_c;

    std::vector<int> row_r0(static_cast<std::size_t>(nr));
    std::vector<int> row_r1(static_cast<std::size_t>(nr));
    std::vector<float> row_wr0(static_cast<std::size_t>(nr));
    std::vector<float> row_wr1(static_cast<std::size_t>(nr));
    for (int r = 0; r < nr; ++r) {
        const float src_r = static_cast<float>(r) * scale_r;
        int r0 = static_cast<int>(src_r);
        int r1 = std::min(r0 + 1, src_rows - 1);
        r0 = std::max(0, r0);
        const float dr = src_r - static_cast<float>(r0);
        row_r0[static_cast<std::size_t>(r)] = r0;
        row_r1[static_cast<std::size_t>(r)] = r1;
        row_wr0[static_cast<std::size_t>(r)] = 1.f - dr;
        row_wr1[static_cast<std::size_t>(r)] = dr;
    }

    std::vector<int> col_c0(static_cast<std::size_t>(nc));
    std::vector<int> col_c1(static_cast<std::size_t>(nc));
    std::vector<float> col_wc0(static_cast<std::size_t>(nc));
    std::vector<float> col_wc1(static_cast<std::size_t>(nc));
    for (int c = 0; c < nc; ++c) {
        const float src_c = static_cast<float>(c) * scale_c;
        int c0 = static_cast<int>(src_c);
        int c1 = std::min(c0 + 1, src_cols - 1);
        c0 = std::max(0, c0);
        const float dc = src_c - static_cast<float>(c0);
        col_c0[static_cast<std::size_t>(c)] = c0;
        col_c1[static_cast<std::size_t>(c)] = c1;
        col_wc0[static_cast<std::size_t>(c)] = 1.f - dc;
        col_wc1[static_cast<std::size_t>(c)] = dc;
    }

    const float* src = img.data.data();
    float* dst = out.data.data();
    const std::size_t channel_count = static_cast<std::size_t>(channels);
    const std::size_t src_stride =
        static_cast<std::size_t>(src_cols) * channel_count;

    for (int r = 0; r < nr; ++r) {
        const int r0 = row_r0[static_cast<std::size_t>(r)];
        const int r1 = row_r1[static_cast<std::size_t>(r)];
        const float wr0 = row_wr0[static_cast<std::size_t>(r)];
        const float wr1 = row_wr1[static_cast<std::size_t>(r)];
        const float* src_row0 = src + static_cast<std::size_t>(r0) * src_stride;
        const float* src_row1 = src + static_cast<std::size_t>(r1) * src_stride;

        for (int c = 0; c < nc; ++c) {
            const int c0 = col_c0[static_cast<std::size_t>(c)];
            const int c1 = col_c1[static_cast<std::size_t>(c)];
            const float wc0 = col_wc0[static_cast<std::size_t>(c)];
            const float wc1 = col_wc1[static_cast<std::size_t>(c)];
            const float w00 = wr0 * wc0;
            const float w01 = wr0 * wc1;
            const float w10 = wr1 * wc0;
            const float w11 = wr1 * wc1;

            const std::size_t off00 = static_cast<std::size_t>(c0) * channel_count;
            const std::size_t off01 = static_cast<std::size_t>(c1) * channel_count;
            // In `int` this is `(r * nc + c) * channels`, which wraps negative once the
            // output passes INT_MAX elements -- 46341 x 46341 single-channel is already
            // past it, and that is an image this type can legitimately hold. An
            // overflowed index is a store before the buffer, not a wrong pixel.
            float* dst_px = dst + (static_cast<std::size_t>(r) * static_cast<std::size_t>(nc) +
                                   static_cast<std::size_t>(c)) * channel_count;
            for (int ch = 0; ch < channels; ++ch) {
                dst_px[ch] = w00 * src_row0[off00 + ch] + w01 * src_row0[off01 + ch]
                           + w10 * src_row1[off00 + ch] + w11 * src_row1[off01 + ch];
            }
        }
    }
    return out;
}

Image imcrop(const Image& img, int r0, int c0, int r1, int c1) {
    r0=std::max(0,r0); c0=std::max(0,c0);
    r1=std::min(img.rows,r1); c1=std::min(img.cols,c1);
    Image out(r1-r0, c1-c0, img.channels);
    for (int r=r0;r<r1;++r) for (int c=c0;c<c1;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r-r0,c-c0,ch)=img.at(r,c,ch);
    return out;
}

Image imflip(const Image& img, bool horizontal) {
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r,c,ch)=horizontal?img.at(r,img.cols-1-c,ch):img.at(img.rows-1-r,c,ch);
    return out;
}

Image imrotate90(const Image& img) {
    Image out(img.cols, img.rows, img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(img.cols-1-c,r,ch)=img.at(r,c,ch);
    return out;
}

/// Two ways this wrote outside its own buffer, and both are settled here rather than
/// left to the caller. A caller can reasonably be asked to keep a request affordable; it
/// cannot be asked to keep the function inside its allocation.
///
///   - **`img.rows + 2*pad` is `int` arithmetic.** At `pad >= (INT_MAX - 2) / 2` it
///     overflows -- undefined behaviour, and in practice negative, which `Image`'s
///     constructor clamps to an EMPTY image. The copy loop below is bounded by `img`'s
///     extents rather than by `out`'s, so it ran anyway and wrote `out.at(r + pad, ...)`
///     -- an index near 2^30 -- into a zero-length vector.
///   - **A negative `pad`** reaches `out.at(r + pad, ...)` with a negative row and
///     column, which `Image::at` converts to `size_t` and reads as an enormous index.
///
/// An input with no padded image to name gets an empty one back, which is what
/// `hough_circles` and the rest of this file do with an argument they cannot honour.
Image impad(const Image& img, int pad, float val) {
    const long long padded_rows = static_cast<long long>(img.rows) + 2LL * pad;
    const long long padded_cols = static_cast<long long>(img.cols) + 2LL * pad;
    constexpr long long kMaxExtent = std::numeric_limits<int>::max();
    if (pad < 0 || padded_rows > kMaxExtent || padded_cols > kMaxExtent) {
        return {};
    }
    Image out(static_cast<int>(padded_rows), static_cast<int>(padded_cols), img.channels,
              val);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r+pad,c+pad,ch)=img.at(r,c,ch);
    return out;
}

// ========================== Filtering ==========================

namespace {

std::vector<float> make_gaussian_kernel_1d(float sigma, int& half) {
    half = std::max(1, static_cast<int>(3 * sigma));
    const int ksize = 2 * half + 1;
    std::vector<float> kernel(static_cast<std::size_t>(ksize));
    float sum = 0.f;
    for (int i = 0; i < ksize; ++i) {
        const float x = static_cast<float>(i - half);
        kernel[static_cast<std::size_t>(i)] = std::exp(-x * x / (2.f * sigma * sigma));
        sum += kernel[static_cast<std::size_t>(i)];
    }
    const float inv_sum = 1.f / sum;
    for (auto& v : kernel) {
        v *= inv_sum;
    }
    return kernel;
}

void conv1d_horizontal_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels,
    const std::vector<float>& kernel, int half) {
    const int ksize = static_cast<int>(kernel.size());
    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            const float* src_row = src + r * cols;
            float* dst_row = dst + r * cols;
            for (int c = 0; c < cols; ++c) {
                float val = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int cc = std::min(std::max(c + d - half, 0), cols - 1);
                    val += kernel[static_cast<std::size_t>(d)] * src_row[cc];
                }
                dst_row[c] = val;
            }
        }
        return;
    }
    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float val = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int cc = std::min(std::max(c + d - half, 0), cols - 1);
                    val += kernel[static_cast<std::size_t>(d)] *
                           src[(r * cols + cc) * channels + ch];
                }
                dst[(r * cols + c) * channels + ch] = val;
            }
        }
    }
}

void box1d_horizontal_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels,
    int half) {
    const int ksize = 2 * half + 1;
    const float inv_ksize = 1.f / static_cast<float>(ksize);
    const int interior_start = half;
    const int interior_end = cols - half;

    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            const float* src_row = src + r * cols;
            float* dst_row = dst + r * cols;

            for (int c = 0; c < std::min(interior_start, cols); ++c) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int cc = std::min(std::max(c + d - half, 0), cols - 1);
                    sum += src_row[cc];
                }
                dst_row[c] = sum * inv_ksize;
            }

            if (interior_start < interior_end) {
                float sum = 0.f;
                for (int d = -half; d <= half; ++d) {
                    sum += src_row[interior_start + d];
                }
                dst_row[interior_start] = sum * inv_ksize;
                for (int c = interior_start + 1; c < interior_end; ++c) {
                    sum += src_row[c + half] - src_row[c - half - 1];
                    dst_row[c] = sum * inv_ksize;
                }
            }

            for (int c = std::max(interior_end, interior_start); c < cols; ++c) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int cc = std::min(std::max(c + d - half, 0), cols - 1);
                    sum += src_row[cc];
                }
                dst_row[c] = sum * inv_ksize;
            }
        }
        return;
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < std::min(interior_start, cols); ++c) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int cc = std::min(std::max(c + d - half, 0), cols - 1);
                    sum += src[(r * cols + cc) * channels + ch];
                }
                dst[(r * cols + c) * channels + ch] = sum * inv_ksize;
            }

            if (interior_start < interior_end) {
                float sum = 0.f;
                for (int d = -half; d <= half; ++d) {
                    sum += src[(r * cols + interior_start + d) * channels + ch];
                }
                dst[(r * cols + interior_start) * channels + ch] = sum * inv_ksize;
                for (int c = interior_start + 1; c < interior_end; ++c) {
                    sum += src[(r * cols + c + half) * channels + ch] -
                           src[(r * cols + c - half - 1) * channels + ch];
                    dst[(r * cols + c) * channels + ch] = sum * inv_ksize;
                }
            }

            for (int c = std::max(interior_end, interior_start); c < cols; ++c) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int cc = std::min(std::max(c + d - half, 0), cols - 1);
                    sum += src[(r * cols + cc) * channels + ch];
                }
                dst[(r * cols + c) * channels + ch] = sum * inv_ksize;
            }
        }
    }
}

void box1d_vertical_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels,
    int half) {
    const int ksize = 2 * half + 1;
    const float inv_ksize = 1.f / static_cast<float>(ksize);
    const int interior_start = half;
    const int interior_end = rows - half;

    if (channels == 1) {
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < std::min(interior_start, rows); ++r) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int rr = std::min(std::max(r + d - half, 0), rows - 1);
                    sum += src[rr * cols + c];
                }
                dst[r * cols + c] = sum * inv_ksize;
            }

            if (interior_start < interior_end) {
                float sum = 0.f;
                for (int d = -half; d <= half; ++d) {
                    sum += src[(interior_start + d) * cols + c];
                }
                dst[interior_start * cols + c] = sum * inv_ksize;
                for (int r = interior_start + 1; r < interior_end; ++r) {
                    sum += src[(r + half) * cols + c] - src[(r - half - 1) * cols + c];
                    dst[r * cols + c] = sum * inv_ksize;
                }
            }

            for (int r = std::max(interior_end, interior_start); r < rows; ++r) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int rr = std::min(std::max(r + d - half, 0), rows - 1);
                    sum += src[rr * cols + c];
                }
                dst[r * cols + c] = sum * inv_ksize;
            }
        }
        return;
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < std::min(interior_start, rows); ++r) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int rr = std::min(std::max(r + d - half, 0), rows - 1);
                    sum += src[(rr * cols + c) * channels + ch];
                }
                dst[(r * cols + c) * channels + ch] = sum * inv_ksize;
            }

            if (interior_start < interior_end) {
                float sum = 0.f;
                for (int d = -half; d <= half; ++d) {
                    sum += src[((interior_start + d) * cols + c) * channels + ch];
                }
                dst[(interior_start * cols + c) * channels + ch] = sum * inv_ksize;
                for (int r = interior_start + 1; r < interior_end; ++r) {
                    sum += src[((r + half) * cols + c) * channels + ch] -
                           src[((r - half - 1) * cols + c) * channels + ch];
                    dst[(r * cols + c) * channels + ch] = sum * inv_ksize;
                }
            }

            for (int r = std::max(interior_end, interior_start); r < rows; ++r) {
                float sum = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int rr = std::min(std::max(r + d - half, 0), rows - 1);
                    sum += src[(rr * cols + c) * channels + ch];
                }
                dst[(r * cols + c) * channels + ch] = sum * inv_ksize;
            }
        }
    }
}

void filter3x3_laplacian_replicate(
    const float* src, float* dst,
    int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const int rm = std::min(std::max(r - 1, 0), rows - 1);
        const int rp = std::min(std::max(r + 1, 0), rows - 1);
        for (int c = 0; c < cols; ++c) {
            const int cm = std::min(std::max(c - 1, 0), cols - 1);
            const int cp = std::min(std::max(c + 1, 0), cols - 1);
            const float center = src[r * cols + c];
            dst[r * cols + c] = src[rm * cols + c] + src[rp * cols + c]
                              + src[r * cols + cm] + src[r * cols + cp]
                              - 4.f * center;
        }
    }
}

Image laplacian3x3(const Image& g) {
    Image out(g.rows, g.cols, 1);
    if (g.empty()) {
        return out;
    }
    filter3x3_laplacian_replicate(g.data.data(), out.data.data(), g.rows, g.cols);
    return out;
}

void filter3x3_sharpen_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels) {
    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            const int rm = std::min(std::max(r - 1, 0), rows - 1);
            const int rp = std::min(std::max(r + 1, 0), rows - 1);
            for (int c = 0; c < cols; ++c) {
                const int cm = std::min(std::max(c - 1, 0), cols - 1);
                const int cp = std::min(std::max(c + 1, 0), cols - 1);
                const float center = src[r * cols + c];
                dst[r * cols + c] = 5.f * center
                                    - src[rm * cols + c]
                                    - src[rp * cols + c]
                                    - src[r * cols + cm]
                                    - src[r * cols + cp];
            }
        }
        return;
    }
    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            const int rm = std::min(std::max(r - 1, 0), rows - 1);
            const int rp = std::min(std::max(r + 1, 0), rows - 1);
            for (int c = 0; c < cols; ++c) {
                const int cm = std::min(std::max(c - 1, 0), cols - 1);
                const int cp = std::min(std::max(c + 1, 0), cols - 1);
                const float center = src[(r * cols + c) * channels + ch];
                dst[(r * cols + c) * channels + ch] = 5.f * center
                    - src[(rm * cols + c) * channels + ch]
                    - src[(rp * cols + c) * channels + ch]
                    - src[(r * cols + cm) * channels + ch]
                    - src[(r * cols + cp) * channels + ch];
            }
        }
    }
}

void conv1d_vertical_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels,
    const std::vector<float>& kernel, int half) {
    const int ksize = static_cast<int>(kernel.size());
    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float val = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int rr = std::min(std::max(r + d - half, 0), rows - 1);
                    val += kernel[static_cast<std::size_t>(d)] * src[rr * cols + c];
                }
                dst[r * cols + c] = val;
            }
        }
        return;
    }
    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float val = 0.f;
                for (int d = 0; d < ksize; ++d) {
                    const int rr = std::min(std::max(r + d - half, 0), rows - 1);
                    val += kernel[static_cast<std::size_t>(d)] *
                           src[(rr * cols + c) * channels + ch];
                }
                dst[(r * cols + c) * channels + ch] = val;
            }
        }
    }
}

bool try_separable_kernel(
    const std::vector<std::vector<float>>& K,
    std::vector<float>& row_k,
    std::vector<float>& col_k) {
    const int kr = static_cast<int>(K.size());
    if (kr == 0) {
        return false;
    }
    const int kc = static_cast<int>(K[0].size());
    if (kc == 0) {
        return false;
    }

    if (kr == 1) {
        col_k = K[0];
        row_k = {1.f};
        return true;
    }
    if (kc == 1) {
        row_k.resize(static_cast<std::size_t>(kr));
        for (int i = 0; i < kr; ++i) {
            row_k[static_cast<std::size_t>(i)] = K[static_cast<std::size_t>(i)][0];
        }
        col_k = {1.f};
        return true;
    }

    int i0 = 0;
    int j0 = 0;
    float pivot = K[static_cast<std::size_t>(i0)][static_cast<std::size_t>(j0)];
    if (std::abs(pivot) < 1e-15f) {
        bool found = false;
        for (int i = 0; i < kr && !found; ++i) {
            for (int j = 0; j < kc; ++j) {
                if (std::abs(K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) >= 1e-15f) {
                    i0 = i;
                    j0 = j;
                    pivot = K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            row_k.assign(static_cast<std::size_t>(kr), 0.f);
            col_k.assign(static_cast<std::size_t>(kc), 0.f);
            return true;
        }
    }

    col_k.resize(static_cast<std::size_t>(kc));
    row_k.resize(static_cast<std::size_t>(kr));
    for (int j = 0; j < kc; ++j) {
        col_k[static_cast<std::size_t>(j)] = K[static_cast<std::size_t>(i0)][static_cast<std::size_t>(j)];
    }
    const float col_pivot = col_k[static_cast<std::size_t>(j0)];
    for (int i = 0; i < kr; ++i) {
        row_k[static_cast<std::size_t>(i)] =
            K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j0)] / col_pivot;
    }

    for (int i = 0; i < kr; ++i) {
        for (int j = 0; j < kc; ++j) {
            const float expected = row_k[static_cast<std::size_t>(i)] * col_k[static_cast<std::size_t>(j)];
            const float actual = K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            if (std::abs(actual - expected) > 1e-6f * (1.f + std::abs(actual))) {
                return false;
            }
        }
    }
    return true;
}

void imfilter_3x3_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels,
    const float k[9]) {
    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            const int sr0 = std::min(std::max(r - 1, 0), rows - 1);
            const int sr1 = r;
            const int sr2 = std::min(r + 1, rows - 1);
            const float* row0 = src + sr0 * cols;
            const float* row1 = src + sr1 * cols;
            const float* row2 = src + sr2 * cols;
            float* dst_row = dst + r * cols;

            for (int c = 0; c < cols; ++c) {
                const int sc0 = std::min(std::max(c - 1, 0), cols - 1);
                const int sc1 = c;
                const int sc2 = std::min(c + 1, cols - 1);
                dst_row[c] = k[0] * row0[sc0] + k[1] * row0[sc1] + k[2] * row0[sc2]
                           + k[3] * row1[sc0] + k[4] * row1[sc1] + k[5] * row1[sc2]
                           + k[6] * row2[sc0] + k[7] * row2[sc1] + k[8] * row2[sc2];
            }
        }
        return;
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            const int sr0 = std::min(std::max(r - 1, 0), rows - 1);
            const int sr1 = r;
            const int sr2 = std::min(r + 1, rows - 1);

            for (int c = 0; c < cols; ++c) {
                const int sc0 = std::min(std::max(c - 1, 0), cols - 1);
                const int sc1 = c;
                const int sc2 = std::min(c + 1, cols - 1);
                float val = 0.f;
                val += k[0] * src[(sr0 * cols + sc0) * channels + ch];
                val += k[1] * src[(sr0 * cols + sc1) * channels + ch];
                val += k[2] * src[(sr0 * cols + sc2) * channels + ch];
                val += k[3] * src[(sr1 * cols + sc0) * channels + ch];
                val += k[4] * src[(sr1 * cols + sc1) * channels + ch];
                val += k[5] * src[(sr1 * cols + sc2) * channels + ch];
                val += k[6] * src[(sr2 * cols + sc0) * channels + ch];
                val += k[7] * src[(sr2 * cols + sc1) * channels + ch];
                val += k[8] * src[(sr2 * cols + sc2) * channels + ch];
                dst[(r * cols + c) * channels + ch] = val;
            }
        }
    }
}

void imfilter_general_replicate(
    const float* src, float* dst,
    int rows, int cols, int channels,
    const float* kernel, int kr, int kc) {
    const int pr = kr / 2;
    const int pc = kc / 2;

    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float val = 0.f;
                for (int dr = 0; dr < kr; ++dr) {
                    const int sr = std::min(std::max(r + dr - pr, 0), rows - 1);
                    const float* src_row = src + sr * cols;
                    const float* k_row = kernel + dr * kc;
                    for (int dc = 0; dc < kc; ++dc) {
                        const int sc = std::min(std::max(c + dc - pc, 0), cols - 1);
                        val += k_row[dc] * src_row[sc];
                    }
                }
                dst[r * cols + c] = val;
            }
        }
        return;
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float val = 0.f;
                for (int dr = 0; dr < kr; ++dr) {
                    const int sr = std::min(std::max(r + dr - pr, 0), rows - 1);
                    for (int dc = 0; dc < kc; ++dc) {
                        const int sc = std::min(std::max(c + dc - pc, 0), cols - 1);
                        val += kernel[dr * kc + dc] *
                               src[(sr * cols + sc) * channels + ch];
                    }
                }
                dst[(r * cols + c) * channels + ch] = val;
            }
        }
    }
}

Image imfilter_impl(const Image& img, const std::vector<std::vector<float>>& K) {
    if (img.empty() || K.empty()) {
        return img;
    }

    const int kr = static_cast<int>(K.size());
    const int kc = static_cast<int>(K[0].size());
    const int rows = img.rows;
    const int cols = img.cols;
    const int channels = img.channels;

    std::vector<float> row_k;
    std::vector<float> col_k;
    if (try_separable_kernel(K, row_k, col_k)) {
        const int half_r = kr / 2;
        const int half_c = kc / 2;
        const std::size_t npix = static_cast<std::size_t>(rows) * cols * channels;

        thread_local std::vector<float> tmp;
        if (tmp.size() < npix) {
            tmp.resize(npix);
        }

        Image out(rows, cols, channels);
        const float* src = img.data.data();
        conv1d_horizontal_replicate(src, tmp.data(), rows, cols, channels, col_k, half_c);
        conv1d_vertical_replicate(tmp.data(), out.data.data(), rows, cols, channels, row_k, half_r);
        return out;
    }

    Image out(rows, cols, channels);
    const float* src = img.data.data();
    float* dst = out.data.data();

    if (kr == 3 && kc == 3) {
        const float k3[9] = {
            K[0][0], K[0][1], K[0][2],
            K[1][0], K[1][1], K[1][2],
            K[2][0], K[2][1], K[2][2],
        };
        imfilter_3x3_replicate(src, dst, rows, cols, channels, k3);
        return out;
    }

    std::vector<float> kernel_flat(static_cast<std::size_t>(kr * kc));
    for (int dr = 0; dr < kr; ++dr) {
        for (int dc = 0; dc < kc; ++dc) {
            kernel_flat[static_cast<std::size_t>(dr * kc + dc)] =
                K[static_cast<std::size_t>(dr)][static_cast<std::size_t>(dc)];
        }
    }
    imfilter_general_replicate(src, dst, rows, cols, channels, kernel_flat.data(), kr, kc);
    return out;
}

} // namespace

Image imfilter(const Image& img, const std::vector<std::vector<float>>& K) {
    return imfilter_impl(img, K);
}

Image imgaussfilt(const Image& img, float sigma) {
    if (img.empty()) {
        return img;
    }

    int half = 0;
    const std::vector<float> kernel = make_gaussian_kernel_1d(sigma, half);

    const int rows = img.rows;
    const int cols = img.cols;
    const int channels = img.channels;
    const std::size_t npix = static_cast<std::size_t>(rows) * cols * channels;

    thread_local std::vector<float> tmp;
    if (tmp.size() < npix) {
        tmp.resize(npix);
    }

    Image out(rows, cols, channels);
    const float* src = img.data.data();
    conv1d_horizontal_replicate(src, tmp.data(), rows, cols, channels, kernel, half);
    conv1d_vertical_replicate(tmp.data(), out.data.data(), rows, cols, channels, kernel, half);
    return out;
}

namespace {

inline void median_compare_swap(float& a, float& b) {
    if (a > b) {
        std::swap(a, b);
    }
}

#define MEDFILT_PIX_SORT(a, b) median_compare_swap((a), (b))

// Devillard opt_med9: fixed 19-comparator network, median at index 4.
inline float median_of_nine(float* p) {
    MEDFILT_PIX_SORT(p[1], p[2]);
    MEDFILT_PIX_SORT(p[4], p[5]);
    MEDFILT_PIX_SORT(p[7], p[8]);
    MEDFILT_PIX_SORT(p[0], p[1]);
    MEDFILT_PIX_SORT(p[3], p[4]);
    MEDFILT_PIX_SORT(p[6], p[7]);
    MEDFILT_PIX_SORT(p[1], p[2]);
    MEDFILT_PIX_SORT(p[4], p[5]);
    MEDFILT_PIX_SORT(p[7], p[8]);
    MEDFILT_PIX_SORT(p[0], p[3]);
    MEDFILT_PIX_SORT(p[5], p[8]);
    MEDFILT_PIX_SORT(p[4], p[7]);
    MEDFILT_PIX_SORT(p[3], p[6]);
    MEDFILT_PIX_SORT(p[1], p[4]);
    MEDFILT_PIX_SORT(p[2], p[5]);
    MEDFILT_PIX_SORT(p[4], p[7]);
    MEDFILT_PIX_SORT(p[4], p[2]);
    MEDFILT_PIX_SORT(p[6], p[4]);
    MEDFILT_PIX_SORT(p[4], p[2]);
    return p[4];
}

// Devillard opt_med25: fixed comparator network, median at index 12.
inline float median_of_twentyfive(float* p) {
    MEDFILT_PIX_SORT(p[0], p[1]);
    MEDFILT_PIX_SORT(p[3], p[4]);
    MEDFILT_PIX_SORT(p[2], p[4]);
    MEDFILT_PIX_SORT(p[2], p[3]);
    MEDFILT_PIX_SORT(p[6], p[7]);
    MEDFILT_PIX_SORT(p[5], p[7]);
    MEDFILT_PIX_SORT(p[5], p[6]);
    MEDFILT_PIX_SORT(p[9], p[10]);
    MEDFILT_PIX_SORT(p[8], p[10]);
    MEDFILT_PIX_SORT(p[8], p[9]);
    MEDFILT_PIX_SORT(p[12], p[13]);
    MEDFILT_PIX_SORT(p[11], p[13]);
    MEDFILT_PIX_SORT(p[11], p[12]);
    MEDFILT_PIX_SORT(p[15], p[16]);
    MEDFILT_PIX_SORT(p[14], p[16]);
    MEDFILT_PIX_SORT(p[14], p[15]);
    MEDFILT_PIX_SORT(p[18], p[19]);
    MEDFILT_PIX_SORT(p[17], p[19]);
    MEDFILT_PIX_SORT(p[17], p[18]);
    MEDFILT_PIX_SORT(p[21], p[22]);
    MEDFILT_PIX_SORT(p[20], p[22]);
    MEDFILT_PIX_SORT(p[20], p[21]);
    MEDFILT_PIX_SORT(p[23], p[24]);
    MEDFILT_PIX_SORT(p[2], p[5]);
    MEDFILT_PIX_SORT(p[3], p[6]);
    MEDFILT_PIX_SORT(p[0], p[6]);
    MEDFILT_PIX_SORT(p[0], p[3]);
    MEDFILT_PIX_SORT(p[4], p[7]);
    MEDFILT_PIX_SORT(p[1], p[7]);
    MEDFILT_PIX_SORT(p[1], p[4]);
    MEDFILT_PIX_SORT(p[11], p[14]);
    MEDFILT_PIX_SORT(p[8], p[14]);
    MEDFILT_PIX_SORT(p[8], p[11]);
    MEDFILT_PIX_SORT(p[12], p[15]);
    MEDFILT_PIX_SORT(p[9], p[15]);
    MEDFILT_PIX_SORT(p[9], p[12]);
    MEDFILT_PIX_SORT(p[13], p[16]);
    MEDFILT_PIX_SORT(p[10], p[16]);
    MEDFILT_PIX_SORT(p[10], p[13]);
    MEDFILT_PIX_SORT(p[20], p[23]);
    MEDFILT_PIX_SORT(p[17], p[23]);
    MEDFILT_PIX_SORT(p[17], p[20]);
    MEDFILT_PIX_SORT(p[21], p[24]);
    MEDFILT_PIX_SORT(p[18], p[24]);
    MEDFILT_PIX_SORT(p[18], p[21]);
    MEDFILT_PIX_SORT(p[19], p[22]);
    MEDFILT_PIX_SORT(p[8], p[17]);
    MEDFILT_PIX_SORT(p[9], p[18]);
    MEDFILT_PIX_SORT(p[0], p[18]);
    MEDFILT_PIX_SORT(p[0], p[9]);
    MEDFILT_PIX_SORT(p[10], p[19]);
    MEDFILT_PIX_SORT(p[1], p[19]);
    MEDFILT_PIX_SORT(p[1], p[10]);
    MEDFILT_PIX_SORT(p[11], p[20]);
    MEDFILT_PIX_SORT(p[2], p[20]);
    MEDFILT_PIX_SORT(p[2], p[11]);
    MEDFILT_PIX_SORT(p[12], p[21]);
    MEDFILT_PIX_SORT(p[3], p[21]);
    MEDFILT_PIX_SORT(p[3], p[12]);
    MEDFILT_PIX_SORT(p[13], p[22]);
    MEDFILT_PIX_SORT(p[4], p[22]);
    MEDFILT_PIX_SORT(p[4], p[13]);
    MEDFILT_PIX_SORT(p[14], p[23]);
    MEDFILT_PIX_SORT(p[5], p[23]);
    MEDFILT_PIX_SORT(p[5], p[14]);
    MEDFILT_PIX_SORT(p[15], p[24]);
    MEDFILT_PIX_SORT(p[6], p[24]);
    MEDFILT_PIX_SORT(p[6], p[15]);
    MEDFILT_PIX_SORT(p[7], p[16]);
    MEDFILT_PIX_SORT(p[7], p[19]);
    MEDFILT_PIX_SORT(p[13], p[21]);
    MEDFILT_PIX_SORT(p[15], p[23]);
    MEDFILT_PIX_SORT(p[7], p[13]);
    MEDFILT_PIX_SORT(p[7], p[15]);
    MEDFILT_PIX_SORT(p[1], p[9]);
    MEDFILT_PIX_SORT(p[3], p[11]);
    MEDFILT_PIX_SORT(p[5], p[17]);
    MEDFILT_PIX_SORT(p[11], p[17]);
    MEDFILT_PIX_SORT(p[9], p[17]);
    MEDFILT_PIX_SORT(p[4], p[10]);
    MEDFILT_PIX_SORT(p[6], p[12]);
    MEDFILT_PIX_SORT(p[7], p[14]);
    MEDFILT_PIX_SORT(p[4], p[6]);
    MEDFILT_PIX_SORT(p[4], p[7]);
    MEDFILT_PIX_SORT(p[12], p[14]);
    MEDFILT_PIX_SORT(p[10], p[14]);
    MEDFILT_PIX_SORT(p[6], p[7]);
    MEDFILT_PIX_SORT(p[10], p[12]);
    MEDFILT_PIX_SORT(p[6], p[10]);
    MEDFILT_PIX_SORT(p[6], p[17]);
    MEDFILT_PIX_SORT(p[12], p[17]);
    MEDFILT_PIX_SORT(p[7], p[17]);
    MEDFILT_PIX_SORT(p[7], p[10]);
    MEDFILT_PIX_SORT(p[12], p[18]);
    MEDFILT_PIX_SORT(p[7], p[12]);
    MEDFILT_PIX_SORT(p[10], p[18]);
    MEDFILT_PIX_SORT(p[12], p[20]);
    MEDFILT_PIX_SORT(p[10], p[20]);
    MEDFILT_PIX_SORT(p[10], p[12]);
    return p[12];
}

#undef MEDFILT_PIX_SORT

inline float sample_replicate(const Image& img, int r, int c, int ch) {
    const int sr = std::min(std::max(r, 0), img.rows - 1);
    const int sc = std::min(std::max(c, 0), img.cols - 1);
    return img.at(sr, sc, ch);
}

void medfilt2_channel_k3(const Image& img, Image& out, int ch) {
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            float window[9];
            int idx = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    window[idx++] = sample_replicate(img, r + dr, c + dc, ch);
                }
            }
            out.at(r, c, ch) = median_of_nine(window);
        }
    }
}

void medfilt2_channel_k5(const Image& img, Image& out, int ch) {
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            float window[25];
            int idx = 0;
            for (int dr = -2; dr <= 2; ++dr) {
                for (int dc = -2; dc <= 2; ++dc) {
                    window[idx++] = sample_replicate(img, r + dr, c + dc, ch);
                }
            }
            out.at(r, c, ch) = median_of_twentyfive(window);
        }
    }
}

void medfilt2_channel_large(const Image& img, Image& out, int ch, int ksize) {
    const int half = ksize / 2;
    const std::size_t win_area = static_cast<std::size_t>(ksize) * static_cast<std::size_t>(ksize);
    const std::size_t mid = win_area / 2;

    thread_local std::vector<float> window;
    if (window.size() < win_area) {
        window.resize(win_area);
    }

    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            std::size_t idx = 0;
            for (int dr = -half; dr <= half; ++dr) {
                for (int dc = -half; dc <= half; ++dc) {
                    window[idx++] = sample_replicate(img, r + dr, c + dc, ch);
                }
            }
            std::nth_element(window.begin(), window.begin() + static_cast<std::ptrdiff_t>(mid),
                             window.begin() + static_cast<std::ptrdiff_t>(win_area));
            out.at(r, c, ch) = window[mid];
        }
    }
}

} // namespace

Image medfilt2(const Image& img, int ksize) {
    if (img.empty()) {
        return img;
    }

    Image out(img.rows, img.cols, img.channels);
    if (ksize <= 1) {
        out.data = img.data;
        return out;
    }

    for (int ch = 0; ch < img.channels; ++ch) {
        if (ksize == 3) {
            medfilt2_channel_k3(img, out, ch);
        } else if (ksize == 5) {
            medfilt2_channel_k5(img, out, ch);
        } else {
            medfilt2_channel_large(img, out, ch, ksize);
        }
    }
    return out;
}

Image bilateral(const Image& img, float sigma_s, float sigma_r) {
    if (img.empty()) {
        return img;
    }

    const int rows = img.rows;
    const int cols = img.cols;
    const int channels = img.channels;
    // The radius follows sigma_s, which is a caller's float and so is unbounded. Two
    // things had to change.
    //
    // A radius past the image is not a wider filter, it is the same filter with the
    // extra taps falling outside every pixel's neighbourhood, so clamping it to the
    // image changes no result and bounds the work. Without the clamp, sigma_s = 1e6
    // asks for a 4000001 x 4000001 kernel.
    //
    // And `ksize * ksize` was `int` arithmetic. At that size the product is 1.6e13,
    // which is not an `int`; the overflow is undefined, and what it did was size the
    // weight buffer from whatever the wrap produced and then index it with
    // `dri * ksize + dci`, overflowing again -- a write outside the buffer rather than
    // a slow filter. The clamp makes the overflow unreachable and the `size_t`
    // arithmetic makes it impossible, and both are worth having: the first is a policy
    // about kernels and the second is about the type.
    const int max_half = std::max(1, std::max(rows, cols));
    const int half = std::min(max_half, std::max(1, static_cast<int>(2.f * sigma_s)));
    const int ksize = 2 * half + 1;

    const float inv_2_sigma_s_sq = 1.f / (2.f * sigma_s * sigma_s);
    const std::size_t kernel_area =
        static_cast<std::size_t>(ksize) * static_cast<std::size_t>(ksize);
    std::vector<float> spatial_weights(kernel_area);
    for (int dri = 0; dri < ksize; ++dri) {
        const int dr = dri - half;
        for (int dci = 0; dci < ksize; ++dci) {
            const int dc = dci - half;
            spatial_weights[static_cast<std::size_t>(dri) *
                                static_cast<std::size_t>(ksize) +
                            static_cast<std::size_t>(dci)] =
                std::exp(-static_cast<float>(dr * dr + dc * dc) * inv_2_sigma_s_sq);
        }
    }

    const float inv_2_sigma_r_sq = 1.f / (2.f * sigma_r * sigma_r);
    constexpr float k_min_range_weight = 1e-6f;

    Image out(rows, cols, channels);
    const float* src = img.data.data();
    float* dst = out.data.data();

    if (channels == 1) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const float ic = src[r * cols + c];
                float num = 0.f;
                float den = 0.f;
                for (int dri = 0; dri < ksize; ++dri) {
                    const int dr = dri - half;
                    const int sr = std::min(std::max(r + dr, 0), rows - 1);
                    const float* src_row = src + sr * cols;
                    const float* sp_row =
                        spatial_weights.data() + static_cast<std::size_t>(dri * ksize);
                    for (int dci = 0; dci < ksize; ++dci) {
                        const int dc = dci - half;
                        const int sc = std::min(std::max(c + dc, 0), cols - 1);
                        const float sp = sp_row[dci];
                        const float neighbor = src_row[sc];
                        const float diff = neighbor - ic;
                        const float rp = std::exp(-diff * diff * inv_2_sigma_r_sq);
                        if (rp < k_min_range_weight) {
                            continue;
                        }
                        const float w = sp * rp;
                        num += w * neighbor;
                        den += w;
                    }
                }
                dst[r * cols + c] = num / (den + 1e-12f);
            }
        }
        return out;
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const float ic = src[(r * cols + c) * channels + ch];
                float num = 0.f;
                float den = 0.f;
                for (int dri = 0; dri < ksize; ++dri) {
                    const int dr = dri - half;
                    const int sr = std::min(std::max(r + dr, 0), rows - 1);
                    const int s_row_base = sr * cols * channels;
                    const float* sp_row =
                        spatial_weights.data() + static_cast<std::size_t>(dri * ksize);
                    for (int dci = 0; dci < ksize; ++dci) {
                        const int dc = dci - half;
                        const int sc = std::min(std::max(c + dc, 0), cols - 1);
                        const float sp = sp_row[dci];
                        const float neighbor = src[s_row_base + sc * channels + ch];
                        const float diff = neighbor - ic;
                        const float rp = std::exp(-diff * diff * inv_2_sigma_r_sq);
                        if (rp < k_min_range_weight) {
                            continue;
                        }
                        const float w = sp * rp;
                        num += w * neighbor;
                        den += w;
                    }
                }
                dst[(r * cols + c) * channels + ch] = num / (den + 1e-12f);
            }
        }
    }
    return out;
}

Image boxfilter(const Image& img, int ksize) {
    if (img.empty()) {
        return img;
    }

    const int half = ksize / 2;
    const int rows = img.rows;
    const int cols = img.cols;
    const int channels = img.channels;
    const std::size_t npix = static_cast<std::size_t>(rows) * cols * channels;

    thread_local std::vector<float> tmp;
    if (tmp.size() < npix) {
        tmp.resize(npix);
    }

    Image out(rows, cols, channels);
    const float* src = img.data.data();
    box1d_horizontal_replicate(src, tmp.data(), rows, cols, channels, half);
    box1d_vertical_replicate(tmp.data(), out.data.data(), rows, cols, channels, half);
    return out;
}

Image sharpen(const Image& img) {
    if (img.empty()) {
        return img;
    }
    Image out(img.rows, img.cols, img.channels);
    filter3x3_sharpen_replicate(
        img.data.data(), out.data.data(),
        img.rows, img.cols, img.channels);
    return out;
}

// ========================== Edge Detection ==========================

static std::pair<Image,Image> sobel_xy(const Image& g) {
    std::vector<std::vector<float>> Kx={{-1,0,1},{-2,0,2},{-1,0,1}};
    std::vector<std::vector<float>> Ky={{-1,-2,-1},{0,0,0},{1,2,1}};
    return {imfilter(g,Kx), imfilter(g,Ky)};
}

Image sobel_x(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    return sobel_xy(g).first;
}
Image sobel_y(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    return sobel_xy(g).second;
}
Image sobel(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    auto [gx,gy]=sobel_xy(g);
    Image out(img.rows,img.cols,1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c)
        out.at(r,c,0)=std::sqrt(gx.at(r,c,0)*gx.at(r,c,0)+gy.at(r,c,0)*gy.at(r,c,0));
    return out;
}
Image prewitt(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    std::vector<std::vector<float>> Kx={{-1,0,1},{-1,0,1},{-1,0,1}};
    std::vector<std::vector<float>> Ky={{-1,-1,-1},{0,0,0},{1,1,1}};
    auto gx=imfilter(g,Kx), gy=imfilter(g,Ky);
    Image out(img.rows,img.cols,1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c)
        out.at(r,c,0)=std::sqrt(gx.at(r,c,0)*gx.at(r,c,0)+gy.at(r,c,0)*gy.at(r,c,0));
    return out;
}
Image laplacian(const Image& img) {
    const Image g = img.channels > 1 ? rgb2gray(img) : img;
    return laplacian3x3(g);
}
Image scharr(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    std::vector<std::vector<float>> Kx={{3,0,-3},{10,0,-10},{3,0,-3}};
    std::vector<std::vector<float>> Ky={{3,10,3},{0,0,0},{-3,-10,-3}};
    auto gx=imfilter(g,Kx), gy=imfilter(g,Ky);
    Image out(img.rows,img.cols,1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c)
        out.at(r,c,0)=std::sqrt(gx.at(r,c,0)*gx.at(r,c,0)+gy.at(r,c,0)*gy.at(r,c,0));
    return out;
}

Image roberts(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    Image gx(g.rows,g.cols,1,0.f), gy(g.rows,g.cols,1,0.f);
    for (int r=0;r<g.rows;++r) for (int c=0;c<g.cols;++c) {
        int r1=std::min(r+1,g.rows-1), c1=std::min(c+1,g.cols-1);
        gx.at(r,c,0)=g.at(r,c,0)-g.at(r1,c1,0);
        gy.at(r,c,0)=g.at(r,c1,0)-g.at(r1,c,0);
    }
    Image out(g.rows,g.cols,1);
    for (int r=0;r<g.rows;++r) for (int c=0;c<g.cols;++c)
        out.at(r,c,0)=std::sqrt(gx.at(r,c,0)*gx.at(r,c,0)+gy.at(r,c,0)*gy.at(r,c,0));
    return out;
}

// LoG edge/blob detector: separable Gaussian blur then fast 3x3 Laplacian.
Image laplacian_of_gaussian(const Image& img, float sigma) {
    const Image g = img.channels > 1 ? rgb2gray(img) : img;
    if (g.empty()) {
        return g;
    }
    return laplacian3x3(imgaussfilt(g, sigma));
}

Image canny(const Image& img, float low, float high, float sigma) {
    auto g=img.channels>1?rgb2gray(img):img;
    // 1. Gaussian smooth
    auto sm=imgaussfilt(g,sigma);
    // 2. Sobel gradients
    auto [gx,gy]=sobel_xy(sm);
    // 3. Non-maximum suppression
    Image nms(img.rows,img.cols,1,0.f);
    for (int r=1;r<img.rows-1;++r) for (int c=1;c<img.cols-1;++c) {
        float angle=std::atan2(gy.at(r,c,0),gx.at(r,c,0))*(float)(180/M_PI);
        if (angle<0) angle+=180;
        float mag=std::sqrt(gx.at(r,c,0)*gx.at(r,c,0)+gy.at(r,c,0)*gy.at(r,c,0));
        float p1=0,p2=0;
        if (angle<22.5||angle>=157.5){p1=sm.at(r,c-1,0);p2=sm.at(r,c+1,0);}
        else if (angle<67.5){p1=sm.at(r-1,c+1,0);p2=sm.at(r+1,c-1,0);}
        else if (angle<112.5){p1=sm.at(r-1,c,0);p2=sm.at(r+1,c,0);}
        else{p1=sm.at(r-1,c-1,0);p2=sm.at(r+1,c+1,0);}
        nms.at(r,c,0)=(mag>=p1&&mag>=p2)?mag:0.f;
    }
    // 4. Double threshold
    Image out(img.rows,img.cols,1,0.f);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        float v=nms.at(r,c,0);
        out.at(r,c,0)=v>=high?1.f:(v>=low?0.5f:0.f);
    }
    // 5. Edge tracking (hysteresis)
    for (int r=1;r<img.rows-1;++r) for (int c=1;c<img.cols-1;++c) {
        if (out.at(r,c,0)==0.5f) {
            bool strong=false;
            for (int dr=-1;dr<=1&&!strong;++dr) for (int dc=-1;dc<=1&&!strong;++dc)
                if (out.at(r+dr,c+dc,0)==1.f) strong=true;
            out.at(r,c,0)=strong?1.f:0.f;
        }
    }
    return out;
}

// ========================== Morphology ==========================

Image imdilate(const Image& img, int ksize) {
    int h=ksize/2;
    Image out(img.rows,img.cols,img.channels,0.f);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float mx=-1e30f;
        for (int dr=-h;dr<=h;++dr) for (int dc=-h;dc<=h;++dc) {
            int sr=std::min(std::max(r+dr,0),img.rows-1);
            int sc=std::min(std::max(c+dc,0),img.cols-1);
            mx=std::max(mx,img.at(sr,sc,ch));
        }
        out.at(r,c,ch)=mx;
    }
    return out;
}

Image imerode(const Image& img, int ksize) {
    int h=ksize/2;
    Image out(img.rows,img.cols,img.channels,1.f);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float mn=1e30f;
        for (int dr=-h;dr<=h;++dr) for (int dc=-h;dc<=h;++dc) {
            int sr=std::min(std::max(r+dr,0),img.rows-1);
            int sc=std::min(std::max(c+dc,0),img.cols-1);
            mn=std::min(mn,img.at(sr,sc,ch));
        }
        out.at(r,c,ch)=mn;
    }
    return out;
}

Image imopen(const Image& img, int ksize) { return imdilate(imerode(img,ksize),ksize); }
Image imclose(const Image& img, int ksize) { return imerode(imdilate(img,ksize),ksize); }
Image imtophat(const Image& img, int ksize) {
    auto opened=imopen(img,ksize);
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r,c,ch)=img.at(r,c,ch)-opened.at(r,c,ch);
    return out;
}
Image imbothat(const Image& img, int ksize) {
    auto closed=imclose(img,ksize);
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r,c,ch)=closed.at(r,c,ch)-img.at(r,c,ch);
    return out;
}

Image imgradient_morph(const Image& img, int ksize) {
    auto dil=imdilate(img,ksize), ero=imerode(img,ksize);
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r,c,ch)=dil.at(r,c,ch)-ero.at(r,c,ch);
    return out;
}

// ========================== Thresholding ==========================

Image threshold_binary(const Image& img, float t) {
    auto g=img.channels>1?rgb2gray(img):img;
    Image out(img.rows,img.cols,1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c)
        out.at(r,c,0)=g.at(r,c,0)>=t?1.f:0.f;
    return out;
}

Image threshold_otsu(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    // Compute histogram
    const int B=256;
    std::vector<int> hist(B,0);
    int N=img.rows*img.cols;
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        int bin=(int)(g.at(r,c,0)*255);
        bin=std::min(std::max(bin,0),B-1);
        hist[bin]++;
    }
    // Otsu
    double total=N, best=0; int best_t=128;
    double sumB=0, wB=0; double sum1=0;
    for (int i=0;i<B;++i) sum1+=i*hist[i];
    for (int t=0;t<B;++t) {
        wB+=hist[t]; if (wB==0) continue;
        double wF=total-wB; if (wF==0) break;
        sumB+=t*hist[t];
        double mB=sumB/wB, mF=(sum1-sumB)/wF;
        double var=wB*wF*(mB-mF)*(mB-mF);
        if (var>best){best=var;best_t=t;}
    }
    // best_t is the LAST bin of the background class -- the loop accumulates wB up
    // to and including t -- so the split is "background <= best_t, foreground above".
    // Thresholding at best_t/255 with threshold_binary's `>=` puts the whole
    // background bin on the foreground side, and on a clean two-mode image that is
    // every pixel: measured, a half-and-half image of 0.60 and 0.92 came back all
    // ones, and so did 0.20 and 0.80, and 0.55 and 0.95. The threshold has to be the
    // first FOREGROUND bin. A pixel lands in bin (int)(v*255), so bin >= best_t + 1
    // is exactly v >= (best_t + 1)/255 and the comparison stays as it is.
    return threshold_binary(g, static_cast<float>(best_t + 1)/255.f);
}

namespace {

struct WatershedNode {
    float height;
    int r, c;
};

// Max-heap on height; matches std::make_heap/pop_heap/push_heap ordering.
class WatershedHeap {
    std::vector<WatershedNode> nodes_;

    static bool less_height(const WatershedNode& a, const WatershedNode& b) {
        return a.height < b.height;
    }

    void sift_up(size_t i) {
        while (i > 0) {
            const size_t p = (i - 1) / 2;
            if (!less_height(nodes_[p], nodes_[i])) break;
            std::swap(nodes_[p], nodes_[i]);
            i = p;
        }
    }

    void sift_down(size_t i) {
        const size_t n = nodes_.size();
        while (true) {
            const size_t left = 2 * i + 1;
            if (left >= n) break;
            size_t best = left;
            const size_t right = left + 1;
            if (right < n && less_height(nodes_[left], nodes_[right])) best = right;
            if (!less_height(nodes_[i], nodes_[best])) break;
            std::swap(nodes_[i], nodes_[best]);
            i = best;
        }
    }

public:
    void clear() { nodes_.clear(); }
    void reserve(size_t n) { nodes_.reserve(n); }
    bool empty() const { return nodes_.empty(); }

    void push(WatershedNode node) {
        nodes_.push_back(node);
        sift_up(nodes_.size() - 1);
    }

    WatershedNode pop() {
        std::swap(nodes_.front(), nodes_.back());
        WatershedNode top = nodes_.back();
        nodes_.pop_back();
        if (!nodes_.empty()) sift_down(0);
        return top;
    }

    void build(std::vector<WatershedNode>&& seed) {
        nodes_ = std::move(seed);
        for (int i = static_cast<int>(nodes_.size()) / 2 - 1; i >= 0; --i)
            sift_down(static_cast<size_t>(i));
    }
};

} // namespace

Image watershed(const Image& gray, const Image& markers) {
    if (gray.empty() || markers.empty()) return Image{};
    if (gray.rows != markers.rows || gray.cols != markers.cols) return Image{};

    const Image g = gray.channels > 1 ? rgb2gray(gray) : gray;
    const Image mk = markers.channels > 1 ? rgb2gray(markers) : markers;

    const int R = g.rows, C = g.cols;
    const size_t N = static_cast<size_t>(R * C);
    constexpr int k_watershed = -1;
    constexpr int k_unlabeled = 0;

    std::vector<float> gray_flat;
    const float* gray_vals = nullptr;
    if (g.channels == 1 && g.data.size() == N) {
        gray_vals = g.data.data();
    } else {
        gray_flat.resize(N);
        for (size_t i = 0; i < N; ++i)
            gray_flat[i] = g.data[i * g.channels];
        gray_vals = gray_flat.data();
    }

    std::vector<int> labels(N, k_unlabeled);
    for (int r = 0; r < R; ++r) {
        const int row_base = r * C;
        for (int c = 0; c < C; ++c) {
            const int m = static_cast<int>(std::lround(mk.at(r, c, 0)));
            if (m > 0) labels[static_cast<size_t>(row_base + c)] = m;
        }
    }

    std::vector<WatershedNode> seed;
    seed.reserve(N);
    for (int r = 0; r < R; ++r) {
        const int row_base = r * C;
        for (int c = 0; c < C; ++c) {
            const size_t idx = static_cast<size_t>(row_base + c);
            if (labels[idx] > 0) seed.push_back({gray_vals[idx], r, c});
        }
    }

    WatershedHeap pq;
    pq.build(std::move(seed));

    const int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};

    while (!pq.empty()) {
        const auto node = pq.pop();
        const int r = node.r, c = node.c;
        const size_t ci = static_cast<size_t>(r * C + c);

        int cur = labels[ci];
        if (cur <= k_unlabeled) continue;

        for (int d = 0; d < 4; ++d) {
            const int nr = r + dr[d], nc = c + dc[d];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;

            const size_t ni = static_cast<size_t>(nr * C + nc);
            int& nl = labels[ni];
            if (nl == k_unlabeled) {
                nl = cur;
                pq.push({gray_vals[ni], nr, nc});
            } else if (nl > k_unlabeled && nl != cur) {
                labels[ci] = k_watershed;
                nl = k_watershed;
                cur = k_watershed;
            }
        }
    }

    Image out(R, C, 1);
    for (size_t i = 0; i < N; ++i)
        out.data[i] = labels[i] > 0 ? static_cast<float>(labels[i]) : 0.f;
    return out;
}

namespace {

struct LabPixel {
    float L, a, b;
};

float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f
                         : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

LabPixel rgb_to_lab(float r, float g, float b) {
    const float R = srgb_to_linear(r);
    const float G = srgb_to_linear(g);
    const float B = srgb_to_linear(b);
    const float X = 0.4124564f * R + 0.3575761f * G + 0.1804375f * B;
    const float Y = 0.2126729f * R + 0.7151522f * G + 0.0721750f * B;
    const float Z = 0.0193339f * R + 0.1191920f * G + 0.9503041f * B;
    constexpr float Xn = 0.95047f, Yn = 1.f, Zn = 1.08883f;
    auto f = [](float t) {
        return t > 0.008856f ? std::cbrt(t) : (7.787f * t + 16.f / 116.f);
    };
    const float fy = f(Y / Yn);
    return {116.f * fy - 16.f, 500.f * (f(X / Xn) - fy), 200.f * (fy - f(Z / Zn))};
}

Image to_rgb(const Image& img) {
    if (img.empty()) return Image{};
    if (img.channels >= 3) return img;
    return gray2rgb(img);
}

struct SlicCenter {
    float L, a, b;
    float x, y;
};

} // namespace

Image slic(const Image& rgb, int num_superpixels, double compactness) {
    if (rgb.empty() || num_superpixels <= 0) return Image{};

    const Image src = to_rgb(rgb);
    const int R = src.rows, C = src.cols;
    const int N = R * C;
    const int K = std::min(num_superpixels, N);
    if (K <= 0) return Image{};

    std::vector<LabPixel> lab(static_cast<size_t>(N));
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            lab[static_cast<size_t>(r * C + c)] =
                rgb_to_lab(src.at(r, c, 0), src.at(r, c, 1), src.at(r, c, 2));
        }

    const float S = std::sqrt(static_cast<float>(N) / static_cast<float>(K));
    const float m = static_cast<float>(compactness);
    const float inv_s = m / (S + 1e-6f);
    const float inv_s2 = inv_s * inv_s;

    std::vector<SlicCenter> centers(static_cast<size_t>(K));
    int cid = 0;
    for (int r = static_cast<int>(S / 2.f); r < R && cid < K; r += std::max(1, static_cast<int>(S)))
        for (int c = static_cast<int>(S / 2.f); c < C && cid < K; c += std::max(1, static_cast<int>(S))) {
            const auto& px = lab[static_cast<size_t>(r * C + c)];
            centers[static_cast<size_t>(cid)] = {px.L, px.a, px.b, static_cast<float>(c),
                                                   static_cast<float>(r)};
            ++cid;
        }
    while (cid < K) {
        const int idx = cid % N;
        const auto& px = lab[static_cast<size_t>(idx)];
        centers[static_cast<size_t>(cid)] = {px.L, px.a, px.b, static_cast<float>(idx % C),
                                               static_cast<float>(idx / C)};
        ++cid;
    }

    std::vector<int> labels(static_cast<size_t>(N), -1);
    std::vector<float> dist(static_cast<size_t>(N), 1e30f);
    std::vector<double> sumL(static_cast<size_t>(K), 0.0);
    std::vector<double> suma(static_cast<size_t>(K), 0.0);
    std::vector<double> sumb(static_cast<size_t>(K), 0.0);
    std::vector<double> sumx(static_cast<size_t>(K), 0.0);
    std::vector<double> sumy(static_cast<size_t>(K), 0.0);
    std::vector<int> count(static_cast<size_t>(K), 0);
    const int search = std::max(1, static_cast<int>(2.f * S));
    constexpr int k_max_iter = 10;

    for (int iter = 0; iter < k_max_iter; ++iter) {
        std::fill(dist.begin(), dist.end(), 1e30f);

        for (int k = 0; k < K; ++k) {
            const auto& ctr = centers[static_cast<size_t>(k)];
            const int cx = static_cast<int>(ctr.x);
            const int cy = static_cast<int>(ctr.y);
            const int r0 = std::max(0, cy - search);
            const int r1 = std::min(R - 1, cy + search);
            const int c0 = std::max(0, cx - search);
            const int c1 = std::min(C - 1, cx + search);

            for (int r = r0; r <= r1; ++r) {
                const float dr = static_cast<float>(r) - ctr.y;
                const int row_base = r * C;
                for (int c = c0; c <= c1; ++c) {
                    const size_t pi = static_cast<size_t>(row_base + c);
                    const float dc_col = static_cast<float>(c) - ctr.x;
                    const float ds = inv_s2 * (dr * dr + dc_col * dc_col);
                    if (ds >= dist[pi]) continue;

                    const auto& px = lab[pi];
                    const float dL = px.L - ctr.L;
                    const float da = px.a - ctr.a;
                    const float db = px.b - ctr.b;
                    const float dc = dL * dL + da * da + db * db;
                    const float D = dc + ds;
                    if (D < dist[pi]) {
                        dist[pi] = D;
                        labels[pi] = k;
                    }
                }
            }
        }

        std::fill(sumL.begin(), sumL.end(), 0.0);
        std::fill(suma.begin(), suma.end(), 0.0);
        std::fill(sumb.begin(), sumb.end(), 0.0);
        std::fill(sumx.begin(), sumx.end(), 0.0);
        std::fill(sumy.begin(), sumy.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);

        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                const int k = labels[static_cast<size_t>(r * C + c)];
                if (k < 0) continue;
                const size_t ki = static_cast<size_t>(k);
                const auto& px = lab[static_cast<size_t>(r * C + c)];
                sumL[ki] += px.L;
                suma[ki] += px.a;
                sumb[ki] += px.b;
                sumx[ki] += c;
                sumy[ki] += r;
                ++count[ki];
            }

        float max_move = 0.f;
        for (int k = 0; k < K; ++k) {
            if (count[static_cast<size_t>(k)] == 0) continue;
            const float inv = 1.f / static_cast<float>(count[static_cast<size_t>(k)]);
            auto& ctr = centers[static_cast<size_t>(k)];
            const float nL = static_cast<float>(sumL[static_cast<size_t>(k)] * inv);
            const float na = static_cast<float>(suma[static_cast<size_t>(k)] * inv);
            const float nb = static_cast<float>(sumb[static_cast<size_t>(k)] * inv);
            const float nx = static_cast<float>(sumx[static_cast<size_t>(k)] * inv);
            const float ny = static_cast<float>(sumy[static_cast<size_t>(k)] * inv);
            max_move = std::max(max_move, std::hypot(nx - ctr.x, ny - ctr.y));
            max_move = std::max(max_move, std::abs(nL - ctr.L));
            ctr = {nL, na, nb, nx, ny};
        }
        if (max_move < 0.01f) break;
    }

    Image out(R, C, 1);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            const int k = labels[static_cast<size_t>(r * C + c)];
            out.at(r, c, 0) = k >= 0 ? static_cast<float>(k + 1) : 0.f;
        }
    return out;
}

// ========================== Graph-Cut Segmentation ==========================

namespace {

// Residual capacities at or below this count as saturated. With floating-point
// capacities an exact-zero test would let a chain of subtractions leave
// residuals of order 1e-17 behind and generate a long tail of near-zero
// augmentations; the epsilon bounds the augmentation count and makes the
// result reproducible. It also means an n-link weight below k_gc_eps (say
// lambda*exp(-50)) is dropped at graph construction and contributes nothing to
// the reported cut value.
constexpr double k_gc_eps = 1e-12;

// parent_[v] sentinels: v is a search-tree root (its parent is S or T), v's
// parent link was destroyed by the last augmentation, and v is free (in
// neither search tree).
constexpr int k_gc_arc_terminal = -1;
constexpr int k_gc_arc_orphan   = -2;
constexpr int k_gc_arc_none     = -3;
// next_[v] sentinel: v is not on the active list.
constexpr int k_gc_no_queue = -1;

// rows*cols above this would overflow the int arc index at connectivity 8
// (up to 8 directed arcs per pixel).
constexpr std::size_t k_gc_max_pixels = 200000000u;

constexpr int    k_gc_two_means_iters = 100;
constexpr double k_gc_two_means_tol   = 1e-12;
constexpr int    k_gc_kmeans_iters    = 10;
constexpr int    k_gc_max_components  = 8;
// Variance floor for a mixture component (std >= 0.01 in [0, 1] intensity
// units), so a component that collects identical values cannot produce an
// infinite density; and a density floor, so -ln(p) stays <= 27.64.
constexpr double k_gc_var_floor   = 1e-4;
constexpr double k_gc_min_density = 1e-12;

// Named widening for container subscripts: every index in this section is a
// non-negative int by construction, and spelling the conversion out keeps the
// signed/unsigned boundary explicit.
inline std::size_t gc_ix(int i) { return static_cast<std::size_t>(i); }

// Boykov-Kolmogorov max-flow / min-cut on a sparse graph with paired reverse
// arcs (Boykov & Kolmogorov, PAMI 2004). Terminal (source/sink) capacities are
// folded into one signed value per node: tr_cap_[v] > 0 is residual capacity
// from S, tr_cap_[v] < 0 is residual capacity to T, so there are no explicit
// terminal node objects. Arcs live in CSR form with an explicit sister array,
// which is cache-friendly and — unlike a per-node linked list — makes the arc
// scan order a fixed function of the input.
//
// Usage: add_terminal() / add_edge() for every node and pair, then finalize(),
// solve(), and source_side() for the inclusion-minimal min-cut source set.
class GcMaxFlow {
public:
    explicit GcMaxFlow(int n_nodes, int edge_hint = 0)
        : n_(n_nodes), tr_cap_(gc_ix(n_nodes), 0.0) {
        degree_.assign(gc_ix(n_nodes), 0);
        edge_p_.reserve(gc_ix(edge_hint));
        edge_q_.reserve(gc_ix(edge_hint));
        edge_w_.reserve(gc_ix(edge_hint));
    }

    // cap_src is the S->v capacity, cap_snk the v->T capacity. min(cap_src,
    // cap_snk) is flow that trivially saturates through v, so it goes straight
    // into the running total and only the difference survives as residual.
    void add_terminal(int v, double cap_src, double cap_snk) {
        flow_ += std::min(cap_src, cap_snk);
        tr_cap_[gc_ix(v)] += cap_src - cap_snk;
    }

    // Undirected pair with the same capacity in both directions, as a Potts
    // term requires. Weights at or below the epsilon are dropped.
    void add_edge(int p, int q, double w) {
        if (!(w > k_gc_eps)) return;
        edge_p_.push_back(p);
        edge_q_.push_back(q);
        edge_w_.push_back(w);
        ++degree_[gc_ix(p)];
        ++degree_[gc_ix(q)];
    }

    // Build the CSR arc arrays from the staged edge list, then release the
    // staging vectors (about a quarter of peak memory on large images).
    void finalize() {
        first_.assign(gc_ix(n_ + 1), 0);
        for (int v = 0; v < n_; ++v)
            first_[gc_ix(v + 1)] = first_[gc_ix(v)] + degree_[gc_ix(v)];

        const int m = static_cast<int>(edge_p_.size());
        arc_head_.assign(gc_ix(2 * m), 0);
        arc_cap_.assign(gc_ix(2 * m), 0.0);
        sister_.assign(gc_ix(2 * m), 0);

        std::vector<int> cursor(first_.begin(), first_.end() - 1);
        for (int k = 0; k < m; ++k) {
            const int p = edge_p_[gc_ix(k)];
            const int q = edge_q_[gc_ix(k)];
            const double w = edge_w_[gc_ix(k)];
            const int ap = cursor[gc_ix(p)]++;
            const int aq = cursor[gc_ix(q)]++;
            arc_head_[gc_ix(ap)] = q;
            arc_cap_[gc_ix(ap)] = w;
            sister_[gc_ix(ap)] = aq;
            arc_head_[gc_ix(aq)] = p;
            arc_cap_[gc_ix(aq)] = w;
            sister_[gc_ix(aq)] = ap;
        }

        std::vector<int>().swap(edge_p_);
        std::vector<int>().swap(edge_q_);
        std::vector<double>().swap(edge_w_);
        std::vector<int>().swap(degree_);
    }

    // Run to completion and return the max-flow value (= the min-cut value of
    // the graph as built). Must be called after finalize().
    double solve() {
        parent_.assign(gc_ix(n_), k_gc_arc_none);
        is_sink_.assign(gc_ix(n_), 0);
        ts_.assign(gc_ix(n_), 0);
        dist_.assign(gc_ix(n_), 0);
        next_.assign(gc_ix(n_), k_gc_no_queue);
        q_first_[0] = q_last_[0] = q_first_[1] = q_last_[1] = -1;
        orphans_.clear();
        time_ = 0;

        // Every node with a live terminal arc is a search-tree root and starts
        // active; every other node is free.
        for (int v = 0; v < n_; ++v) {
            const double t = tr_cap_[gc_ix(v)];
            if (t > k_gc_eps || t < -k_gc_eps) {
                is_sink_[gc_ix(v)] = t > k_gc_eps ? 0 : 1;
                parent_[gc_ix(v)] = k_gc_arc_terminal;
                dist_[gc_ix(v)] = 1;
                set_active(v);
            }
        }

        int current = -1;
        while (true) {
            int i = current;
            if (i >= 0) {
                next_[gc_ix(i)] = k_gc_no_queue;
                if (parent_[gc_ix(i)] == k_gc_arc_none) i = -1;
            }
            if (i < 0) {
                i = next_active();
                if (i < 0) break;
            }

            const int middle = grow(i);
            ++time_;

            if (middle >= 0) {
                next_[gc_ix(i)] = i;  // keep i active across the augmentation
                current = i;
                augment(middle);
                while (!orphans_.empty()) {
                    const int o = orphans_.front();
                    orphans_.pop_front();
                    if (is_sink_[gc_ix(o)]) process_sink_orphan(o);
                    else                    process_source_orphan(o);
                }
            } else {
                current = -1;
            }
        }
        return flow_;
    }

    // Source side of the inclusion-minimal minimum cut: the nodes reachable
    // from S in the final residual graph. Minimum cuts form a lattice closed
    // under intersection, so this set is exactly the intersection of every
    // optimal labelling's source side -- independent of which maximum flow the
    // solver happened to find, and hence reproducible.
    std::vector<char> source_side() const {
        std::vector<char> in_s(gc_ix(n_), 0);
        std::vector<int> stack;
        for (int v = 0; v < n_; ++v)
            if (tr_cap_[gc_ix(v)] > k_gc_eps) {
                in_s[gc_ix(v)] = 1;
                stack.push_back(v);
            }
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            for (int a = first_[gc_ix(v)]; a < first_[gc_ix(v + 1)]; ++a) {
                if (!(arc_cap_[gc_ix(a)] > k_gc_eps)) continue;
                const int u = arc_head_[gc_ix(a)];
                if (in_s[gc_ix(u)]) continue;
                in_s[gc_ix(u)] = 1;
                stack.push_back(u);
            }
        }
        return in_s;
    }

private:
    void set_active(int v) {
        if (next_[gc_ix(v)] != k_gc_no_queue) return;  // already queued
        if (q_last_[1] >= 0) next_[gc_ix(q_last_[1])] = v;
        else                 q_first_[1] = v;
        q_last_[1] = v;
        next_[gc_ix(v)] = v;  // tail marker
    }

    // Drain queue 0, swapping queue 1 into it when it runs dry. A queued node
    // whose parent link has since been destroyed is skipped.
    int next_active() {
        while (true) {
            int i = q_first_[0];
            if (i < 0) {
                q_first_[0] = i = q_first_[1];
                q_last_[0] = q_last_[1];
                q_first_[1] = -1;
                q_last_[1] = -1;
                if (i < 0) return -1;
            }
            if (next_[gc_ix(i)] == i) { q_first_[0] = -1; q_last_[0] = -1; }
            else                        q_first_[0] = next_[gc_ix(i)];
            next_[gc_ix(i)] = k_gc_no_queue;
            if (parent_[gc_ix(i)] != k_gc_arc_none) return i;
        }
    }

    void add_orphan_front(int v) { parent_[gc_ix(v)] = k_gc_arc_orphan; orphans_.push_front(v); }
    void add_orphan_back(int v)  { parent_[gc_ix(v)] = k_gc_arc_orphan; orphans_.push_back(v); }

    // STAGE 1: grow i's search tree along arcs that still have residual
    // capacity, adopting free neighbours and shortening the distance labels of
    // stale ones. Returns the arc that first touches the opposite tree -- always
    // oriented source-side -> sink-side -- or -1 if the tree could not grow.
    int grow(int i) {
        const int a_begin = first_[gc_ix(i)];
        const int a_end = first_[gc_ix(i + 1)];
        if (!is_sink_[gc_ix(i)]) {
            for (int a = a_begin; a < a_end; ++a) {
                if (!(arc_cap_[gc_ix(a)] > k_gc_eps)) continue;
                const int j = arc_head_[gc_ix(a)];
                if (parent_[gc_ix(j)] == k_gc_arc_none) {
                    is_sink_[gc_ix(j)] = 0;
                    adopt(j, sister_[gc_ix(a)], i);
                    set_active(j);
                } else if (is_sink_[gc_ix(j)]) {
                    return a;
                } else if (ts_[gc_ix(j)] <= ts_[gc_ix(i)] && dist_[gc_ix(j)] > dist_[gc_ix(i)]) {
                    adopt(j, sister_[gc_ix(a)], i);
                }
            }
        } else {
            for (int a = a_begin; a < a_end; ++a) {
                // In the sink tree flow runs j -> i, so the capacity that
                // matters lives on the reverse arc.
                const int s = sister_[gc_ix(a)];
                if (!(arc_cap_[gc_ix(s)] > k_gc_eps)) continue;
                const int j = arc_head_[gc_ix(a)];
                if (parent_[gc_ix(j)] == k_gc_arc_none) {
                    is_sink_[gc_ix(j)] = 1;
                    adopt(j, s, i);
                    set_active(j);
                } else if (!is_sink_[gc_ix(j)]) {
                    return s;  // flip: the returned arc must run source -> sink
                } else if (ts_[gc_ix(j)] <= ts_[gc_ix(i)] && dist_[gc_ix(j)] > dist_[gc_ix(i)]) {
                    adopt(j, s, i);
                }
            }
        }
        return -1;
    }

    // Hang j off i via the arc j -> i, inheriting i's timestamp.
    void adopt(int j, int arc_to_parent, int i) {
        parent_[gc_ix(j)] = arc_to_parent;
        ts_[gc_ix(j)] = ts_[gc_ix(i)];
        dist_[gc_ix(j)] = dist_[gc_ix(i)] + 1;
    }

    // STAGE 2: push the bottleneck along source-root -> middle -> sink-root and
    // orphan every node whose parent arc saturates. parent_[v] is the arc from
    // v to its parent, so in the source tree (where flow runs parent -> v) the
    // usable capacity is on its sister, and in the sink tree (flow runs v ->
    // parent) it is on the arc itself.
    void augment(int middle) {
        // 2a: bottleneck up the source tree.
        double b = arc_cap_[gc_ix(middle)];
        int i = arc_head_[gc_ix(sister_[gc_ix(middle)])];
        while (true) {
            const int a = parent_[gc_ix(i)];
            if (a == k_gc_arc_terminal) break;
            b = std::min(b, arc_cap_[gc_ix(sister_[gc_ix(a)])]);
            i = arc_head_[gc_ix(a)];
        }
        b = std::min(b, tr_cap_[gc_ix(i)]);

        // 2b: bottleneck down the sink tree.
        int j = arc_head_[gc_ix(middle)];
        while (true) {
            const int a = parent_[gc_ix(j)];
            if (a == k_gc_arc_terminal) break;
            b = std::min(b, arc_cap_[gc_ix(a)]);
            j = arc_head_[gc_ix(a)];
        }
        b = std::min(b, -tr_cap_[gc_ix(j)]);

        arc_cap_[gc_ix(sister_[gc_ix(middle)])] += b;
        arc_cap_[gc_ix(middle)] -= b;

        i = arc_head_[gc_ix(sister_[gc_ix(middle)])];
        while (true) {
            const int a = parent_[gc_ix(i)];
            if (a == k_gc_arc_terminal) break;
            const int s = sister_[gc_ix(a)];
            arc_cap_[gc_ix(a)] += b;
            arc_cap_[gc_ix(s)] -= b;
            const int nxt = arc_head_[gc_ix(a)];
            if (!(arc_cap_[gc_ix(s)] > k_gc_eps)) add_orphan_front(i);
            i = nxt;
        }
        tr_cap_[gc_ix(i)] -= b;
        if (!(tr_cap_[gc_ix(i)] > k_gc_eps)) add_orphan_front(i);

        j = arc_head_[gc_ix(middle)];
        while (true) {
            const int a = parent_[gc_ix(j)];
            if (a == k_gc_arc_terminal) break;
            const int s = sister_[gc_ix(a)];
            arc_cap_[gc_ix(s)] += b;
            arc_cap_[gc_ix(a)] -= b;
            const int nxt = arc_head_[gc_ix(a)];
            if (!(arc_cap_[gc_ix(a)] > k_gc_eps)) add_orphan_front(j);
            j = nxt;
        }
        tr_cap_[gc_ix(j)] += b;
        if (!(tr_cap_[gc_ix(j)] < -k_gc_eps)) add_orphan_front(j);

        flow_ += b;
    }

    // Walk start up to its tree root, returning the hop count, or -1 if the
    // walk hits an orphan or a free node (start is not rooted this pass).
    // Stamps the current time onto every terminal-rooted node it reaches.
    int origin_dist(int start) {
        int d = 0;
        int j = start;
        while (true) {
            if (ts_[gc_ix(j)] == time_) return d + dist_[gc_ix(j)];
            const int a = parent_[gc_ix(j)];
            ++d;
            if (a == k_gc_arc_terminal) {
                ts_[gc_ix(j)] = time_;
                dist_[gc_ix(j)] = 1;
                return d;
            }
            if (a == k_gc_arc_orphan || a == k_gc_arc_none) return -1;
            j = arc_head_[gc_ix(a)];
        }
    }

    // Stamp the freshly measured distances back down the path origin_dist()
    // just walked. The ts_ guard is what keeps parent_[k] from ever being a
    // sentinel here: origin_dist() stamped every node on the path first.
    void relabel_path(int a0, int d) {
        int k = arc_head_[gc_ix(a0)];
        int dd = d;
        while (ts_[gc_ix(k)] != time_) {
            ts_[gc_ix(k)] = time_;
            dist_[gc_ix(k)] = dd--;
            k = arc_head_[gc_ix(parent_[gc_ix(k)])];
        }
    }

    // STAGE 3a: re-parent a source-tree orphan onto the closest still-rooted
    // source neighbour; failing that, free it and orphan its own children.
    void process_source_orphan(int i) {
        int best_arc = -1;
        int d_min = 0;
        for (int a0 = first_[gc_ix(i)]; a0 < first_[gc_ix(i + 1)]; ++a0) {
            if (!(arc_cap_[gc_ix(sister_[gc_ix(a0)])] > k_gc_eps)) continue;
            const int j = arc_head_[gc_ix(a0)];
            if (is_sink_[gc_ix(j)]) continue;
            const int pj = parent_[gc_ix(j)];
            if (pj == k_gc_arc_none || pj == k_gc_arc_orphan) continue;
            const int d = origin_dist(j);
            if (d < 0) continue;
            if (best_arc < 0 || d < d_min) { best_arc = a0; d_min = d; }
            relabel_path(a0, d);
        }

        parent_[gc_ix(i)] = best_arc >= 0 ? best_arc : k_gc_arc_none;
        if (best_arc >= 0) {
            ts_[gc_ix(i)] = time_;
            dist_[gc_ix(i)] = d_min + 1;
            return;
        }

        ts_[gc_ix(i)] = 0;  // i is free now; drop its stale timestamp
        for (int a0 = first_[gc_ix(i)]; a0 < first_[gc_ix(i + 1)]; ++a0) {
            const int j = arc_head_[gc_ix(a0)];
            if (is_sink_[gc_ix(j)]) continue;
            const int pj = parent_[gc_ix(j)];
            if (pj == k_gc_arc_none) continue;
            if (arc_cap_[gc_ix(sister_[gc_ix(a0)])] > k_gc_eps) set_active(j);
            if (pj != k_gc_arc_terminal && pj != k_gc_arc_orphan && arc_head_[gc_ix(pj)] == i)
                add_orphan_back(j);
        }
    }

    // STAGE 3b: the mirror image of 3a for the sink tree -- the residual test
    // uses the arc itself instead of its sister, and the tree membership test
    // flips.
    void process_sink_orphan(int i) {
        int best_arc = -1;
        int d_min = 0;
        for (int a0 = first_[gc_ix(i)]; a0 < first_[gc_ix(i + 1)]; ++a0) {
            if (!(arc_cap_[gc_ix(a0)] > k_gc_eps)) continue;
            const int j = arc_head_[gc_ix(a0)];
            if (!is_sink_[gc_ix(j)]) continue;
            const int pj = parent_[gc_ix(j)];
            if (pj == k_gc_arc_none || pj == k_gc_arc_orphan) continue;
            const int d = origin_dist(j);
            if (d < 0) continue;
            if (best_arc < 0 || d < d_min) { best_arc = a0; d_min = d; }
            relabel_path(a0, d);
        }

        parent_[gc_ix(i)] = best_arc >= 0 ? best_arc : k_gc_arc_none;
        if (best_arc >= 0) {
            ts_[gc_ix(i)] = time_;
            dist_[gc_ix(i)] = d_min + 1;
            return;
        }

        ts_[gc_ix(i)] = 0;
        for (int a0 = first_[gc_ix(i)]; a0 < first_[gc_ix(i + 1)]; ++a0) {
            const int j = arc_head_[gc_ix(a0)];
            if (!is_sink_[gc_ix(j)]) continue;
            const int pj = parent_[gc_ix(j)];
            if (pj == k_gc_arc_none) continue;
            if (arc_cap_[gc_ix(a0)] > k_gc_eps) set_active(j);
            if (pj != k_gc_arc_terminal && pj != k_gc_arc_orphan && arc_head_[gc_ix(pj)] == i)
                add_orphan_back(j);
        }
    }

    int n_ = 0;
    double flow_ = 0.0;
    std::vector<double> tr_cap_;
    std::vector<int> degree_;
    std::vector<int> edge_p_, edge_q_;
    std::vector<double> edge_w_;
    std::vector<int> first_, arc_head_, sister_;
    std::vector<double> arc_cap_;
    std::vector<int> parent_, ts_, dist_, next_;
    std::vector<char> is_sink_;
    std::deque<int> orphans_;
    int q_first_[2] = {-1, -1};
    int q_last_[2] = {-1, -1};
    int time_ = 0;
};

// Flatten an image to one double per pixel. Deliberately stricter than this
// module's usual `channels > 1 ? rgb2gray(img) : img`: rgb2gray() reads
// channels 0, 1 and 2 unconditionally, which would run past the end of a
// 2-channel image, so only `channels >= 3` goes through it and everything else
// takes channel 0. Returns {} for an empty or malformed image.
std::vector<double> gc_gray_plane(const Image& img) {
    if (img.empty() || img.rows <= 0 || img.cols <= 0 || img.channels <= 0) return {};
    const std::size_t n = gc_ix(img.rows) * gc_ix(img.cols);
    if (img.data.size() < n * gc_ix(img.channels)) return {};

    std::vector<double> out(n, 0.0);
    if (img.channels >= 3) {
        const Image g = rgb2gray(img);
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(g.data[i]);
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = static_cast<double>(img.data[i * gc_ix(img.channels)]);
    }
    return out;
}

// Deterministic 1-D 2-means (Lloyd) split, initialised at the extremes. A
// uniform input leaves both centres equal, which is what makes an unseeded
// uniform image score every pixel identically.
void gc_two_means(const std::vector<double>& v, double& mu_lo, double& mu_hi) {
    double a = v[0], b = v[0];
    for (const double x : v) { a = std::min(a, x); b = std::max(b, x); }

    for (int it = 0; it < k_gc_two_means_iters; ++it) {
        const double mid = 0.5 * (a + b);
        double sa = 0.0, sb = 0.0;
        int na = 0, nb = 0;
        for (const double x : v) {
            if (x <= mid) { sa += x; ++na; }
            else          { sb += x; ++nb; }
        }
        const double na_mean = na > 0 ? sa / static_cast<double>(na) : a;
        const double nb_mean = nb > 0 ? sb / static_cast<double>(nb) : b;
        const double move = std::max(std::abs(na_mean - a), std::abs(nb_mean - b));
        a = na_mean;
        b = nb_mean;
        if (move < k_gc_two_means_tol) break;
    }
    mu_lo = std::min(a, b);
    mu_hi = std::max(a, b);
}

struct GcEdge { int p, q; double w; };

// Contrast-sensitive Potts n-links, enumerated with forward-only offsets so
// each undirected pair is created exactly once. Also returns the Boykov-Jolly
// hard-constraint capacity K = 1 + max over p of the incident weight sum:
// moving a hard-foreground pixel to the sink side saves K and costs at most
// K - 1, so violating a seed is never optimal.
void gc_build_edges(const std::vector<double>& I, int R, int C, double lambda, double sigma,
                    int connectivity, std::vector<GcEdge>& edges, double& K) {
    const double lam = lambda > 0.0 ? lambda : 0.0;
    const int n_off = connectivity == 8 ? 4 : 2;
    const int dr[4] = {0, 1, 1, 1};
    const int dc[4] = {1, 0, 1, -1};
    const double inv_dist[4] = {1.0, 1.0, 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};
    const double denom = sigma > 0.0 ? 2.0 * sigma * sigma : 0.0;

    edges.clear();
    std::vector<double> wsum(gc_ix(R) * gc_ix(C), 0.0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            const int p = r * C + c;
            for (int k = 0; k < n_off; ++k) {
                const int nr = r + dr[k], nc = c + dc[k];
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                const int q = nr * C + nc;
                const double d = I[gc_ix(p)] - I[gc_ix(q)];
                // sigma <= 0 takes the exact limit: a hard Potts term that only
                // links neighbours of exactly equal intensity.
                const double w = denom > 0.0
                                     ? lam * std::exp(-(d * d) / denom) * inv_dist[k]
                                     : (d == 0.0 ? lam * inv_dist[k] : 0.0);
                if (!(w > k_gc_eps)) continue;
                edges.push_back({p, q, w});
                wsum[gc_ix(p)] += w;
                wsum[gc_ix(q)] += w;
            }
        }

    double max_wsum = 0.0;
    for (const double x : wsum) max_wsum = std::max(max_wsum, x);
    K = 1.0 + max_wsum;
}

// Shared pipeline behind graph_cut_segment() and min_cut_value(), so the two
// entry points cannot drift apart: they must agree on the means, the hard
// constraints, the n-links and the reparametrisation.
struct GcSolution {
    std::vector<char> foreground;
    double energy = 0.0;
    bool ok = false;
};

GcSolution gc_run(const Image& gray, const Image& fg_seeds, const Image& bg_seeds,
                  double lambda, double sigma, int connectivity) {
    GcSolution sol;
    if (gray.empty() || gray.rows <= 0 || gray.cols <= 0) return sol;
    if (gc_ix(gray.rows) * gc_ix(gray.cols) > k_gc_max_pixels) return sol;
    if (!fg_seeds.empty() && (fg_seeds.rows != gray.rows || fg_seeds.cols != gray.cols)) return sol;
    if (!bg_seeds.empty() && (bg_seeds.rows != gray.rows || bg_seeds.cols != gray.cols)) return sol;

    const std::vector<double> I = gc_gray_plane(gray);
    if (I.empty()) return sol;
    const std::vector<double> F = gc_gray_plane(fg_seeds);
    const std::vector<double> B = gc_gray_plane(bg_seeds);

    const int R = gray.rows, C = gray.cols;
    const int N = R * C;

    // Seed classification; a pixel marked in both masks is foreground.
    std::vector<char> is_fg(gc_ix(N), 0), is_bg(gc_ix(N), 0);
    int n_fg = 0, n_bg = 0;
    for (int i = 0; i < N; ++i) {
        if (!F.empty() && F[gc_ix(i)] >= 0.5) { is_fg[gc_ix(i)] = 1; ++n_fg; }
        else if (!B.empty() && B[gc_ix(i)] >= 0.5) { is_bg[gc_ix(i)] = 1; ++n_bg; }
    }

    double mu_fg = 0.0, mu_bg = 0.0;
    if (n_fg > 0 && n_bg > 0) {
        double sf = 0.0, sb = 0.0;
        for (int i = 0; i < N; ++i) {
            if (is_fg[gc_ix(i)]) sf += I[gc_ix(i)];
            else if (is_bg[gc_ix(i)]) sb += I[gc_ix(i)];
        }
        mu_fg = sf / static_cast<double>(n_fg);
        mu_bg = sb / static_cast<double>(n_bg);
    } else if (n_fg > 0) {
        // Only foreground seeded: the background mean is taken over everything
        // the foreground mask does not cover.
        double sf = 0.0, so = 0.0;
        int n_other = 0;
        for (int i = 0; i < N; ++i) {
            if (is_fg[gc_ix(i)]) sf += I[gc_ix(i)];
            else { so += I[gc_ix(i)]; ++n_other; }
        }
        mu_fg = sf / static_cast<double>(n_fg);
        mu_bg = n_other > 0 ? so / static_cast<double>(n_other) : mu_fg;
    } else if (n_bg > 0) {
        double sb = 0.0, so = 0.0;
        int n_other = 0;
        for (int i = 0; i < N; ++i) {
            if (is_bg[gc_ix(i)]) sb += I[gc_ix(i)];
            else { so += I[gc_ix(i)]; ++n_other; }
        }
        mu_bg = sb / static_cast<double>(n_bg);
        mu_fg = n_other > 0 ? so / static_cast<double>(n_other) : mu_bg;
    } else {
        gc_two_means(I, mu_bg, mu_fg);
    }

    std::vector<GcEdge> edges;
    double K = 0.0;
    gc_build_edges(I, R, C, lambda, sigma, connectivity, edges, K);

    GcMaxFlow g(N, static_cast<int>(edges.size()));

    // Reparametrise so both terminal capacities are non-negative and one of
    // them is zero: subtracting m = min(D_fg, D_bg) from both shifts E by the
    // constant offset for every labelling, so the argmin is unchanged. An arc
    // S->p is severed when p lands on the sink side, so its capacity is the
    // price of calling p background, i.e. D_bg.
    double offset = 0.0;
    for (int i = 0; i < N; ++i) {
        double d_fg = 0.0, d_bg = 0.0;
        if (is_fg[gc_ix(i)]) { d_fg = 0.0; d_bg = K; }
        else if (is_bg[gc_ix(i)]) { d_fg = K; d_bg = 0.0; }
        else {
            const double x = I[gc_ix(i)];
            d_fg = (x - mu_fg) * (x - mu_fg);
            d_bg = (x - mu_bg) * (x - mu_bg);
        }
        const double m = std::min(d_fg, d_bg);
        offset += m;
        g.add_terminal(i, d_bg - m, d_fg - m);
    }
    for (const GcEdge& e : edges) g.add_edge(e.p, e.q, e.w);

    g.finalize();
    const double flow = g.solve();

    sol.foreground = g.source_side();
    sol.energy = offset + flow;
    sol.ok = true;
    return sol;
}

// A 1-D Gaussian mixture over intensities, fitted by hard assignment.
struct GcMixture {
    std::vector<double> pi, mu, var;

    double density(double x) const {
        double p = 0.0;
        for (std::size_t j = 0; j < pi.size(); ++j) {
            if (pi[j] <= 0.0) continue;  // component collected no pixels
            const double d = x - mu[j];
            p += pi[j] * std::exp(-(d * d) / (2.0 * var[j])) / std::sqrt(2.0 * M_PI * var[j]);
        }
        return p;
    }
};

// Deterministic 1-D mixture fit: Lloyd k-means seeded at the sorted sample's
// (j + 0.5)/K quantiles, then Gaussian moments per cluster. No random
// restarts, so repeated GrabCut calls give bit-identical results.
GcMixture gc_fit_mixture(std::vector<double> vals, int n_components) {
    GcMixture mix;
    if (vals.empty()) return mix;

    std::sort(vals.begin(), vals.end());
    const int n = static_cast<int>(vals.size());
    int K = std::min(std::max(n_components, 1), k_gc_max_components);
    K = std::min(K, n);

    std::vector<double> mu(gc_ix(K), 0.0);
    for (int j = 0; j < K; ++j) {
        const long long q = (2LL * j + 1) * static_cast<long long>(n) / (2LL * K);
        const int idx = std::min(std::max(static_cast<int>(q), 0), n - 1);
        mu[gc_ix(j)] = vals[gc_ix(idx)];
    }

    std::vector<int> assign(gc_ix(n), -1);
    for (int it = 0; it < k_gc_kmeans_iters; ++it) {
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            int best = 0;
            double best_d = std::abs(vals[gc_ix(i)] - mu[0]);
            for (int j = 1; j < K; ++j) {
                const double d = std::abs(vals[gc_ix(i)] - mu[gc_ix(j)]);
                if (d < best_d) { best_d = d; best = j; }  // ties keep the lowest index
            }
            if (assign[gc_ix(i)] != best) { assign[gc_ix(i)] = best; changed = true; }
        }
        std::vector<double> sum(gc_ix(K), 0.0);
        std::vector<int> count(gc_ix(K), 0);
        for (int i = 0; i < n; ++i) {
            const int j = assign[gc_ix(i)];
            sum[gc_ix(j)] += vals[gc_ix(i)];
            ++count[gc_ix(j)];
        }
        for (int j = 0; j < K; ++j)
            if (count[gc_ix(j)] > 0)
                mu[gc_ix(j)] = sum[gc_ix(j)] / static_cast<double>(count[gc_ix(j)]);
        if (!changed) break;
    }

    std::vector<double> sq(gc_ix(K), 0.0);
    std::vector<int> count(gc_ix(K), 0);
    for (int i = 0; i < n; ++i) {
        const int j = assign[gc_ix(i)];
        const double d = vals[gc_ix(i)] - mu[gc_ix(j)];
        sq[gc_ix(j)] += d * d;
        ++count[gc_ix(j)];
    }

    mix.pi.assign(gc_ix(K), 0.0);
    mix.mu = mu;
    mix.var.assign(gc_ix(K), k_gc_var_floor);
    for (int j = 0; j < K; ++j) {
        if (count[gc_ix(j)] <= 0) continue;
        const double cnt = static_cast<double>(count[gc_ix(j)]);
        mix.pi[gc_ix(j)] = cnt / static_cast<double>(n);
        mix.var[gc_ix(j)] = std::max(sq[gc_ix(j)] / cnt, k_gc_var_floor);
    }
    return mix;
}

} // namespace

Image graph_cut_segment(const Image& gray, const Image& fg_seeds, const Image& bg_seeds,
                        double lambda, double sigma, int connectivity) {
    const GcSolution sol = gc_run(gray, fg_seeds, bg_seeds, lambda, sigma, connectivity);
    if (!sol.ok) return Image{};

    Image out(gray.rows, gray.cols, 1);
    for (std::size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = sol.foreground[i] ? 1.f : 0.f;
    return out;
}

Image graph_cut_segment(const Image& gray, double lambda, double sigma, int connectivity) {
    return graph_cut_segment(gray, Image{}, Image{}, lambda, sigma, connectivity);
}

double min_cut_value(const Image& gray, const Image& fg_seeds, const Image& bg_seeds,
                     double lambda, double sigma, int connectivity) {
    const GcSolution sol = gc_run(gray, fg_seeds, bg_seeds, lambda, sigma, connectivity);
    return sol.ok ? sol.energy : 0.0;
}

double min_cut_value(const Image& gray, double lambda, double sigma, int connectivity) {
    return min_cut_value(gray, Image{}, Image{}, lambda, sigma, connectivity);
}

Image grabcut_segment(const Image& gray, int r0, int c0, int r1, int c1, int iterations,
                      int n_components, double lambda, double sigma, int connectivity) {
    if (gray.empty() || gray.rows <= 0 || gray.cols <= 0) return Image{};
    if (gc_ix(gray.rows) * gc_ix(gray.cols) > k_gc_max_pixels) return Image{};

    const std::vector<double> I = gc_gray_plane(gray);
    if (I.empty()) return Image{};

    const int R = gray.rows, C = gray.cols;
    const int N = R * C;

    // Half-open rectangle clamped to the image, matching imcrop().
    r0 = std::max(0, r0);
    c0 = std::max(0, c0);
    r1 = std::min(R, r1);
    c1 = std::min(C, c1);

    Image out(R, C, 1);
    if (r1 <= r0 || c1 <= c0) return out;  // empty rectangle: all background

    std::vector<char> inside(gc_ix(N), 0);
    for (int r = r0; r < r1; ++r)
        for (int c = c0; c < c1; ++c) inside[gc_ix(r * C + c)] = 1;

    // The intensities never change, so the n-links and K are built once and
    // reused for every pass.
    std::vector<GcEdge> edges;
    double K = 0.0;
    gc_build_edges(I, R, C, lambda, sigma, connectivity, edges, K);

    std::vector<char> lab(inside);
    const int iters = std::max(1, iterations);
    for (int it = 0; it < iters; ++it) {
        std::vector<double> vals_fg, vals_bg;
        vals_fg.reserve(gc_ix(N));
        vals_bg.reserve(gc_ix(N));
        for (int i = 0; i < N; ++i) {
            if (lab[gc_ix(i)]) vals_fg.push_back(I[gc_ix(i)]);
            else               vals_bg.push_back(I[gc_ix(i)]);
        }
        if (vals_fg.empty() || vals_bg.empty()) break;  // no model left to fit

        const GcMixture mix_fg = gc_fit_mixture(vals_fg, n_components);
        const GcMixture mix_bg = gc_fit_mixture(vals_bg, n_components);

        GcMaxFlow g(N, static_cast<int>(edges.size()));
        for (int i = 0; i < N; ++i) {
            double d_fg = 0.0, d_bg = 0.0;
            if (!inside[gc_ix(i)]) {
                d_fg = K;  // hard background outside the rectangle
                d_bg = 0.0;
            } else {
                const double x = I[gc_ix(i)];
                d_fg = -std::log(std::max(mix_fg.density(x), k_gc_min_density));
                d_bg = -std::log(std::max(mix_bg.density(x), k_gc_min_density));
            }
            const double m = std::min(d_fg, d_bg);
            g.add_terminal(i, d_bg - m, d_fg - m);
        }
        for (const GcEdge& e : edges) g.add_edge(e.p, e.q, e.w);

        g.finalize();
        (void)g.solve();  // only the cut is wanted here, not its value
        const std::vector<char> s = g.source_side();

        bool changed = false;
        for (int i = 0; i < N; ++i) {
            const char next_lab = (inside[gc_ix(i)] && s[gc_ix(i)]) ? 1 : 0;
            if (next_lab != lab[gc_ix(i)]) changed = true;
            lab[gc_ix(i)] = next_lab;
        }
        if (!changed) break;  // converged
    }

    for (int i = 0; i < N; ++i) out.data[gc_ix(i)] = lab[gc_ix(i)] ? 1.f : 0.f;
    return out;
}

// ========================== Histogram ==========================

std::vector<int> imhist(const Image& img, int nbins) {
    auto g=img.channels>1?rgb2gray(img):img;
    std::vector<int> h(nbins,0);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        int bin=(int)(g.at(r,c,0)*(nbins-1));
        bin=std::min(std::max(bin,0),nbins-1);
        h[bin]++;
    }
    return h;
}

Image histeq(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    auto h=imhist(g,256); int N=img.rows*img.cols;
    std::vector<float> cdf(256,0); cdf[0]=(float)h[0]/N;
    for (int i=1;i<256;++i) cdf[i]=cdf[i-1]+(float)h[i]/N;
    Image out(img.rows,img.cols,1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        int bin=(int)(g.at(r,c,0)*255);
        bin=std::min(std::max(bin,0),255);
        out.at(r,c,0)=cdf[bin];
    }
    return out;
}

Image imadjust(const Image& img, float in_lo, float in_hi, float out_lo, float out_hi) {
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float v=(img.at(r,c,ch)-in_lo)/(in_hi-in_lo+1e-12f);
        v=std::min(std::max(v,0.f),1.f)*( out_hi-out_lo)+out_lo;
        out.at(r,c,ch)=v;
    }
    return out;
}

namespace {

// Locate the pair of interpolation-node indices bracketing `pos`, plus the
// blend weight toward the second node. Positions outside [centers.front(),
// centers.back()] clamp to the nearest edge node (weight 0).
void clahe_bracket(float pos, const std::vector<float>& centers, int& i0, int& i1, float& w) {
    int n=(int)centers.size();
    if (n<=1) { i0=i1=0; w=0.f; return; }
    if (pos<=centers[0]) { i0=i1=0; w=0.f; return; }
    if (pos>=centers[n-1]) { i0=i1=n-1; w=0.f; return; }
    int i=(int)(std::upper_bound(centers.begin(),centers.end(),pos)-centers.begin())-1;
    i=std::min(std::max(i,0),n-2);
    i0=i; i1=i+1;
    w=(pos-centers[i0])/(centers[i1]-centers[i0]+1e-12f);
}

} // namespace

Image adapthisteq(const Image& img, int tile_size, float clip_limit) {
    if (img.empty()) return Image();
    auto g=img.channels>1?rgb2gray(img):img;
    const int R=g.rows, C=g.cols, nbins=256;

    int ts=std::max(1,tile_size);
    int n_tr=std::max(1,(R+ts-1)/ts);
    int n_tc=std::max(1,(C+ts-1)/ts);

    std::vector<int> row_bound(n_tr+1), col_bound(n_tc+1);
    for (int i=0;i<=n_tr;++i) row_bound[i]=std::min(R,i*ts);
    for (int j=0;j<=n_tc;++j) col_bound[j]=std::min(C,j*ts);

    std::vector<float> center_r(n_tr), center_c(n_tc);
    for (int ti=0;ti<n_tr;++ti) center_r[ti]=0.5f*(row_bound[ti]+row_bound[ti+1]-1);
    for (int tj=0;tj<n_tc;++tj) center_c[tj]=0.5f*(col_bound[tj]+col_bound[tj+1]-1);

    // Per-tile clipped-histogram CDF mapping tables (bin -> equalised value in [0,1]).
    std::vector<std::vector<float>> tile_map(n_tr*n_tc, std::vector<float>(nbins,0.f));
    for (int ti=0;ti<n_tr;++ti) for (int tj=0;tj<n_tc;++tj) {
        int r0=row_bound[ti], r1=row_bound[ti+1];
        int c0=col_bound[tj], c1=col_bound[tj+1];
        int N=(r1-r0)*(c1-c0);
        if (N<=0) continue;
        std::vector<float> hist(nbins,0.f);
        for (int r=r0;r<r1;++r) for (int c=c0;c<c1;++c) {
            int bin=(int)(g.at(r,c,0)*255.f);
            bin=std::min(std::max(bin,0),255);
            hist[bin]+=1.f;
        }
        float clim=std::max(1.f,clip_limit*(float)N);
        float excess=0.f;
        for (int i=0;i<nbins;++i) if (hist[i]>clim) { excess+=hist[i]-clim; hist[i]=clim; }
        // Redistribute the clipped-off mass uniformly (in exact fractional
        // amounts, so no bin is arbitrarily favoured) over all bins,
        // re-clipping any bin pushed back over the limit, for a bounded
        // number of rounds until the leftover excess is negligible.
        for (int iter=0;iter<20 && excess>1e-6f;++iter) {
            float add=excess/(float)nbins;
            excess=0.f;
            for (int i=0;i<nbins;++i) {
                hist[i]+=add;
                if (hist[i]>clim) { excess+=hist[i]-clim; hist[i]=clim; }
            }
        }
        float total=0.f;
        for (int i=0;i<nbins;++i) total+=hist[i];
        auto& map=tile_map[ti*n_tc+tj];
        float cum=0.f;
        float inv_total=total>0.f?1.f/total:0.f;
        for (int i=0;i<nbins;++i) { cum+=hist[i]; map[i]=cum*inv_total; }
    }

    // Precompute per-row/per-column bracketing tile indices and blend weights.
    std::vector<int> ri0(R), ri1(R); std::vector<float> rw(R);
    for (int r=0;r<R;++r) clahe_bracket((float)r, center_r, ri0[r], ri1[r], rw[r]);
    std::vector<int> ci0(C), ci1(C); std::vector<float> cw(C);
    for (int c=0;c<C;++c) clahe_bracket((float)c, center_c, ci0[c], ci1[c], cw[c]);

    Image out(R,C,1);
    for (int r=0;r<R;++r) {
        int t0=ri0[r], t1=ri1[r]; float wr=rw[r];
        for (int c=0;c<C;++c) {
            int s0=ci0[c], s1=ci1[c]; float wc=cw[c];
            int bin=(int)(g.at(r,c,0)*255.f);
            bin=std::min(std::max(bin,0),255);
            const auto& m00=tile_map[t0*n_tc+s0];
            const auto& m01=tile_map[t0*n_tc+s1];
            const auto& m10=tile_map[t1*n_tc+s0];
            const auto& m11=tile_map[t1*n_tc+s1];
            float top=m00[bin]*(1.f-wc)+m01[bin]*wc;
            float bot=m10[bin]*(1.f-wc)+m11[bin]*wc;
            out.at(r,c,0)=top*(1.f-wr)+bot*wr;
        }
    }
    return out;
}

// ========================== DFT Magnitude ==========================

Image dft_magnitude(const Image& img) {
    auto g=img.channels>1?rgb2gray(img):img;
    int R=g.rows, C=g.cols;
    using Cplx=std::complex<double>;
    // Intentional O(N^2) naive DFT reference path; use ms::fft for production sizes.
    std::vector<Cplx> F(R*C,0);
    for (int u=0;u<R;++u) for (int v=0;v<C;++v) {
        Cplx sum=0;
        for (int r=0;r<R;++r) for (int c=0;c<C;++c)
            sum+=(double)g.at(r,c,0)*std::exp(Cplx(0,-2*M_PI*(u*r/(double)R+v*c/(double)C)));
        F[u*C+v]=sum;
    }
    Image out(R,C,1);
    for (int r=0;r<R;++r) for (int c=0;c<C;++c)
        out.at(r,c,0)=(float)std::abs(F[r*C+c]);
    return out;
}

// ========================== Radon ==========================

std::vector<std::vector<float>> radon(const Image& img, const std::vector<float>& theta_deg) {
    auto g=img.channels>1?rgb2gray(img):img;
    int R=g.rows, C=g.cols;
    int n_theta=theta_deg.size(), n_proj=std::max(R,C);
    std::vector<std::vector<float>> sinogram(n_theta, std::vector<float>(n_proj,0));
    float cx=C/2.f, cy=R/2.f;
    for (int ti=0;ti<n_theta;++ti) {
        float th=theta_deg[ti]*(float)(M_PI/180.0);
        float costh=std::cos(th), sinth=std::sin(th);
        for (int ri=0;ri<n_proj;++ri) {
            float t=ri-n_proj/2.f;
            float sum=0; int cnt=0;
            for (int s=-(int)(std::max(R,C)/2);s<=(int)(std::max(R,C)/2);++s) {
                float x=t*costh-s*sinth+cx;
                float y=t*sinth+s*costh+cy;
                if (x>=0&&x<C-1&&y>=0&&y<R-1) { sum+=bilinear_sample(g,y,x,0); cnt++; }
            }
            sinogram[ti][ri]=cnt>0?sum/cnt:0.f;
        }
    }
    return sinogram;
}

static float sample_sinogram(const std::vector<float>& row, float ri_f) {
    int n=(int)row.size();
    if (n==0) return 0.f;
    if (ri_f<=0.f) return row[0];
    if (ri_f>=n-1) return row[n-1];
    int i0=(int)ri_f;
    float frac=ri_f-i0;
    return row[i0]*(1.f-frac)+row[i0+1]*frac;
}

// Unfiltered backprojection: inverse of radon() using the same projection convention.
Image iradon(const std::vector<std::vector<float>>& sinogram,
             const std::vector<float>& theta_deg) {
    if (sinogram.empty()||sinogram[0].empty()||theta_deg.empty())
        return Image();
    int n_theta=(int)sinogram.size(), n_proj=(int)sinogram[0].size();
    int R=n_proj, C=n_proj;
    Image out(R,C,1,0.f);
    float cx=C/2.f, cy=R/2.f;
    float scale=(float)(M_PI/(2.0*theta_deg.size()));
    for (int r=0;r<R;++r) for (int c=0;c<C;++c) {
        float sum=0;
        for (int ti=0;ti<n_theta;++ti) {
            float th=theta_deg[ti]*(float)(M_PI/180.0);
            float costh=std::cos(th), sinth=std::sin(th);
            float t=(c-cx)*costh+(r-cy)*sinth;
            float ri_f=t+n_proj/2.f;
            sum+=sample_sinogram(sinogram[ti], ri_f);
        }
        out.at(r,c,0)=sum*scale;
    }
    return out;
}

// ========================== Hough Transform ==========================

std::vector<HoughLine> hough_lines(const Image& img, double edge_threshold,
                                    int n_theta, int n_rho, int vote_threshold) {
    if (img.empty()||n_theta<=0||n_rho<=0) return {};
    auto g=img.channels>1?rgb2gray(img):img;
    int width=g.cols, height=g.rows;
    double rho_max=std::sqrt((double)width*width+(double)height*height);

    std::vector<double> costab(n_theta), sintab(n_theta);
    for (int ti=0;ti<n_theta;++ti) {
        double theta=ti*M_PI/n_theta;
        costab[ti]=std::cos(theta); sintab[ti]=std::sin(theta);
    }

    std::vector<std::vector<int>> acc(n_rho, std::vector<int>(n_theta,0));
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        if ((double)g.at(y,x,0)<=edge_threshold) continue;
        for (int ti=0;ti<n_theta;++ti) {
            double rho=x*costab[ti]+y*sintab[ti];
            int rho_bin=(int)std::lround((rho+rho_max)/(2.0*rho_max)*(n_rho-1));
            rho_bin=std::clamp(rho_bin,0,n_rho-1);
            acc[rho_bin][ti]++;
        }
    }

    std::vector<HoughLine> lines;
    for (int rb=0;rb<n_rho;++rb) for (int tb=0;tb<n_theta;++tb) {
        int votes=acc[rb][tb];
        if (votes<vote_threshold) continue;
        bool peak=true;
        for (int drb=-1;drb<=1&&peak;++drb) for (int dtb=-1;dtb<=1&&peak;++dtb) {
            if (drb==0&&dtb==0) continue;
            int nr=rb+drb, nt=tb+dtb;
            if (nr<0||nr>=n_rho||nt<0||nt>=n_theta) continue;  // no wraparound
            if (acc[nr][nt]>=votes) peak=false;
        }
        if (!peak) continue;
        double rho=(n_rho>1) ? (-rho_max+rb*(2.0*rho_max)/(n_rho-1)) : 0.0;
        double theta=tb*M_PI/n_theta;
        lines.push_back({rho,theta,votes});
    }
    std::sort(lines.begin(),lines.end(),
              [](const HoughLine& a, const HoughLine& b){ return a.votes>b.votes; });
    return lines;
}

std::vector<HoughCircle> hough_circles(const Image& img, double edge_threshold,
                                        double r_min, double r_max,
                                        int r_step, int vote_threshold) {
    if (img.empty()||r_step<=0||r_min>r_max) return {};
    auto g=img.channels>1?rgb2gray(img):img;
    int width=g.cols, height=g.rows;

    std::vector<int> radii;
    for (int r=(int)std::ceil(r_min); r<=(int)std::floor(r_max); r+=r_step)
        radii.push_back(r);
    if (radii.empty()) return {};

    int n_r=(int)radii.size();
    std::vector<std::vector<std::vector<int>>> acc(
        n_r, std::vector<std::vector<int>>(height, std::vector<int>(width,0)));

    for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        if ((double)g.at(y,x,0)<=edge_threshold) continue;
        for (int ri=0;ri<n_r;++ri) {
            int rad=radii[ri];
            int n_angles=std::max(8,(int)std::lround(2.0*M_PI*rad));
            for (int ai=0;ai<n_angles;++ai) {
                double theta=2.0*M_PI*ai/n_angles;
                int cx=(int)std::lround(x+rad*std::cos(theta));
                int cy=(int)std::lround(y+rad*std::sin(theta));
                if (cx>=0&&cx<width&&cy>=0&&cy<height)
                    acc[ri][cy][cx]++;
            }
        }
    }

    std::vector<HoughCircle> circles;
    for (int ri=0;ri<n_r;++ri) for (int cy=0;cy<height;++cy) for (int cx=0;cx<width;++cx) {
        int votes=acc[ri][cy][cx];
        if (votes<vote_threshold) continue;
        bool peak=true;
        for (int dri=-1;dri<=1&&peak;++dri) for (int dcy=-1;dcy<=1&&peak;++dcy)
            for (int dcx=-1;dcx<=1&&peak;++dcx) {
                if (dri==0&&dcy==0&&dcx==0) continue;
                int nri=ri+dri, ncy=cy+dcy, ncx=cx+dcx;
                if (nri<0||nri>=n_r) continue;
                if (ncy<0||ncy>=height||ncx<0||ncx>=width) continue;
                if (acc[nri][ncy][ncx]>=votes) peak=false;
            }
        if (!peak) continue;
        circles.push_back({(double)cx,(double)cy,(double)radii[ri],votes});
    }
    std::sort(circles.begin(),circles.end(),
              [](const HoughCircle& a, const HoughCircle& b){ return a.votes>b.votes; });
    return circles;
}

// ========================== Harris Corner Detector ==========================

static std::tuple<Image,Image,Image> structure_tensor_smoothed(const Image& g) {
    auto [gx,gy]=sobel_xy(g);
    Image Ixx(g.rows,g.cols,1), Ixy(g.rows,g.cols,1), Iyy(g.rows,g.cols,1);
    for (int r=0;r<g.rows;++r) for (int c=0;c<g.cols;++c) {
        float x=gx.at(r,c,0), y=gy.at(r,c,0);
        Ixx.at(r,c,0)=x*x; Ixy.at(r,c,0)=x*y; Iyy.at(r,c,0)=y*y;
    }
    return {imgaussfilt(Ixx,1.5f), imgaussfilt(Ixy,1.5f), imgaussfilt(Iyy,1.5f)};
}

static float min_eigenvalue_2x2(float a, float b, float d) {
    float trace=a+d;
    float det=a*d-b*b;
    float disc=trace*trace-4.f*det;
    if (disc<0.f) disc=0.f;
    return (trace-std::sqrt(disc))/2.f;
}

std::vector<KeyPoint> harris(const Image& img, float k, float threshold) {
    auto g=img.channels>1?rgb2gray(img):img;
    auto [sxx,sxy,syy]=structure_tensor_smoothed(g);
    Image R_img(img.rows,img.cols,1);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) {
        float a=sxx.at(r,c,0), b=sxy.at(r,c,0), d=syy.at(r,c,0);
        R_img.at(r,c,0)=(a*d-b*b)-k*(a+d)*(a+d);
    }
    float mx=0;
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) mx=std::max(mx,R_img.at(r,c,0));
    std::vector<KeyPoint> kps;
    for (int r=1;r<img.rows-1;++r) for (int c=1;c<img.cols-1;++c) {
        float v=R_img.at(r,c,0);
        if (v>threshold*mx) {
            bool peak=true;
            for (int dr=-1;dr<=1&&peak;++dr) for (int dc=-1;dc<=1&&peak;++dc)
                if (R_img.at(r+dr,c+dc,0)>v) peak=false;
            if (peak) kps.push_back({(float)c,(float)r,v});
        }
    }
    return kps;
}

std::vector<KeyPoint> shi_tomasi(const Image& img, int n, float quality_level) {
    auto g=img.channels>1?rgb2gray(img):img;
    auto [sxx,sxy,syy]=structure_tensor_smoothed(g);
    Image R_img(g.rows,g.cols,1);
    for (int r=0;r<g.rows;++r) for (int c=0;c<g.cols;++c) {
        float a=sxx.at(r,c,0), b=sxy.at(r,c,0), d=syy.at(r,c,0);
        R_img.at(r,c,0)=min_eigenvalue_2x2(a,b,d);
    }
    float mx=0;
    for (int r=0;r<g.rows;++r) for (int c=0;c<g.cols;++c) mx=std::max(mx,R_img.at(r,c,0));
    std::vector<KeyPoint> kps;
    for (int r=1;r<g.rows-1;++r) for (int c=1;c<g.cols-1;++c) {
        float v=R_img.at(r,c,0);
        if (v>quality_level*mx) {
            bool peak=true;
            for (int dr=-1;dr<=1&&peak;++dr) for (int dc=-1;dc<=1&&peak;++dc)
                if (R_img.at(r+dr,c+dc,0)>v) peak=false;
            if (peak) kps.push_back({(float)c,(float)r,v});
        }
    }
    std::sort(kps.begin(),kps.end(),[](const KeyPoint& a,const KeyPoint& b){return a.response>b.response;});
    if ((int)kps.size()>n) kps.resize(n);
    return kps;
}

// ========================== Connected Components ==========================

std::vector<std::vector<int>> label_components(const Image& bw) {
    int R=bw.rows, C=bw.cols;
    std::vector<std::vector<int>> labels(R,std::vector<int>(C,-1));
    int id=0;
    for (int r=0;r<R;++r) for (int c=0;c<C;++c) {
        if (bw.at(r,c,0)<0.5f||labels[r][c]>=0) continue;
        // BFS
        std::vector<std::pair<int,int>> q={{r,c}}; labels[r][c]=id;
        for (size_t qi=0;qi<q.size();++qi) {
            auto [qr,qc]=q[qi];
            int dr[]={-1,1,0,0},dc[]={0,0,-1,1};
            for (int d=0;d<4;++d) {
                int nr=qr+dr[d],nc=qc+dc[d];
                if (nr>=0&&nr<R&&nc>=0&&nc<C&&bw.at(nr,nc,0)>=0.5f&&labels[nr][nc]<0)
                { labels[nr][nc]=id; q.push_back({nr,nc}); }
            }
        }
        ++id;
    }
    return labels;
}

int count_components(const Image& bw) {
    auto labels=label_components(bw);
    int mx=-1;
    for (auto& row:labels) for (int v:row) mx=std::max(mx,v);
    return mx+1;
}

// ========================== Local Features (FAST / ORB / SIFT) ==========================

namespace {

// Grayscale view used by every feature detector below. Unlike the module's
// usual `img.channels > 1 ? rgb2gray(img) : img` idiom this is safe for
// 2-channel images (rgb2gray would read channel 2 out of bounds).
Image feature_gray(const Image& img) {
    if (img.empty() || img.rows <= 0 || img.cols <= 0 || img.channels <= 0) {
        return Image{};
    }
    if (img.channels == 1) {
        return img;
    }
    if (img.channels >= 3) {
        return rgb2gray(img);
    }
    Image g(img.rows, img.cols, 1);
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            g.at(r, c, 0) = img.at(r, c, 0);
        }
    }
    return g;
}

// ---------------------------------- FAST-9 ----------------------------------

constexpr int k_fast_ring     = 16;  // pixels on the Bresenham circle
constexpr int k_fast_arc      = 9;   // FAST-9: minimum contiguous arc length
constexpr int k_fast_border   = 3;   // circle radius

// The 16 circle offsets as {dcol, drow}. Index 0 is the pixel directly above
// the centre and the ring proceeds clockwise on screen, so antipodal pairs are
// (k, k + 8) and all index arithmetic is done modulo 16.
constexpr int k_fast_circle[k_fast_ring][2] = {
    { 0, -3}, { 1, -3}, { 2, -2}, { 3, -1},
    { 3,  0}, { 3,  1}, { 2,  2}, { 1,  3},
    { 0,  3}, {-1,  3}, {-2,  2}, {-3,  1},
    {-3,  0}, {-3, -1}, {-2, -2}, {-1, -3}
};

// Length of the longest circular run of `true` in f. `start_out` receives the
// smallest start index achieving that length, which is well defined: a proper
// sub-run can never tie its parent run's length, so the smallest achieving
// start is always a genuine run start (or 0 when all 16 entries are true).
int fast_longest_run(const std::array<bool, k_fast_ring>& f, int& start_out) {
    int best = 0;
    start_out = 0;
    for (int s = 0; s < k_fast_ring; ++s) {
        int n = 0;
        while (n < k_fast_ring && f[static_cast<std::size_t>((s + n) & 15)]) {
            ++n;
        }
        if (n > best) {
            best = n;
            start_out = s;
        }
    }
    return best;
}

// Shared FAST-9 core. `g` must be single channel. `border` is the number of
// untestable frame pixels; ORB passes its larger descriptor-patch border so
// that no keypoint it keeps can sit closer than 16 px to an edge. Corners come
// back in raster order.
std::vector<KeyPoint> fast_corners_impl(const Image& g, float threshold,
                                        int border, bool nonmax) {
    std::vector<KeyPoint> out;
    if (g.empty() || g.rows <= 0 || g.cols <= 0 || g.channels < 1) {
        return out;
    }
    const int b = std::max(border, k_fast_border);
    if (g.rows < 2 * b + 1 || g.cols < 2 * b + 1) {
        return out;
    }
    const float t = std::max(threshold, 0.f);

    // Pass 1: arc score for every testable pixel, 0 where no arc reaches 9.
    Image score(g.rows, g.cols, 1, 0.f);
    for (int r = b; r < g.rows - b; ++r) {
        for (int c = b; c < g.cols - b; ++c) {
            const float ip = g.at(r, c, 0);
            std::array<float, k_fast_ring> v{};
            std::array<bool, k_fast_ring> bright{};
            std::array<bool, k_fast_ring> dark{};
            for (int k = 0; k < k_fast_ring; ++k) {
                const std::size_t uk = static_cast<std::size_t>(k);
                const float pv = g.at(r + k_fast_circle[uk][1], c + k_fast_circle[uk][0], 0);
                v[uk] = pv;
                bright[uk] = (pv - ip) > t;
                dark[uk] = (ip - pv) > t;
            }
            int sb = 0;
            int sd = 0;
            const int lb = fast_longest_run(bright, sb);
            const int ld = fast_longest_run(dark, sd);
            float vb = 0.f;
            float vd = 0.f;
            if (lb >= k_fast_arc) {
                for (int n = 0; n < lb; ++n) {
                    const std::size_t k = static_cast<std::size_t>((sb + n) & 15);
                    vb += (v[k] - ip) - t;
                }
            }
            if (ld >= k_fast_arc) {
                for (int n = 0; n < ld; ++n) {
                    const std::size_t k = static_cast<std::size_t>((sd + n) & 15);
                    vd += (ip - v[k]) - t;
                }
            }
            score.at(r, c, 0) = std::max(vb, vd);
        }
    }

    // Pass 2: collect in raster order, optionally keeping 3x3 maxima only.
    for (int r = b; r < g.rows - b; ++r) {
        for (int c = b; c < g.cols - b; ++c) {
            const float v = score.at(r, c, 0);
            if (v <= 0.f) {
                continue;
            }
            if (nonmax) {
                bool keep = true;
                for (int dr = -1; dr <= 1 && keep; ++dr) {
                    for (int dc = -1; dc <= 1 && keep; ++dc) {
                        if (dr == 0 && dc == 0) {
                            continue;
                        }
                        const int nr = r + dr;
                        const int nc = c + dc;
                        const bool inside = nr >= 0 && nr < g.rows && nc >= 0 && nc < g.cols;
                        const float vn = inside ? score.at(nr, nc, 0) : 0.f;
                        if (vn > v) {
                            keep = false;
                        } else if (vn == v && (nr < r || (nr == r && nc < c))) {
                            keep = false;  // earlier in raster order wins the tie
                        }
                    }
                }
                if (!keep) {
                    continue;
                }
            }
            out.push_back(KeyPoint{static_cast<float>(c), static_cast<float>(r), v});
        }
    }
    return out;
}

// ----------------------------------- ORB ------------------------------------

constexpr int    k_orb_patch_half      = 15;   // 31x31 orientation patch
constexpr int    k_orb_border          = 16;   // patch half + 1
constexpr int    k_orb_min_side        = 2 * k_orb_border + 1;  // 33
constexpr int    k_orb_max_levels      = 16;
constexpr int    k_orb_pairs           = 256;
constexpr float  k_orb_desc_blur_sigma = 1.0f;  // -> 7-tap kernel, ORB's 7x7
constexpr double k_orb_pyr_blur        = 0.5;   // anti-alias sigma coefficient
constexpr double k_orb_pattern_sigma   = 6.2;   // = 31 / 5 (Calonder G-II)
constexpr std::uint64_t k_orb_pattern_seed = 0x9E3779B97F4A7C15ull;

// Row extents of the radius-15 disc: k_orb_umax[|dy|] = floor(sqrt(225 - dy*dy)).
// The patch therefore holds 31 + 2 * sum(2*u + 1) = 709 pixels.
constexpr int k_orb_umax[16] = {15, 14, 14, 14, 14, 14, 13, 13, 12, 12, 11, 10, 9, 7, 5, 0};

// SplitMix64: a 64-bit integer PRNG with no platform-dependent state. Used only
// to build the rBRIEF sampling pattern, and only through operations that are
// exact in IEEE-754 double, so the table is bit-identical everywhere.
struct SplitMix64 {
    std::uint64_t state = 0;

    std::uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Uniform in the open interval (0, 1): 53 mantissa bits, exactly representable.
    double next_unit() {
        return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
    // Irwin-Hall(12) - 6: mean 0, variance 1, support [-6, 6]. Box-Muller is
    // deliberately avoided because libm's log/cos are not bit-portable.
    double next_gauss() {
        double s = 0.0;
        for (int i = 0; i < 12; ++i) {
            s += next_unit();
        }
        return s - 6.0;
    }
};

// Sort key for FAST corners within one pyramid level: strongest first, then
// raster order, so truncating to the level's budget is reproducible.
bool orb_level_stronger(const KeyPoint& a, const KeyPoint& b) {
    if (a.response != b.response) {
        return a.response > b.response;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.x < b.x;
}

// Total order over emitted features: strongest first, ties by octave then
// raster order. Used by both ORB and SIFT so the output is fully reproducible.
bool feature_stronger(const FeatureKeyPoint& a, const FeatureKeyPoint& b) {
    if (a.response != b.response) {
        return a.response > b.response;
    }
    if (a.octave != b.octave) {
        return a.octave < b.octave;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    if (a.scale != b.scale) {
        return a.scale < b.scale;
    }
    return a.orientation < b.orientation;
}

// Intensity-centroid orientation over the 31x31 disc, on the unblurred level.
// The caller guarantees the patch is fully inside the image (ORB's detector
// border is 16 = patch half + 1). atan2(+0, +0) is +0, so a perfectly
// symmetric patch yields orientation exactly 0.
double orb_patch_orientation(const Image& level, int r, int c) {
    double m10 = 0.0;
    double m01 = 0.0;
    for (int dy = -k_orb_patch_half; dy <= k_orb_patch_half; ++dy) {
        const int u = k_orb_umax[static_cast<std::size_t>(std::abs(dy))];
        for (int dx = -u; dx <= u; ++dx) {
            const double v = static_cast<double>(level.at(r + dy, c + dx, 0));
            m10 += static_cast<double>(dx) * v;
            m01 += static_cast<double>(dy) * v;
        }
    }
    return std::atan2(m01, m10);
}

// Steered BRIEF against the fixed 256-pair pattern, read from a blurred copy
// of the level. Rotation can push a sample out to radius 15*sqrt(2) ~= 21.2,
// i.e. up to ~6 px past the detector border, so samples are clamped to the
// image (replicate, consistent with the rest of this module).
OrbDescriptor orb_descriptor(const Image& blurred, int r, int c, double theta) {
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const std::array<std::array<int, 4>, 256>& pat = orb_sampling_pattern();
    OrbDescriptor desc{};
    for (int j = 0; j < k_orb_pairs; ++j) {
        const std::array<int, 4>& q = pat[static_cast<std::size_t>(j)];
        const double q0 = static_cast<double>(q[0]);
        const double q1 = static_cast<double>(q[1]);
        const double q2 = static_cast<double>(q[2]);
        const double q3 = static_cast<double>(q[3]);
        const int x1 = static_cast<int>(std::lround(ct * q0 - st * q1));
        const int y1 = static_cast<int>(std::lround(st * q0 + ct * q1));
        const int x2 = static_cast<int>(std::lround(ct * q2 - st * q3));
        const int y2 = static_cast<int>(std::lround(st * q2 + ct * q3));
        const int r1 = std::clamp(r + y1, 0, blurred.rows - 1);
        const int c1 = std::clamp(c + x1, 0, blurred.cols - 1);
        const int r2 = std::clamp(r + y2, 0, blurred.rows - 1);
        const int c2 = std::clamp(c + x2, 0, blurred.cols - 1);
        if (blurred.at(r1, c1, 0) < blurred.at(r2, c2, 0)) {
            const std::size_t byte = static_cast<std::size_t>(j >> 3);
            desc[byte] = static_cast<std::uint8_t>(desc[byte] | (1u << (j & 7)));
        }
    }
    return desc;
}

// ----------------------------- descriptor matching ---------------------------

template <typename Desc, typename DistFn>
std::vector<DescriptorMatch> match_descriptors_impl(const std::vector<Desc>& query,
                                                    const std::vector<Desc>& train,
                                                    float ratio_threshold,
                                                    bool cross_check,
                                                    DistFn dist) {
    std::vector<DescriptorMatch> out;
    if (query.empty() || train.empty()) {
        return out;
    }
    const float inf = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < query.size(); ++i) {
        float d1 = inf;
        float d2 = inf;
        std::size_t j1 = 0;
        bool found = false;
        for (std::size_t j = 0; j < train.size(); ++j) {
            const float d = dist(query[i], train[j]);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                j1 = j;
                found = true;
            } else if (d < d2) {
                d2 = d;
            }
        }
        if (!found) {
            continue;
        }
        // Lowe's ratio test; skipped when there is no second neighbour at all.
        if (train.size() >= 2u && !(d1 < ratio_threshold * d2)) {
            continue;
        }
        if (cross_check) {
            float best = inf;
            std::size_t bi = 0;
            bool rev = false;
            for (std::size_t q = 0; q < query.size(); ++q) {
                const float d = dist(query[q], train[j1]);
                if (d < best) {
                    best = d;
                    bi = q;
                    rev = true;
                }
            }
            if (!rev || bi != i) {
                continue;
            }
        }
        DescriptorMatch m;
        m.query_index = static_cast<int>(i);
        m.train_index = static_cast<int>(j1);
        m.distance = d1;
        out.push_back(m);
    }
    return out;
}

// ----------------------------------- SIFT ------------------------------------

constexpr int    k_sift_border          = 5;     // extrema search margin
constexpr int    k_sift_max_interp      = 5;     // refinement iterations
constexpr int    k_sift_min_side        = 16;
constexpr int    k_sift_max_octaves     = 8;
constexpr double k_sift_init_sigma      = 0.5;   // assumed blur of the input
constexpr int    k_sift_ori_bins        = 36;
constexpr double k_sift_ori_sig_fctr    = 1.5;
constexpr double k_sift_ori_radius      = 4.5;   // = 3 * k_sift_ori_sig_fctr
constexpr double k_sift_ori_peak_ratio  = 0.8;
constexpr int    k_sift_descr_width     = 4;     // d
constexpr int    k_sift_descr_hist_bins = 8;     // n
constexpr double k_sift_descr_scl_fctr  = 3.0;
constexpr float  k_sift_descr_mag_thr   = 0.2f;
constexpr double k_sift_det_eps         = 1e-12;

// First and second differences of the DoG stack at one sample, in double.
struct SiftDeriv {
    double dx = 0.0, dy = 0.0, ds = 0.0;
    double dxx = 0.0, dyy = 0.0, dss = 0.0;
    double dxy = 0.0, dxs = 0.0, dys = 0.0;
};

SiftDeriv sift_derivatives(const Image& a, const Image& b, const Image& c, int lr, int lc) {
    SiftDeriv d;
    const double v = static_cast<double>(b.at(lr, lc, 0));
    d.dx = (static_cast<double>(b.at(lr, lc + 1, 0)) - static_cast<double>(b.at(lr, lc - 1, 0))) * 0.5;
    d.dy = (static_cast<double>(b.at(lr + 1, lc, 0)) - static_cast<double>(b.at(lr - 1, lc, 0))) * 0.5;
    d.ds = (static_cast<double>(c.at(lr, lc, 0)) - static_cast<double>(a.at(lr, lc, 0))) * 0.5;
    const double v2 = 2.0 * v;
    d.dxx = static_cast<double>(b.at(lr, lc + 1, 0)) + static_cast<double>(b.at(lr, lc - 1, 0)) - v2;
    d.dyy = static_cast<double>(b.at(lr + 1, lc, 0)) + static_cast<double>(b.at(lr - 1, lc, 0)) - v2;
    d.dss = static_cast<double>(c.at(lr, lc, 0)) + static_cast<double>(a.at(lr, lc, 0)) - v2;
    d.dxy = (static_cast<double>(b.at(lr + 1, lc + 1, 0)) - static_cast<double>(b.at(lr + 1, lc - 1, 0))
           - static_cast<double>(b.at(lr - 1, lc + 1, 0)) + static_cast<double>(b.at(lr - 1, lc - 1, 0))) * 0.25;
    d.dxs = (static_cast<double>(c.at(lr, lc + 1, 0)) - static_cast<double>(c.at(lr, lc - 1, 0))
           - static_cast<double>(a.at(lr, lc + 1, 0)) + static_cast<double>(a.at(lr, lc - 1, 0))) * 0.25;
    d.dys = (static_cast<double>(c.at(lr + 1, lc, 0)) - static_cast<double>(c.at(lr - 1, lc, 0))
           - static_cast<double>(a.at(lr + 1, lc, 0)) + static_cast<double>(a.at(lr - 1, lc, 0))) * 0.25;
    return d;
}

// Lowe's iterative sub-pixel / sub-scale refinement plus the low-contrast and
// edge-response rejections. On success (lr, lc, li) hold the integer sample the
// fit converged on and (xc, xr, xi) the fractional offsets from it.
bool sift_adjust_extremum(const std::vector<Image>& dogs, int layers, int& li, int& lr, int& lc,
                          double contrast_threshold, double edge_threshold,
                          double& xc, double& xr, double& xi, double& contr) {
    xc = 0.0;
    xr = 0.0;
    xi = 0.0;
    contr = 0.0;
    for (int iter = 0; iter < k_sift_max_interp; ++iter) {
        const Image& a = dogs[static_cast<std::size_t>(li - 1)];
        const Image& b = dogs[static_cast<std::size_t>(li)];
        const Image& c = dogs[static_cast<std::size_t>(li + 1)];
        const SiftDeriv d = sift_derivatives(a, b, c, lr, lc);

        const double det = d.dxx * (d.dyy * d.dss - d.dys * d.dys)
                         - d.dxy * (d.dxy * d.dss - d.dys * d.dxs)
                         + d.dxs * (d.dxy * d.dys - d.dyy * d.dxs);
        if (std::abs(det) < k_sift_det_eps) {
            return false;  // singular Hessian: a ridge, a plateau or a step edge
        }
        const double x0 = (d.dx * (d.dyy * d.dss - d.dys * d.dys)
                         - d.dxy * (d.dy * d.dss - d.dys * d.ds)
                         + d.dxs * (d.dy * d.dys - d.dyy * d.ds)) / det;
        const double x1 = (d.dxx * (d.dy * d.dss - d.ds * d.dys)
                         - d.dx * (d.dxy * d.dss - d.dys * d.dxs)
                         + d.dxs * (d.dxy * d.ds - d.dy * d.dxs)) / det;
        const double x2 = (d.dxx * (d.dyy * d.ds - d.dy * d.dys)
                         - d.dxy * (d.dxy * d.ds - d.dy * d.dxs)
                         + d.dx * (d.dxy * d.dys - d.dyy * d.dxs)) / det;
        xc = -x0;
        xr = -x1;
        xi = -x2;

        if (std::abs(xc) < 0.5 && std::abs(xr) < 0.5 && std::abs(xi) < 0.5) {
            const double t = d.dx * xc + d.dy * xr + d.ds * xi;
            contr = static_cast<double>(b.at(lr, lc, 0)) + t * 0.5;
            if (std::abs(contr) * static_cast<double>(layers) < contrast_threshold) {
                return false;
            }
            const double tr = d.dxx + d.dyy;
            const double det2 = d.dxx * d.dyy - d.dxy * d.dxy;
            if (det2 <= 0.0) {
                return false;  // saddle: the two principal curvatures disagree in sign
            }
            if (tr * tr * edge_threshold >= (edge_threshold + 1.0) * (edge_threshold + 1.0) * det2) {
                return false;  // edge-like: curvature ratio above the limit
            }
            return true;
        }
        if (std::abs(xc) > 1e6 || std::abs(xr) > 1e6 || std::abs(xi) > 1e6) {
            return false;
        }
        lc += static_cast<int>(std::lround(xc));
        lr += static_cast<int>(std::lround(xr));
        li += static_cast<int>(std::lround(xi));
        if (li < 1 || li > layers) {
            return false;
        }
        if (lc < k_sift_border || lc >= b.cols - k_sift_border) {
            return false;
        }
        if (lr < k_sift_border || lr >= b.rows - k_sift_border) {
            return false;
        }
    }
    return false;  // never converged
}

// 36-bin gradient orientation histogram over a square window, Gaussian
// weighted with sigma = 1.5 * scl_octv, then circularly smoothed with
// [1 4 6 4 1] / 16 (Lowe's window; the square support is what OpenCV uses too).
void sift_orientation_hist(const Image& g, int lr, int lc, double scl_octv,
                           std::array<double, k_sift_ori_bins>& smoothed) {
    std::array<double, k_sift_ori_bins> hist{};
    const int radius = static_cast<int>(std::lround(k_sift_ori_radius * scl_octv));
    const double sigma_w = k_sift_ori_sig_fctr * scl_octv;
    const double exp_scl = -1.0 / (2.0 * sigma_w * sigma_w);
    const double two_pi = 2.0 * M_PI;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int y = lr + dy;
        if (y <= 0 || y >= g.rows - 1) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            const int x = lc + dx;
            if (x <= 0 || x >= g.cols - 1) {
                continue;
            }
            const double gx = static_cast<double>(g.at(y, x + 1, 0)) - static_cast<double>(g.at(y, x - 1, 0));
            const double gy = static_cast<double>(g.at(y + 1, x, 0)) - static_cast<double>(g.at(y - 1, x, 0));
            const double w = std::exp(static_cast<double>(dx * dx + dy * dy) * exp_scl);
            const double mag = std::sqrt(gx * gx + gy * gy);
            double ori = std::atan2(gy, gx);
            if (ori < 0.0) {
                ori += two_pi;
            }
            int bin = static_cast<int>(std::lround(ori * static_cast<double>(k_sift_ori_bins) / two_pi))
                    % k_sift_ori_bins;
            if (bin < 0) {
                bin += k_sift_ori_bins;
            }
            hist[static_cast<std::size_t>(bin)] += w * mag;
        }
    }
    for (int j = 0; j < k_sift_ori_bins; ++j) {
        const std::size_t j0 = static_cast<std::size_t>(j);
        const std::size_t jm2 = static_cast<std::size_t>((j + k_sift_ori_bins - 2) % k_sift_ori_bins);
        const std::size_t jm1 = static_cast<std::size_t>((j + k_sift_ori_bins - 1) % k_sift_ori_bins);
        const std::size_t jp1 = static_cast<std::size_t>((j + 1) % k_sift_ori_bins);
        const std::size_t jp2 = static_cast<std::size_t>((j + 2) % k_sift_ori_bins);
        smoothed[j0] = (hist[jm2] + hist[jp2]) * (1.0 / 16.0)
                     + (hist[jm1] + hist[jp1]) * (4.0 / 16.0)
                     + hist[j0] * (6.0 / 16.0);
    }
}

// 4x4x8 trilinearly-interpolated gradient histogram descriptor, L2-normalised,
// clipped at 0.2 and re-normalised (Lowe section 6.1).
SiftDescriptor sift_descriptor(const Image& g, int pr, int pc, double scl_octv, double ori) {
    const int d = k_sift_descr_width;
    const int n = k_sift_descr_hist_bins;
    const double cs = std::cos(ori);
    const double sn = std::sin(ori);
    const double bins_per_rad = static_cast<double>(n) / (2.0 * M_PI);
    const double exp_scl = -1.0 / (static_cast<double>(d) * static_cast<double>(d) * 0.5);
    const double hist_width = k_sift_descr_scl_fctr * scl_octv;
    int radius = static_cast<int>(std::lround(hist_width * 1.4142135623730951 * (d + 1) * 0.5));
    const double diag = std::sqrt(static_cast<double>(g.rows) * static_cast<double>(g.rows)
                                + static_cast<double>(g.cols) * static_cast<double>(g.cols));
    radius = std::min(radius, static_cast<int>(diag));

    std::array<double, (k_sift_descr_width + 2) * (k_sift_descr_width + 2) * (k_sift_descr_hist_bins + 2)> hist{};

    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            // Rotate the sample into the feature's own frame and express it in
            // units of one spatial bin.
            const double c_rot = (static_cast<double>(j) * cs + static_cast<double>(i) * sn) / hist_width;
            const double r_rot = (-static_cast<double>(j) * sn + static_cast<double>(i) * cs) / hist_width;
            double rbin = r_rot + static_cast<double>(d) / 2.0 - 0.5;
            double cbin = c_rot + static_cast<double>(d) / 2.0 - 0.5;
            const int rr = pr + i;
            const int cc = pc + j;
            if (!(rbin > -1.0 && rbin < static_cast<double>(d) && cbin > -1.0 && cbin < static_cast<double>(d))) {
                continue;
            }
            if (!(rr > 0 && rr < g.rows - 1 && cc > 0 && cc < g.cols - 1)) {
                continue;
            }
            const double gx = static_cast<double>(g.at(rr, cc + 1, 0)) - static_cast<double>(g.at(rr, cc - 1, 0));
            const double gy = static_cast<double>(g.at(rr + 1, cc, 0)) - static_cast<double>(g.at(rr - 1, cc, 0));
            const double mag = std::sqrt(gx * gx + gy * gy)
                             * std::exp((c_rot * c_rot + r_rot * r_rot) * exp_scl);
            double obin = (std::atan2(gy, gx) - ori) * bins_per_rad;

            const int r0 = static_cast<int>(std::floor(rbin));
            const int c0 = static_cast<int>(std::floor(cbin));
            int o0 = static_cast<int>(std::floor(obin));
            rbin -= static_cast<double>(r0);
            cbin -= static_cast<double>(c0);
            obin -= static_cast<double>(o0);
            o0 = ((o0 % n) + n) % n;

            const double v_r1 = mag * rbin;
            const double v_r0 = mag - v_r1;
            const double v_11 = v_r1 * cbin;
            const double v_10 = v_r1 - v_11;
            const double v_01 = v_r0 * cbin;
            const double v_00 = v_r0 - v_01;
            const double a111 = v_11 * obin;
            const double a110 = v_11 - a111;
            const double a101 = v_10 * obin;
            const double a100 = v_10 - a101;
            const double a011 = v_01 * obin;
            const double a010 = v_01 - a011;
            const double a001 = v_00 * obin;
            const double a000 = v_00 - a001;

            const std::size_t idx = static_cast<std::size_t>(((r0 + 1) * (d + 2) + (c0 + 1)) * (n + 2) + o0);
            hist[idx] += a000;
            hist[idx + 1] += a001;
            hist[idx + static_cast<std::size_t>(n + 2)] += a010;
            hist[idx + static_cast<std::size_t>(n + 3)] += a011;
            hist[idx + static_cast<std::size_t>((d + 2) * (n + 2))] += a100;
            hist[idx + static_cast<std::size_t>((d + 2) * (n + 2) + 1)] += a101;
            hist[idx + static_cast<std::size_t>((d + 3) * (n + 2))] += a110;
            hist[idx + static_cast<std::size_t>((d + 3) * (n + 2) + 1)] += a111;
        }
    }

    SiftDescriptor dst{};
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const std::size_t idx = static_cast<std::size_t>(((i + 1) * (d + 2) + (j + 1)) * (n + 2));
            // Fold the circular orientation wrap-around back into bins 0 and 1.
            hist[idx] += hist[idx + static_cast<std::size_t>(n)];
            hist[idx + 1] += hist[idx + static_cast<std::size_t>(n + 1)];
            for (int k = 0; k < n; ++k) {
                dst[static_cast<std::size_t>((i * d + j) * n + k)] =
                    static_cast<float>(hist[idx + static_cast<std::size_t>(k)]);
            }
        }
    }

    double nrm = 0.0;
    for (const float v : dst) {
        nrm += static_cast<double>(v) * static_cast<double>(v);
    }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-12) {
        dst.fill(0.f);
        return dst;
    }
    const float inv = static_cast<float>(1.0 / nrm);
    for (float& v : dst) {
        v = std::min(v * inv, k_sift_descr_mag_thr);
    }
    nrm = 0.0;
    for (const float v : dst) {
        nrm += static_cast<double>(v) * static_cast<double>(v);
    }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-12) {
        dst.fill(0.f);
        return dst;
    }
    const float inv2 = static_cast<float>(1.0 / nrm);
    for (float& v : dst) {
        v *= inv2;
    }
    return dst;
}

}  // namespace

std::vector<KeyPoint> fast_corners(const Image& img, float threshold, bool nonmax_suppression) {
    const Image g = feature_gray(img);
    return fast_corners_impl(g, threshold, k_fast_border, nonmax_suppression);
}

const std::array<std::array<int, 4>, 256>& orb_sampling_pattern() {
    // Magic static: thread-safe, order-independent, no global constructor and
    // no dynamic allocation beyond the table itself.
    static const std::array<std::array<int, 4>, 256> pattern = [] {
        std::array<std::array<int, 4>, 256> p{};
        SplitMix64 rng{k_orb_pattern_seed};
        for (std::size_t j = 0; j < p.size(); ++j) {
            std::array<int, 4> q{0, 0, 1, 1};
            // All four coordinates are drawn before the test, so the PRNG
            // consumption is a fixed 48 next() calls per attempt regardless of
            // which coordinate went out of range. Measured: 9 redraws in total
            // across the whole table, so the bound is never approached.
            for (int attempt = 0; attempt < 1000; ++attempt) {
                std::array<int, 4> t{};
                bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    const long v = std::lround(k_orb_pattern_sigma * rng.next_gauss());
                    if (v < -k_orb_patch_half || v > k_orb_patch_half) {
                        ok = false;
                    }
                    t[static_cast<std::size_t>(k)] = static_cast<int>(v);
                }
                if (ok && !(t[0] == t[2] && t[1] == t[3])) {
                    q = t;
                    break;
                }
            }
            p[j] = q;
        }
        return p;
    }();
    return pattern;
}

OrbFeatures orb_detect_and_compute(const Image& img, int max_features, float fast_threshold,
                                   int n_levels, float scale_factor) {
    OrbFeatures out;
    if (max_features <= 0) {
        return out;
    }
    const Image g = feature_gray(img);
    if (g.empty() || g.rows < k_orb_min_side || g.cols < k_orb_min_side) {
        return out;
    }
    const int levels = std::clamp(n_levels, 1, k_orb_max_levels);
    const double sf = std::clamp(static_cast<double>(scale_factor), 1.01, 4.0);

    // Pyramid. Every level is resampled from level 0 rather than from its
    // predecessor, so interpolation error does not accumulate and the
    // level -> base mapping stays exactly (r * s, c * s).
    std::vector<Image> pyr;
    std::vector<double> pyr_scale;
    for (int i = 0; i < levels; ++i) {
        const double s = std::pow(sf, static_cast<double>(i));
        const int rows_i = 1 + static_cast<int>(std::floor(static_cast<double>(g.rows - 1) / s));
        const int cols_i = 1 + static_cast<int>(std::floor(static_cast<double>(g.cols - 1) / s));
        if (rows_i < k_orb_min_side || cols_i < k_orb_min_side) {
            break;
        }
        if (i == 0) {
            pyr.push_back(g);
        } else {
            const double sigma_i = k_orb_pyr_blur * std::sqrt(s * s - 1.0);
            const Image blurred = (sigma_i > 1e-3) ? imgaussfilt(g, static_cast<float>(sigma_i)) : g;
            Image level(rows_i, cols_i, 1);
            const float max_r = static_cast<float>(g.rows - 1);
            const float max_c = static_cast<float>(g.cols - 1);
            for (int r = 0; r < rows_i; ++r) {
                const float sr = std::min(static_cast<float>(static_cast<double>(r) * s), max_r);
                for (int c = 0; c < cols_i; ++c) {
                    const float sc = std::min(static_cast<float>(static_cast<double>(c) * s), max_c);
                    level.at(r, c, 0) = bilinear_sample(blurred, sr, sc, 0);
                }
            }
            pyr.push_back(std::move(level));
        }
        pyr_scale.push_back(s);
    }
    if (pyr.empty()) {
        return out;
    }

    // Feature budget per level: geometric in 1 / scale_factor, so coarse levels
    // are not starved by the much denser fine ones. The last level absorbs the
    // rounding remainder.
    const std::size_t n_lv = pyr.size();
    std::vector<int> budget(n_lv, 0);
    const double f = 1.0 / sf;
    double denom = 0.0;
    for (std::size_t i = 0; i < n_lv; ++i) {
        denom += std::pow(f, static_cast<double>(i));
    }
    int assigned = 0;
    for (std::size_t i = 0; i + 1 < n_lv; ++i) {
        const double share = static_cast<double>(max_features) * std::pow(f, static_cast<double>(i)) / denom;
        const int n_i = std::max(0, static_cast<int>(std::lround(share)));
        budget[i] = n_i;
        assigned += n_i;
    }
    budget[n_lv - 1] = std::max(0, max_features - assigned);

    for (std::size_t i = 0; i < n_lv; ++i) {
        if (budget[i] <= 0) {
            continue;
        }
        const Image& level = pyr[i];
        std::vector<KeyPoint> corners = fast_corners_impl(level, fast_threshold, k_orb_border, true);
        if (corners.empty()) {
            continue;
        }
        std::stable_sort(corners.begin(), corners.end(), orb_level_stronger);
        if (static_cast<int>(corners.size()) > budget[i]) {
            corners.resize(static_cast<std::size_t>(budget[i]));
        }
        // ORB smooths with a 7x7 kernel before sampling the BRIEF pairs so a
        // bit is not decided by a single noisy pixel; once per level, not once
        // per keypoint.
        const Image blurred = imgaussfilt(level, k_orb_desc_blur_sigma);
        const double s = pyr_scale[i];
        for (const KeyPoint& kp : corners) {
            const int r = static_cast<int>(kp.y);
            const int c = static_cast<int>(kp.x);
            const double theta = orb_patch_orientation(level, r, c);
            FeatureKeyPoint fk;
            fk.x = static_cast<float>(static_cast<double>(c) * s);
            fk.y = static_cast<float>(static_cast<double>(r) * s);
            fk.scale = static_cast<float>(s);
            fk.orientation = static_cast<float>(theta);
            fk.response = kp.response;
            fk.octave = static_cast<int>(i);
            out.keypoints.push_back(fk);
            out.descriptors.push_back(orb_descriptor(blurred, r, c, theta));
        }
    }

    // Total order over the merged levels, then the global cap.
    std::vector<std::size_t> order(out.keypoints.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&out](std::size_t a, std::size_t b) {
        return feature_stronger(out.keypoints[a], out.keypoints[b]);
    });
    std::vector<FeatureKeyPoint> kps;
    std::vector<OrbDescriptor> descs;
    const std::size_t cap = std::min(order.size(), static_cast<std::size_t>(max_features));
    kps.reserve(cap);
    descs.reserve(cap);
    for (std::size_t i = 0; i < cap; ++i) {
        kps.push_back(out.keypoints[order[i]]);
        descs.push_back(out.descriptors[order[i]]);
    }
    out.keypoints = std::move(kps);
    out.descriptors = std::move(descs);
    return out;
}

SiftFeatures sift_detect_and_compute(const Image& img, int max_features, int n_octave_layers,
                                     float contrast_threshold, float edge_threshold, float sigma) {
    SiftFeatures out;
    const Image g = feature_gray(img);
    if (g.empty()) {
        return out;
    }
    const int min_side = std::min(g.rows, g.cols);
    if (min_side < k_sift_min_side) {
        return out;
    }
    const int layers = std::clamp(n_octave_layers, 1, 8);
    const double sig0 = std::max(static_cast<double>(sigma), 0.01);
    const double contrast = static_cast<double>(contrast_threshold);
    const double edge = static_cast<double>(edge_threshold);
    const int n_oct_max = std::clamp(
        static_cast<int>(std::floor(std::log2(static_cast<double>(min_side)))) - 2,
        1, k_sift_max_octaves);

    // Incremental blurs: layer i of every octave is layer i-1 blurred by
    // sig[i], so that layer i carries an absolute blur of sig0 * 2^(i/layers).
    const int n_layers = layers + 3;
    const double kf = std::pow(2.0, 1.0 / static_cast<double>(layers));
    std::vector<double> sig(static_cast<std::size_t>(n_layers), 0.0);
    sig[0] = sig0;
    for (int i = 1; i < n_layers; ++i) {
        const double prev = sig0 * std::pow(kf, static_cast<double>(i - 1));
        const double cur = sig0 * std::pow(kf, static_cast<double>(i));
        sig[static_cast<std::size_t>(i)] = std::sqrt(std::max(cur * cur - prev * prev, 1e-8));
    }

    // The input is assumed to already carry k_sift_init_sigma of blur, so only
    // the difference is applied when building the base of octave 0.
    const double sig_diff = std::sqrt(std::max(sig0 * sig0 - k_sift_init_sigma * k_sift_init_sigma, 0.01));

    std::vector<std::vector<Image>> gpyr;
    std::vector<std::vector<Image>> dpyr;
    for (int o = 0; o < n_oct_max; ++o) {
        std::vector<Image> gl;
        gl.reserve(static_cast<std::size_t>(n_layers));
        if (o == 0) {
            gl.push_back(imgaussfilt(g, static_cast<float>(sig_diff)));
        } else {
            const Image& prev = gpyr[static_cast<std::size_t>(o - 1)][static_cast<std::size_t>(layers)];
            const int rows_o = std::max(1, prev.rows / 2);
            const int cols_o = std::max(1, prev.cols / 2);
            if (rows_o < 2 * k_sift_border + 2 || cols_o < 2 * k_sift_border + 2) {
                break;  // nothing detectable below this size
            }
            Image dn(rows_o, cols_o, 1);
            for (int r = 0; r < rows_o; ++r) {
                for (int c = 0; c < cols_o; ++c) {
                    dn.at(r, c, 0) = prev.at(2 * r, 2 * c, 0);
                }
            }
            gl.push_back(std::move(dn));
        }
        for (int i = 1; i < n_layers; ++i) {
            Image blurred = imgaussfilt(gl[static_cast<std::size_t>(i - 1)],
                                        static_cast<float>(sig[static_cast<std::size_t>(i)]));
            gl.push_back(std::move(blurred));
        }
        std::vector<Image> dl;
        dl.reserve(static_cast<std::size_t>(n_layers - 1));
        for (int i = 0; i + 1 < n_layers; ++i) {
            const Image& a = gl[static_cast<std::size_t>(i)];
            const Image& b = gl[static_cast<std::size_t>(i + 1)];
            Image diff(a.rows, a.cols, 1);
            for (std::size_t p = 0; p < diff.data.size(); ++p) {
                diff.data[p] = b.data[p] - a.data[p];
            }
            dl.push_back(std::move(diff));
        }
        gpyr.push_back(std::move(gl));
        dpyr.push_back(std::move(dl));
    }
    const int n_octaves = static_cast<int>(dpyr.size());

    // Extrema of the DoG stack, refined, oriented and described.
    std::vector<FeatureKeyPoint> raw_kps;
    std::vector<SiftDescriptor> raw_descs;
    const double prefilter = 0.5 * contrast / static_cast<double>(layers);
    std::array<double, k_sift_ori_bins> smoothed{};
    for (int o = 0; o < n_octaves; ++o) {
        const std::vector<Image>& dogs = dpyr[static_cast<std::size_t>(o)];
        const std::vector<Image>& gauss = gpyr[static_cast<std::size_t>(o)];
        for (int i = 1; i <= layers; ++i) {
            const Image& dc = dogs[static_cast<std::size_t>(i)];
            for (int r = k_sift_border; r < dc.rows - k_sift_border; ++r) {
                for (int c = k_sift_border; c < dc.cols - k_sift_border; ++c) {
                    const double val = static_cast<double>(dc.at(r, c, 0));
                    if (std::abs(val) <= prefilter) {
                        continue;
                    }
                    // Plateau-tolerant 3x3x3 test (>= / <=); the duplicates a
                    // plateau produces are removed after refinement.
                    bool is_max = val > 0.0;
                    bool is_min = val < 0.0;
                    for (int dl = -1; dl <= 1 && (is_max || is_min); ++dl) {
                        const Image& nd = dogs[static_cast<std::size_t>(i + dl)];
                        for (int dr = -1; dr <= 1 && (is_max || is_min); ++dr) {
                            for (int dc2 = -1; dc2 <= 1; ++dc2) {
                                if (dl == 0 && dr == 0 && dc2 == 0) {
                                    continue;
                                }
                                const double nv = static_cast<double>(nd.at(r + dr, c + dc2, 0));
                                if (is_max && val < nv) {
                                    is_max = false;
                                }
                                if (is_min && val > nv) {
                                    is_min = false;
                                }
                            }
                        }
                    }
                    if (!is_max && !is_min) {
                        continue;
                    }
                    int li = i;
                    int lr = r;
                    int lc = c;
                    double xc = 0.0;
                    double xr = 0.0;
                    double xi = 0.0;
                    double contr = 0.0;
                    if (!sift_adjust_extremum(dogs, layers, li, lr, lc, contrast, edge,
                                              xc, xr, xi, contr)) {
                        continue;
                    }
                    const double pw = static_cast<double>(1 << o);
                    const double scl_octv = sig0 * std::pow(2.0, (static_cast<double>(li) + xi)
                                                                  / static_cast<double>(layers));
                    const double bx = (static_cast<double>(lc) + xc) * pw;
                    const double by = (static_cast<double>(lr) + xr) * pw;

                    const Image& gimg = gauss[static_cast<std::size_t>(li)];
                    sift_orientation_hist(gimg, lr, lc, scl_octv, smoothed);
                    double max_val = 0.0;
                    for (const double v : smoothed) {
                        max_val = std::max(max_val, v);
                    }

                    FeatureKeyPoint fk;
                    fk.x = static_cast<float>(bx);
                    fk.y = static_cast<float>(by);
                    fk.scale = static_cast<float>(scl_octv * pw);
                    fk.response = static_cast<float>(std::abs(contr));
                    fk.octave = o;

                    const int pr = static_cast<int>(std::lround(static_cast<double>(lr) + xr));
                    const int pc = static_cast<int>(std::lround(static_cast<double>(lc) + xc));
                    if (max_val <= 0.0) {
                        // Flat neighbourhood: no dominant direction exists.
                        fk.orientation = 0.f;
                        raw_kps.push_back(fk);
                        raw_descs.push_back(sift_descriptor(gimg, pr, pc, scl_octv, 0.0));
                        continue;
                    }
                    const double peak_thr = k_sift_ori_peak_ratio * max_val;
                    for (int j = 0; j < k_sift_ori_bins; ++j) {
                        const double l = smoothed[static_cast<std::size_t>(
                            (j + k_sift_ori_bins - 1) % k_sift_ori_bins)];
                        const double rgt = smoothed[static_cast<std::size_t>((j + 1) % k_sift_ori_bins)];
                        const double cur = smoothed[static_cast<std::size_t>(j)];
                        if (!(cur > l && cur > rgt && cur >= peak_thr)) {
                            continue;
                        }
                        const double den = l - 2.0 * cur + rgt;
                        double bin_hat = (std::abs(den) < 1e-12)
                                             ? static_cast<double>(j)
                                             : static_cast<double>(j) + 0.5 * (l - rgt) / den;
                        if (bin_hat < 0.0) {
                            bin_hat += static_cast<double>(k_sift_ori_bins);
                        }
                        if (bin_hat >= static_cast<double>(k_sift_ori_bins)) {
                            bin_hat -= static_cast<double>(k_sift_ori_bins);
                        }
                        double angle = bin_hat * (2.0 * M_PI / static_cast<double>(k_sift_ori_bins));
                        if (angle > M_PI) {
                            angle -= 2.0 * M_PI;
                        }
                        fk.orientation = static_cast<float>(angle);
                        raw_kps.push_back(fk);
                        raw_descs.push_back(sift_descriptor(gimg, pr, pc, scl_octv, angle));
                    }
                }
            }
        }
    }

    // Total order, then plateau-duplicate suppression in that order so the
    // strongest representative of each cluster is the one kept.
    std::vector<std::size_t> order(raw_kps.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&raw_kps](std::size_t a, std::size_t b) {
        return feature_stronger(raw_kps[a], raw_kps[b]);
    });

    for (const std::size_t z : order) {
        const FeatureKeyPoint& a = raw_kps[z];
        bool dup = false;
        for (const FeatureKeyPoint& b : out.keypoints) {
            if (b.octave != a.octave) {
                continue;
            }
            const double pw = static_cast<double>(1 << a.octave);
            if (std::abs(static_cast<double>(a.x - b.x)) / pw >= 0.5) {
                continue;
            }
            if (std::abs(static_cast<double>(a.y - b.y)) / pw >= 0.5) {
                continue;
            }
            if (a.scale <= 0.f || b.scale <= 0.f) {
                continue;
            }
            if (std::abs(std::log(static_cast<double>(a.scale) / static_cast<double>(b.scale))) >= 0.05) {
                continue;
            }
            double da = static_cast<double>(a.orientation) - static_cast<double>(b.orientation);
            while (da > M_PI) {
                da -= 2.0 * M_PI;
            }
            while (da <= -M_PI) {
                da += 2.0 * M_PI;
            }
            if (std::abs(da) < 5.0 * M_PI / 180.0) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        out.keypoints.push_back(a);
        out.descriptors.push_back(raw_descs[z]);
        if (max_features > 0 && static_cast<int>(out.keypoints.size()) >= max_features) {
            break;
        }
    }
    return out;
}

int hamming_distance(const OrbDescriptor& a, const OrbDescriptor& b) {
    int d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::popcount(static_cast<std::uint8_t>(a[i] ^ b[i]));
    }
    return d;
}

float l2_distance(const SiftDescriptor& a, const SiftDescriptor& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}

std::vector<DescriptorMatch> match_descriptors(const std::vector<OrbDescriptor>& query,
                                               const std::vector<OrbDescriptor>& train,
                                               float ratio_threshold, bool cross_check) {
    return match_descriptors_impl(query, train, ratio_threshold, cross_check,
                                  [](const OrbDescriptor& a, const OrbDescriptor& b) {
                                      return static_cast<float>(hamming_distance(a, b));
                                  });
}

std::vector<DescriptorMatch> match_descriptors(const std::vector<SiftDescriptor>& query,
                                               const std::vector<SiftDescriptor>& train,
                                               float ratio_threshold, bool cross_check) {
    return match_descriptors_impl(query, train, ratio_threshold, cross_check,
                                  [](const SiftDescriptor& a, const SiftDescriptor& b) {
                                      return l2_distance(a, b);
                                  });
}

} // namespace image
} // namespace ms
