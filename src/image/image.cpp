#define _USE_MATH_DEFINES
#include "ms/image/image.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace image {

// ========================== Image ==========================

Image::Image(int r, int c, int ch, float fill)
    : rows(r), cols(c), channels(ch), data(r*c*ch, fill) {}

float& Image::at(int r, int c, int ch) { return data[(r*cols+c)*channels+ch]; }
float  Image::at(int r, int c, int ch) const { return data[(r*cols+c)*channels+ch]; }

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
    Image out(nr, nc, img.channels);
    float sr=(float)(img.rows-1)/(nr-1+1e-6f), sc=(float)(img.cols-1)/(nc-1+1e-6f);
    for (int r=0;r<nr;++r) for (int c=0;c<nc;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r,c,ch)=bilinear_sample(img, r*sr, c*sc, ch);
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

Image impad(const Image& img, int pad, float val) {
    Image out(img.rows+2*pad,img.cols+2*pad,img.channels,val);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch)
        out.at(r+pad,c+pad,ch)=img.at(r,c,ch);
    return out;
}

// ========================== Filtering ==========================

Image imfilter(const Image& img, const std::vector<std::vector<float>>& K) {
    int kr=K.size(), kc=K[0].size(), pr=kr/2, pc=kc/2;
    Image out(img.rows, img.cols, img.channels, 0.f);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float val=0;
        for (int dr=0;dr<kr;++dr) for (int dc=0;dc<kc;++dc) {
            int sr=std::min(std::max(r+dr-pr,0),img.rows-1);
            int sc=std::min(std::max(c+dc-pc,0),img.cols-1);
            val+=K[dr][dc]*img.at(sr,sc,ch);
        }
        out.at(r,c,ch)=val;
    }
    return out;
}

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

} // namespace

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

Image medfilt2(const Image& img, int ksize) {
    int half=ksize/2;
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        std::vector<float> vals;
        for (int dr=-half;dr<=half;++dr) for (int dc=-half;dc<=half;++dc) {
            int sr=std::min(std::max(r+dr,0),img.rows-1);
            int sc=std::min(std::max(c+dc,0),img.cols-1);
            vals.push_back(img.at(sr,sc,ch));
        }
        std::nth_element(vals.begin(),vals.begin()+vals.size()/2,vals.end());
        out.at(r,c,ch)=vals[vals.size()/2];
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
    const int half = std::max(1, static_cast<int>(2.f * sigma_s));
    const int ksize = 2 * half + 1;

    const float inv_2_sigma_s_sq = 1.f / (2.f * sigma_s * sigma_s);
    std::vector<float> spatial_weights(static_cast<std::size_t>(ksize * ksize));
    for (int dri = 0; dri < ksize; ++dri) {
        const int dr = dri - half;
        for (int dci = 0; dci < ksize; ++dci) {
            const int dc = dci - half;
            spatial_weights[static_cast<std::size_t>(dri * ksize + dci)] =
                std::exp(-(dr * dr + dc * dc) * inv_2_sigma_s_sq);
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
    int h=ksize/2;
    std::vector<std::vector<float>> K(ksize,std::vector<float>(ksize,1.f/(ksize*ksize)));
    return imfilter(img,K);
}

Image sharpen(const Image& img) {
    std::vector<std::vector<float>> K={{0,-1,0},{-1,5,-1},{0,-1,0}};
    return imfilter(img,K);
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
    auto g=img.channels>1?rgb2gray(img):img;
    std::vector<std::vector<float>> K={{0,1,0},{1,-4,1},{0,1,0}};
    return imfilter(g,K);
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

// LoG edge/blob detector: Gaussian smooth then Laplacian (sequential composition).
Image laplacian_of_gaussian(const Image& img, float sigma) {
    return laplacian(imgaussfilt(img, sigma));
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
    return threshold_binary(g, best_t/255.f);
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
    // Simple DFT O(N^2) for small images
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

} // namespace image
} // namespace ms
