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

Image imgaussfilt(const Image& img, float sigma) {
    int half=std::max(1,(int)(3*sigma));
    int ksize=2*half+1;
    std::vector<float> g1d(ksize);
    float sum=0;
    for (int i=0;i<ksize;++i) {
        float x=i-half;
        g1d[i]=std::exp(-x*x/(2*sigma*sigma));
        sum+=g1d[i];
    }
    for (auto& v:g1d) v/=sum;
    // Separable: horizontal then vertical
    Image tmp(img.rows, img.cols, img.channels, 0.f);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float val=0;
        for (int d=0;d<ksize;++d) {
            int cc=std::min(std::max(c+d-half,0),img.cols-1);
            val+=g1d[d]*img.at(r,cc,ch);
        }
        tmp.at(r,c,ch)=val;
    }
    Image out(img.rows, img.cols, img.channels, 0.f);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float val=0;
        for (int d=0;d<ksize;++d) {
            int rr=std::min(std::max(r+d-half,0),img.rows-1);
            val+=g1d[d]*tmp.at(rr,c,ch);
        }
        out.at(r,c,ch)=val;
    }
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
    int half=std::max(1,(int)(2*sigma_s));
    Image out(img.rows,img.cols,img.channels);
    for (int r=0;r<img.rows;++r) for (int c=0;c<img.cols;++c) for (int ch=0;ch<img.channels;++ch) {
        float num=0,den=0;
        float ic=img.at(r,c,ch);
        for (int dr=-half;dr<=half;++dr) for (int dc=-half;dc<=half;++dc) {
            int sr=std::min(std::max(r+dr,0),img.rows-1);
            int sc_=std::min(std::max(c+dc,0),img.cols-1);
            float sp=std::exp(-(dr*dr+dc*dc)/(2*sigma_s*sigma_s));
            float diff=img.at(sr,sc_,ch)-ic;
            float rp=std::exp(-diff*diff/(2*sigma_r*sigma_r));
            float w=sp*rp;
            num+=w*img.at(sr,sc_,ch); den+=w;
        }
        out.at(r,c,ch)=num/(den+1e-12f);
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
