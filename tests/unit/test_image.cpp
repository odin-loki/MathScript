#define _USE_MATH_DEFINES
#include "ms/image/image.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <set>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::image;

// ---- Construction ----

TEST(ImageBasic, Construction) {
    Image img(4,4,3);
    EXPECT_EQ(img.rows, 4);
    EXPECT_EQ(img.cols, 4);
    EXPECT_EQ(img.channels, 3);
    EXPECT_EQ((int)img.data.size(), 4*4*3);
}

TEST(ImageBasic, AtAccess) {
    Image img(3,3,1,0.f);
    img.at(1,1,0)=0.5f;
    EXPECT_FLOAT_EQ(img.at(1,1,0), 0.5f);
    EXPECT_FLOAT_EQ(img.at(0,0,0), 0.0f);
}

TEST(ImageBasic, EmptyAndFromData) {
    Image empty;
    EXPECT_TRUE(empty.empty());
    std::vector<float> d={1.f,0.f,0.f, 0.f,1.f,0.f};
    auto img=im_from_data(2,1,3,d);
    EXPECT_EQ(img.rows, 2);
    EXPECT_EQ(img.cols, 1);
    EXPECT_FLOAT_EQ(img.at(0,0,0), 1.f);
    EXPECT_FLOAT_EQ(img.at(1,0,1), 1.f);
}

// ---- Color ----

TEST(ImageColor, RGB2Gray) {
    Image img(2,2,3);
    img.at(0,0,0)=1.f; img.at(0,0,1)=0.f; img.at(0,0,2)=0.f;  // red
    auto g=rgb2gray(img);
    EXPECT_EQ(g.channels, 1);
    EXPECT_NEAR(g.at(0,0,0), 0.299f, 0.01f);
}

TEST(ImageColor, Gray2RGB) {
    Image g(2,2,1,0.5f);
    auto rgb=gray2rgb(g);
    EXPECT_EQ(rgb.channels, 3);
    EXPECT_FLOAT_EQ(rgb.at(0,0,0), 0.5f);
}

TEST(ImageColor, RGB_HSV_Roundtrip) {
    Image img(2,2,3);
    img.at(0,0,0)=0.8f; img.at(0,0,1)=0.2f; img.at(0,0,2)=0.4f;
    auto hsv=rgb2hsv(img);
    auto rgb=hsv2rgb(hsv);
    EXPECT_NEAR(rgb.at(0,0,0), 0.8f, 0.01f);
    EXPECT_NEAR(rgb.at(0,0,1), 0.2f, 0.01f);
    EXPECT_NEAR(rgb.at(0,0,2), 0.4f, 0.01f);
}

// ---- Geometric ----

TEST(ImageGeom, Resize) {
    Image img(4,4,1);
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) img.at(r,c,0)=r*0.1f+c*0.1f;
    auto s=imresize(img,2,2);
    EXPECT_EQ(s.rows, 2);
    EXPECT_EQ(s.cols, 2);
}

TEST(ImageGeom, Crop) {
    Image img(8,8,1,0.f);
    for (int r=2;r<6;++r) for (int c=2;c<6;++c) img.at(r,c,0)=1.f;
    auto crop=imcrop(img,2,2,6,6);
    EXPECT_EQ(crop.rows, 4);
    EXPECT_EQ(crop.cols, 4);
    EXPECT_FLOAT_EQ(crop.at(0,0,0), 1.f);
}

TEST(ImageGeom, Flip) {
    Image img(2,2,1,0.f);
    img.at(0,0,0)=1.f;
    auto fl=imflip(img,true);
    EXPECT_FLOAT_EQ(fl.at(0,1,0), 1.f);
}

TEST(ImageGeom, FlipVertical) {
    Image img(3,2,1,0.f);
    img.at(0,0,0)=1.f;
    img.at(2,0,0)=0.5f;
    auto fl=imflip(img,false);
    EXPECT_FLOAT_EQ(fl.at(2,0,0), 1.f);
    EXPECT_FLOAT_EQ(fl.at(0,0,0), 0.5f);
}

TEST(ImageGeom, Pad) {
    Image img(2,2,1,1.f);
    auto padded=impad(img,1,0.f);
    EXPECT_EQ(padded.rows, 4);
    EXPECT_FLOAT_EQ(padded.at(0,0,0), 0.f);
    EXPECT_FLOAT_EQ(padded.at(1,1,0), 1.f);
}

TEST(ImageGeom, Rotate90) {
    Image img(2,3,1,0.f);
    img.at(0,0,0)=1.f;
    auto rot=imrotate90(img);
    EXPECT_EQ(rot.rows, 3);
    EXPECT_EQ(rot.cols, 2);
    EXPECT_FLOAT_EQ(rot.at(2,0,0), 1.f);
}

// ---- Filtering ----

TEST(ImageFilter, ImfilterIdentity) {
    Image img(3,3,1,0.f);
    img.at(1,1,0)=1.f;
    std::vector<std::vector<float>> K={{0,0,0},{0,1,0},{0,0,0}};
    auto out=imfilter(img,K);
    EXPECT_FLOAT_EQ(out.at(1,1,0), 1.f);
    EXPECT_FLOAT_EQ(out.at(0,0,0), 0.f);
}

TEST(ImageFilter, GaussianBlur) {
    Image img(5,5,1,0.f);
    img.at(2,2,0)=1.f;
    auto blurred=imgaussfilt(img,1.0f);
    // Center should be less than 1 (spread out)
    EXPECT_LT(blurred.at(2,2,0), 1.f);
    EXPECT_GT(blurred.at(2,2,0), 0.1f);
}

TEST(ImageFilter, BoxFilter) {
    Image img(5,5,1,1.f);
    auto box=boxfilter(img,3);
    // Constant image stays constant
    EXPECT_NEAR(box.at(2,2,0), 1.f, 0.01f);
}

TEST(ImageFilter, MedianFilter) {
    Image img(5,5,1,0.f);
    img.at(2,2,0)=1.f;  // single spike
    auto med=medfilt2(img,3);
    // Spike removed by median
    EXPECT_NEAR(med.at(2,2,0), 0.f, 0.1f);
}

TEST(ImageFilter, MedianFiveByFive) {
    Image img(7,7,1,0.f);
    img.at(3,3,0)=1.f;
    auto med=medfilt2(img,5);
    EXPECT_EQ(med.rows, 7);
    EXPECT_NEAR(med.at(3,3,0), 0.f, 0.15f);
}

TEST(ImageFilter, Sharpen) {
    Image img(5,5,1,0.5f);
    auto s=sharpen(img);
    EXPECT_EQ(s.rows, 5);
}

TEST(ImageFilter, BilateralFinite) {
    Image img(5,5,1,0.5f);
    img.at(2,2,0)=1.f;
    auto out=bilateral(img,1.0f,0.1f);
    EXPECT_EQ(out.rows, 5);
    for (float v:out.data) EXPECT_TRUE(std::isfinite(v));
}

// ---- Edge Detection ----

TEST(ImageEdge, Sobel) {
    // Step edge
    Image img(5,5,1,0.f);
    for (int r=0;r<5;++r) for (int c=3;c<5;++c) img.at(r,c,0)=1.f;
    auto edges=sobel(img);
    EXPECT_EQ(edges.channels, 1);
    EXPECT_GT(edges.at(2,2,0), 0.0f);  // Edge at column 2-3
}

TEST(ImageEdge, SobelXYAndPrewitt) {
    Image img(5,5,1,0.f);
    for (int r=0;r<5;++r) for (int c=3;c<5;++c) img.at(r,c,0)=1.f;
    auto gx=sobel_x(img);
    auto gy=sobel_y(img);
    EXPECT_GT(std::abs(gx.at(2,2,0)), 0.0f);
    auto pw=prewitt(img);
    EXPECT_GT(pw.at(2,2,0), 0.0f);
    auto sc=scharr(img);
    EXPECT_GT(sc.at(2,2,0), 0.0f);
}

TEST(ImageEdge, Canny) {
    Image img(10,10,1,0.f);
    for (int r=0;r<10;++r) for (int c=5;c<10;++c) img.at(r,c,0)=1.f;
    auto edges=canny(img, 0.1f, 0.3f, 1.0f);
    EXPECT_EQ(edges.rows, 10);
    EXPECT_EQ(edges.cols, 10);
}

TEST(ImageEdge, CannyCustomSigma) {
    Image img(8,8,1,0.f);
    for (int r=0;r<8;++r) for (int c=4;c<8;++c) img.at(r,c,0)=1.f;
    auto edges=canny(img, 0.05f, 0.2f, 2.0f);
    EXPECT_EQ(edges.rows, 8);
    EXPECT_GT(edges.at(3,3,0), 0.0f);
}

TEST(ImageEdge, Laplacian) {
    Image img(5,5,1,0.f);
    img.at(2,2,0)=1.f;
    auto lap=laplacian(img);
    EXPECT_EQ(lap.rows, 5);
}

TEST(ImageEdge, RobertsFlat) {
    Image img(5,5,1,0.5f);
    auto edges=roberts(img);
    EXPECT_EQ(edges.channels, 1);
    for (float v:edges.data) EXPECT_NEAR(v, 0.f, 0.01f);
}

TEST(ImageEdge, RobertsStepEdge) {
    Image img(5,5,1,0.f);
    for (int r=0;r<5;++r) for (int c=3;c<5;++c) img.at(r,c,0)=1.f;
    auto edges=roberts(img);
    EXPECT_GT(edges.at(2,2,0), 0.0f);
    float row_max=0;
    for (int c=0;c<5;++c) row_max=std::max(row_max, edges.at(2,c,0));
    EXPECT_GT(row_max, 0.5f);
}

TEST(ImageEdge, RobertsHalfPlane) {
    Image img(6,6,1,0.f);
    for (int r=0;r<6;++r) for (int c=3;c<6;++c) img.at(r,c,0)=1.f;
    auto edges=roberts(img);
    float edge_sum=0;
    for (int r=0;r<6;++r) for (int c=0;c<6;++c) edge_sum+=edges.at(r,c,0);
    EXPECT_GT(edge_sum, 0.5f);
}

TEST(ImageEdge, RobertsGrayscaleFromRGB) {
    Image img(4,4,3,0.f);
    for (int r=0;r<4;++r) for (int c=2;c<4;++c) {
        img.at(r,c,0)=1.f; img.at(r,c,1)=1.f; img.at(r,c,2)=1.f;
    }
    auto edges=roberts(img);
    EXPECT_EQ(edges.channels, 1);
    EXPECT_GT(edges.at(2,1,0), 0.0f);
}

TEST(ImageEdge, LaplacianOfGaussianFlat) {
    Image img(7,7,1,0.4f);
    auto log=laplacian_of_gaussian(img, 1.0f);
    EXPECT_EQ(log.rows, 7);
    for (float v:log.data) EXPECT_NEAR(v, 0.f, 0.05f);
}

TEST(ImageEdge, LaplacianOfGaussianEdge) {
    Image img(9,9,1,0.f);
    for (int r=0;r<9;++r) for (int c=5;c<9;++c) img.at(r,c,0)=1.f;
    auto log=laplacian_of_gaussian(img, 0.8f);
    float edge_resp=0;
    for (int r=0;r<9;++r) {
        edge_resp=std::max(edge_resp, std::abs(log.at(r,4,0)));
        edge_resp=std::max(edge_resp, std::abs(log.at(r,5,0)));
    }
    EXPECT_GT(edge_resp, 0.01f);
}

TEST(ImageEdge, LaplacianOfGaussianSigmaAttenuates) {
    Image img(11,11,1,0.f);
    for (int r=0;r<11;++r) for (int c=6;c<11;++c) img.at(r,c,0)=1.f;
    auto log_small=laplacian_of_gaussian(img, 0.5f);
    auto log_large=laplacian_of_gaussian(img, 2.5f);
    float resp_small=0, resp_large=0;
    for (int r=0;r<11;++r) {
        resp_small=std::max(resp_small, std::abs(log_small.at(r,5,0)));
        resp_large=std::max(resp_large, std::abs(log_large.at(r,5,0)));
    }
    EXPECT_GT(resp_small, resp_large);
}

TEST(ImageEdge, LaplacianOfGaussianComposition) {
    Image img(5,5,1,0.f);
    img.at(2,2,0)=1.f;
    auto composed=laplacian(imgaussfilt(img, 1.0f));
    auto log=laplacian_of_gaussian(img, 1.0f);
    EXPECT_NEAR(log.at(2,2,0), composed.at(2,2,0), 1e-5f);
}

// ---- Morphology ----

TEST(ImageMorph, Dilate) {
    Image img(5,5,1,0.f);
    img.at(2,2,0)=1.f;
    auto d=imdilate(img,3);
    EXPECT_GT(d.at(2,2,0), 0.f);
    EXPECT_GT(d.at(1,1,0), 0.f);  // Should spread
}

TEST(ImageMorph, Erode) {
    Image img(5,5,1,1.f);
    img.at(0,0,0)=0.f;
    auto e=imerode(img,3);
    EXPECT_FLOAT_EQ(e.at(0,0,0), 0.f);
}

TEST(ImageMorph, OpenClose) {
    Image img(5,5,1,0.f);
    img.at(2,2,0)=1.f;  // single spike
    auto opened=imopen(img,3);
    EXPECT_NEAR(opened.at(2,2,0), 0.f, 0.1f);  // Spike removed
}

TEST(ImageMorph, CloseTopHatBothat) {
    Image img(7,7,1,0.f);
    for (int r=2;r<5;++r) for (int c=2;c<5;++c) img.at(r,c,0)=0.8f;
    auto closed=imclose(img,3);
    EXPECT_GT(closed.at(3,3,0), 0.5f);
    auto tophat=imtophat(img,3);
    EXPECT_GE(tophat.at(3,3,0), 0.f);
    auto bothat=imbothat(img,3);
    EXPECT_GE(bothat.at(0,0,0), 0.f);
}

TEST(ImageMorph, GradientMorphEdge) {
    Image img(9,9,1,0.f);
    for (int r=2;r<7;++r) for (int c=2;c<7;++c) img.at(r,c,0)=1.f;
    auto grad=imgradient_morph(img,3);
    EXPECT_GT(grad.at(2,4,0), 0.1f);
    EXPECT_GT(grad.at(4,2,0), 0.1f);
}

TEST(ImageMorph, GradientMorphInteriorExterior) {
    Image img(9,9,1,0.f);
    for (int r=2;r<7;++r) for (int c=2;c<7;++c) img.at(r,c,0)=1.f;
    auto grad=imgradient_morph(img,3);
    EXPECT_NEAR(grad.at(4,4,0), 0.f, 0.05f);
    EXPECT_NEAR(grad.at(0,0,0), 0.f, 0.05f);
}

TEST(ImageMorph, GradientMorphComposition) {
    Image img(7,7,1,0.f);
    for (int r=2;r<5;++r) for (int c=2;c<5;++c) img.at(r,c,0)=1.f;
    auto dil=imdilate(img,3), ero=imerode(img,3);
    auto grad=imgradient_morph(img,3);
    EXPECT_NEAR(grad.at(3,3,0), dil.at(3,3,0)-ero.at(3,3,0), 1e-5f);
}

// ---- Thresholding ----

TEST(ImageThresh, Binary) {
    Image img(2,2,1,0.f);
    img.at(0,0,0)=0.7f; img.at(0,1,0)=0.3f;
    auto t=threshold_binary(img,0.5f);
    EXPECT_FLOAT_EQ(t.at(0,0,0), 1.f);
    EXPECT_FLOAT_EQ(t.at(0,1,0), 0.f);
}

TEST(ImageThresh, Otsu) {
    Image img(4,4,1,0.f);
    for (int r=0;r<4;++r) for (int c=2;c<4;++c) img.at(r,c,0)=1.f;
    auto t=threshold_otsu(img);
    EXPECT_EQ(t.rows, 4);
    EXPECT_EQ(t.channels, 1);
}

// ---- Histogram ----

TEST(ImageHist, Uniform) {
    Image img(4,4,1,0.5f);
    auto h=imhist(img,256);
    EXPECT_EQ(h.size(), 256u);
    // All pixels at ~0.5 = bin 127
    EXPECT_GT(h[127], 0);
}

TEST(ImageHist, CustomBinCount) {
    Image img(2,2,1,0.f);
    img.at(0,0,0)=0.0f; img.at(0,1,0)=0.25f;
    img.at(1,0,0)=0.5f; img.at(1,1,0)=0.75f;
    auto h=imhist(img,4);
    EXPECT_EQ(h.size(), 4u);
    // imhist maps v in [0,1] to bin floor(v*(nbins-1))
    EXPECT_EQ(h[0], 2);  // 0.0 and 0.25 -> bin 0
    EXPECT_EQ(h[1], 1);  // 0.5
    EXPECT_EQ(h[2], 1);  // 0.75
    EXPECT_EQ(h[3], 0);
}

TEST(ImageHist, HistEq) {
    Image img(4,4,1,0.f);
    for (int i=0;i<8;++i) img.data[i]=0.1f*(i%4);
    auto eq=histeq(img);
    EXPECT_EQ(eq.rows, 4);
}

TEST(ImageHist, Imadjust) {
    Image img(2,2,1,0.f);
    img.at(0,0,0)=0.2f; img.at(0,1,0)=0.8f;
    auto adj=imadjust(img, 0.f, 1.f, 0.f, 1.f);
    EXPECT_NEAR(adj.at(0,0,0), 0.2f, 0.01f);
    EXPECT_NEAR(adj.at(0,1,0), 0.8f, 0.01f);
    auto bright=imadjust(img, 0.f, 1.f, 0.5f, 1.f);
    EXPECT_GT(bright.at(0,0,0), adj.at(0,0,0));
}

// ---- CLAHE (adaptive histogram equalisation) ----

static float value_range(const Image& img, int r0, int r1, int c0, int c1) {
    float mn=1e9f, mx=-1e9f;
    for (int r=r0;r<r1;++r) for (int c=c0;c<c1;++c) {
        float v=img.at(r,c,0);
        mn=std::min(mn,v); mx=std::max(mx,v);
    }
    return mx-mn;
}

TEST(ImageAdaptHistEq, LocalContrastBeatsGlobalHistEq) {
    // 32x32 image: a full-range ramp background (so the global histogram is
    // busy/dense) plus an 8x8 tile-aligned patch (rows/cols 8-15) with a
    // low-contrast checkerboard (values ~100/255 vs ~110/255), the classic
    // scenario where global histeq barely helps but CLAHE, which equalises
    // that tile using only its own local content, helps a lot.
    const int N=32;
    Image img(N,N,1,0.f);
    for (int r=0;r<N;++r) for (int c=0;c<N;++c) img.at(r,c,0)=(float)(r*N+c)/(float)(N*N-1);
    for (int r=8;r<16;++r) for (int c=8;c<16;++c)
        img.at(r,c,0)=((r+c)%2==0)?(100.f/255.f):(110.f/255.f);

    auto eq=histeq(img);
    auto clahe=adapthisteq(img, 8, 1.0f);

    // Compare near the checkerboard tile's center, where bilinear blending
    // is dominated by that tile's own (checkerboard-only) mapping.
    float range_orig=value_range(img, 10,14, 10,14);
    float range_histeq=value_range(eq, 10,14, 10,14);
    float range_clahe=value_range(clahe, 10,14, 10,14);

    EXPECT_GT(range_clahe, range_orig);
    EXPECT_GT(range_clahe, range_histeq);
}

TEST(ImageAdaptHistEq, ClipLimitMonotonicity) {
    // Single 8x8 tile filled entirely with a narrow noisy band (bins
    // 125-129), no dominant single spike -- the classic near-uniform
    // region where unclipped local equalisation over-amplifies tiny
    // differences. A lower clip_limit should visibly rein that in.
    Image img(8,8,1,0.f);
    int bins[8*8];
    int idx=0;
    for (int b=0;b<5;++b) {
        int count = (b<4)?13:12;
        for (int k=0;k<count;++k) bins[idx++]=125+b;
    }
    for (int i=0;i<64;++i) img.data[i]=(bins[i]+0.5f)/255.f;

    auto low=adapthisteq(img, 8, 0.05f);
    auto high=adapthisteq(img, 8, 0.5f);

    float range_low=value_range(low, 0,8, 0,8);
    float range_high=value_range(high, 0,8, 0,8);
    EXPECT_LT(range_low, range_high);
}

TEST(ImageAdaptHistEq, UniformImageUnchanged) {
    Image img(16,16,1,0.5f);
    auto out=adapthisteq(img, 4, 0.01f);
    for (float v:out.data) EXPECT_NEAR(v, 0.5f, 0.1f);
}

TEST(ImageAdaptHistEq, DimensionsAndRange) {
    Image img(20,24,1,0.f);
    for (int r=0;r<20;++r) for (int c=0;c<24;++c) img.at(r,c,0)=(float)((r*7+c*3)%256)/255.f;
    auto out=adapthisteq(img, 6, 0.1f);
    EXPECT_EQ(out.rows, 20);
    EXPECT_EQ(out.cols, 24);
    EXPECT_EQ(out.channels, 1);
    for (float v:out.data) { EXPECT_GE(v, 0.f); EXPECT_LE(v, 1.f); }
}

TEST(ImageAdaptHistEq, RgbChannelHandling) {
    Image img(8,8,3,0.f);
    for (int r=0;r<8;++r) for (int c=0;c<8;++c) {
        float v=(float)((r*8+c)%256)/255.f;
        img.at(r,c,0)=v; img.at(r,c,1)=v; img.at(r,c,2)=v;
    }
    auto out=adapthisteq(img, 4, 0.1f);
    EXPECT_EQ(out.channels, 1);
    EXPECT_EQ(out.rows, 8);
    EXPECT_EQ(out.cols, 8);
}

TEST(ImageAdaptHistEq, DegenerateTileSizeOne) {
    Image img(6,6,1,0.f);
    for (int r=0;r<6;++r) for (int c=0;c<6;++c) img.at(r,c,0)=(float)((r*6+c)%9)/8.f;
    auto out=adapthisteq(img, 1, 0.1f);
    EXPECT_EQ(out.rows, 6);
    EXPECT_EQ(out.cols, 6);
    for (float v:out.data) { EXPECT_TRUE(std::isfinite(v)); EXPECT_GE(v, 0.f); EXPECT_LE(v, 1.f); }
}

TEST(ImageAdaptHistEq, DegenerateTileSizeLargerThanImage) {
    Image img(6,6,1,0.f);
    for (int r=0;r<6;++r) for (int c=0;c<6;++c) img.at(r,c,0)=(float)((r*6+c)%9)/8.f;
    auto out=adapthisteq(img, 1000, 0.1f);
    EXPECT_EQ(out.rows, 6);
    EXPECT_EQ(out.cols, 6);
    for (float v:out.data) { EXPECT_TRUE(std::isfinite(v)); EXPECT_GE(v, 0.f); EXPECT_LE(v, 1.f); }
}

TEST(ImageAdaptHistEq, EmptyImage) {
    Image empty;
    auto out=adapthisteq(empty, 8, 0.01f);
    EXPECT_TRUE(out.empty());
}

TEST(ImageTransform, DftMagnitudeSize) {
    Image img(4,4,1);
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) img.at(r,c,0)=0.5f;
    auto mag=dft_magnitude(img);
    EXPECT_EQ(mag.rows, 4);
    EXPECT_EQ(mag.cols, 4);
    EXPECT_EQ(mag.channels, 1);
}

TEST(ImageTransform, RadonFinite) {
    Image img(8,8,1,0.f);
    for (int r=2;r<6;++r) for (int c=2;c<6;++c) img.at(r,c,0)=1.f;
    auto sino=radon(img,{0.f,45.f,90.f});
    EXPECT_EQ(sino.size(), 3u);
    for (auto& row:sino)
        for (float v:row)
            EXPECT_TRUE(std::isfinite(v));
}

static float normalized_cross_correlation(const Image& a, const Image& b) {
    float mean_a=0, mean_b=0;
    int n=a.rows*a.cols;
    for (int r=0;r<a.rows;++r) for (int c=0;c<a.cols;++c) {
        mean_a+=a.at(r,c,0); mean_b+=b.at(r,c,0);
    }
    mean_a/=n; mean_b/=n;
    float num=0, den_a=0, den_b=0;
    for (int r=0;r<a.rows;++r) for (int c=0;c<a.cols;++c) {
        float da=a.at(r,c,0)-mean_a, db=b.at(r,c,0)-mean_b;
        num+=da*db; den_a+=da*da; den_b+=db*db;
    }
    return num/(std::sqrt(den_a*den_b)+1e-12f);
}

TEST(ImageTransform, IradonRoundTrip) {
    Image img(16,16,1,0.f);
    for (int r=5;r<11;++r) for (int c=5;c<11;++c) img.at(r,c,0)=1.f;
    std::vector<float> angles;
    for (int i=0;i<180;++i) angles.push_back((float)i);
    auto sino=radon(img, angles);
    auto recon=iradon(sino, angles);
    EXPECT_EQ(recon.rows, 16);
    EXPECT_EQ(recon.cols, 16);
    float ncc=normalized_cross_correlation(img, recon);
    EXPECT_GT(ncc, 0.3f);
    int peak_r=0, peak_c=0;
    float peak_val=0;
    for (int r=0;r<16;++r) for (int c=0;c<16;++c)
        if (recon.at(r,c,0)>peak_val) { peak_val=recon.at(r,c,0); peak_r=r; peak_c=c; }
    EXPECT_NEAR(peak_r, 8, 2);
    EXPECT_NEAR(peak_c, 8, 2);
}

TEST(ImageTransform, IradonFinite) {
    Image img(8,8,1,0.f);
    img.at(4,4,0)=1.f;
    auto sino=radon(img,{0.f,30.f,60.f,90.f});
    auto recon=iradon(sino,{0.f,30.f,60.f,90.f});
    for (float v:recon.data) EXPECT_TRUE(std::isfinite(v));
}

TEST(ImageTransform, IradonEmpty) {
    auto recon=iradon({}, {});
    EXPECT_TRUE(recon.empty());
}

// ---- Hough Transform ----

TEST(HoughLines, HorizontalLine) {
    // Horizontal line row=25 across the full width -> normal form theta=pi/2, rho=25
    // (verifies this implementation's own x*cos(theta)+y*sin(theta)=rho convention).
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;
    auto lines=hough_lines(img, 0.5);
    ASSERT_FALSE(lines.empty());
    EXPECT_NEAR(lines[0].theta, M_PI/2.0, 0.02);
    EXPECT_NEAR(lines[0].rho, 25.0, 1.0);
    EXPECT_GE(lines[0].votes, 50);
}

TEST(HoughLines, VerticalLine) {
    // Vertical line col=25 across the full height -> normal form theta=0, rho=25.
    Image img(50,50,1,0.f);
    for (int r=0;r<50;++r) img.at(r,25,0)=1.f;
    auto lines=hough_lines(img, 0.5);
    ASSERT_FALSE(lines.empty());
    EXPECT_NEAR(lines[0].theta, 0.0, 0.02);
    EXPECT_NEAR(lines[0].rho, 25.0, 1.0);
    EXPECT_GE(lines[0].votes, 50);
}

TEST(HoughLines, DiagonalLine) {
    // Anti-diagonal x+y=49 -> normal form theta=pi/4, rho=49/sqrt(2)=34.648.
    Image img(50,50,1,0.f);
    for (int r=0;r<50;++r) img.at(r,49-r,0)=1.f;
    auto lines=hough_lines(img, 0.5);
    ASSERT_FALSE(lines.empty());
    EXPECT_NEAR(lines[0].theta, M_PI/4.0, 0.03);
    EXPECT_NEAR(lines[0].rho, 49.0/std::sqrt(2.0), 1.0);
    EXPECT_GE(lines[0].votes, 50);
}

TEST(HoughLines, BlankImageNoLines) {
    Image img(50,50,1,0.f);
    auto lines=hough_lines(img, 0.1, 180, 200, 1);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, EmptyImageReturnsEmpty) {
    Image empty;
    auto lines=hough_lines(empty, 0.5);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, TwoPerpendicularLinesBothDetected) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;   // horizontal: theta=pi/2, rho=25
    for (int r=0;r<50;++r) img.at(r,10,0)=1.f;   // vertical:   theta=0,    rho=10
    auto lines=hough_lines(img, 0.5);
    ASSERT_GE(lines.size(), 2u);

    bool found_horizontal=false, found_vertical=false;
    for (const auto& l:lines) {
        if (std::abs(l.theta-M_PI/2.0)<0.02 && std::abs(l.rho-25.0)<1.0) found_horizontal=true;
        if (std::abs(l.theta-0.0)<0.02 && std::abs(l.rho-10.0)<1.0) found_vertical=true;
    }
    EXPECT_TRUE(found_horizontal);
    EXPECT_TRUE(found_vertical);
}

TEST(HoughLines, DegenerateNThetaZero) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;
    auto lines=hough_lines(img, 0.5, 0, 200, 50);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, DegenerateNThetaNegative) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;
    auto lines=hough_lines(img, 0.5, -5, 200, 50);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, DegenerateNRhoZero) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;
    auto lines=hough_lines(img, 0.5, 180, 0, 50);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, DegenerateNRhoNegative) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;
    auto lines=hough_lines(img, 0.5, 180, -10, 50);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, VotesProportionalToLineLength) {
    // n_rho is set finer than the 200-bin default here so that the 25-pixel
    // line's votes for theta bins adjacent to its true peak spread across
    // more than one rho bin (rather than tying with the peak's count and
    // getting rejected by the strict-local-max check).
    Image full(50,50,1,0.f);
    for (int c=0;c<50;++c) full.at(10,c,0)=1.f;    // 50-pixel line
    Image half(50,50,1,0.f);
    for (int c=0;c<25;++c) half.at(10,c,0)=1.f;    // 25-pixel line

    auto full_lines=hough_lines(full, 0.5, 180, 500, 10);
    auto half_lines=hough_lines(half, 0.5, 180, 500, 10);
    ASSERT_FALSE(full_lines.empty());
    ASSERT_FALSE(half_lines.empty());
    EXPECT_EQ(full_lines[0].votes, 50);
    EXPECT_EQ(half_lines[0].votes, 25);
    EXPECT_GT(full_lines[0].votes, half_lines[0].votes);
}

TEST(HoughLines, EdgeThresholdFiltersLowIntensityPixels) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=0.3f;  // below edge_threshold
    auto lines=hough_lines(img, 0.5, 180, 200, 5);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, VoteThresholdFiltersWeakPeaks) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;   // only 50 votes possible
    auto lines=hough_lines(img, 0.5, 180, 200, 1000);
    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, ResultsSortedDescendingByVotes) {
    Image img(50,50,1,0.f);
    for (int c=0;c<50;++c) img.at(25,c,0)=1.f;
    for (int r=0;r<50;++r) img.at(r,10,0)=1.f;
    auto lines=hough_lines(img, 0.5, 180, 200, 5);
    for (size_t i=1;i<lines.size();++i) EXPECT_GE(lines[i-1].votes, lines[i].votes);
}

TEST(HoughLines, PeakDetectionAvoidsClusteredDuplicates) {
    // A loose vote_threshold would flood many near-duplicate cells clustered
    // around the true peak if we didn't require a strict local maximum;
    // check the surviving peak count stays small for a single clean line.
    Image img(50,50,1,0.f);
    for (int r=0;r<50;++r) img.at(r,25,0)=1.f;
    auto lines=hough_lines(img, 0.5, 180, 200, 10);
    EXPECT_LT(lines.size(), 10u);
    ASSERT_FALSE(lines.empty());
    EXPECT_NEAR(lines[0].theta, 0.0, 0.02);
    EXPECT_NEAR(lines[0].rho, 25.0, 1.0);
}

// ---- Circle Hough Transform ----

namespace {

void draw_circle_outline(Image& img, int cx, int cy, int r, float val=1.f) {
    int n=std::max(8,(int)std::lround(2*M_PI*r));
    for (int i=0;i<n;++i) {
        double theta=2.0*M_PI*i/n;
        int x=(int)std::lround(cx+r*std::cos(theta));
        int y=(int)std::lround(cy+r*std::sin(theta));
        if (x>=0&&x<img.cols&&y>=0&&y<img.rows) img.at(y,x,0)=val;
    }
}

} // namespace

TEST(HoughCircles, SingleCircleOutline) {
    const int cx=40, cy=40, r=20;
    Image img(80,80,1,0.f);
    draw_circle_outline(img, cx, cy, r);
    auto circles=hough_circles(img, 0.5, 15, 25, 1, 20);
    ASSERT_FALSE(circles.empty());
    EXPECT_NEAR(circles[0].cx, (double)cx, 2.0);
    EXPECT_NEAR(circles[0].cy, (double)cy, 2.0);
    EXPECT_NEAR(circles[0].r, (double)r, 2.0);
    EXPECT_GE(circles[0].votes, 20);
}

TEST(HoughCircles, TwoSeparatedCircles) {
    Image img(120,120,1,0.f);
    draw_circle_outline(img, 30, 30, 15);
    draw_circle_outline(img, 85, 85, 22);
    auto circles=hough_circles(img, 0.5, 10, 30, 1, 15);
    ASSERT_GE(circles.size(), 2u);

    bool found_small=false, found_large=false;
    for (const auto& c:circles) {
        if (std::abs(c.cx-30.0)<3.0&&std::abs(c.cy-30.0)<3.0&&std::abs(c.r-15.0)<3.0)
            found_small=true;
        if (std::abs(c.cx-85.0)<3.0&&std::abs(c.cy-85.0)<3.0&&std::abs(c.r-22.0)<3.0)
            found_large=true;
    }
    EXPECT_TRUE(found_small);
    EXPECT_TRUE(found_large);
}

TEST(HoughCircles, BlankImageNoCircles) {
    Image img(60,60,1,0.f);
    auto circles=hough_circles(img, 0.5, 5, 20, 1, 5);
    EXPECT_TRUE(circles.empty());
}

TEST(HoughCircles, EmptyImageReturnsEmpty) {
    Image empty;
    auto circles=hough_circles(empty, 0.5, 5, 20);
    EXPECT_TRUE(circles.empty());
}

TEST(HoughCircles, RadiusOutsideSearchRangeNotDetected) {
    const int cx=50, cy=50, r=30;
    Image img(100,100,1,0.f);
    draw_circle_outline(img, cx, cy, r);
    auto circles=hough_circles(img, 0.5, 5, 15, 1, 10);
    for (const auto& c:circles) {
        EXPECT_FALSE(std::abs(c.cx-cx)<3.0&&std::abs(c.cy-cy)<3.0&&std::abs(c.r-r)<3.0);
    }
}

TEST(HoughCircles, CannyPipelineIntegration) {
    const int cx=50, cy=50, r=25;
    Image img(100,100,1,0.f);
    for (int y=0;y<100;++y) for (int x=0;x<100;++x)
        if ((x-cx)*(x-cx)+(y-cy)*(y-cy)<=r*r) img.at(y,x,0)=1.f;
    auto edges=canny(img, 0.1f, 0.3f, 1.5f);
    auto circles=hough_circles(edges, 0.5, 18, 32, 1, 15);
    ASSERT_FALSE(circles.empty());
    EXPECT_NEAR(circles[0].cx, (double)cx, 4.0);
    EXPECT_NEAR(circles[0].cy, (double)cy, 4.0);
    EXPECT_NEAR(circles[0].r, (double)r, 4.0);
}

TEST(HoughCircles, DegenerateRStepZero) {
    Image img(50,50,1,0.f);
    draw_circle_outline(img, 25, 25, 10);
    auto circles=hough_circles(img, 0.5, 5, 15, 0, 10);
    EXPECT_TRUE(circles.empty());
}

TEST(HoughCircles, ResultsSortedDescendingByVotes) {
    Image img(120,120,1,0.f);
    draw_circle_outline(img, 30, 30, 12);
    draw_circle_outline(img, 85, 85, 22);
    auto circles=hough_circles(img, 0.5, 8, 28, 1, 10);
    for (size_t i=1;i<circles.size();++i) EXPECT_GE(circles[i-1].votes, circles[i].votes);
}

// ---- Harris ----

TEST(ImageHarris, CornerDetection) {
    Image img(10,10,1,0.f);
    // Create a corner: bright in one quadrant
    for (int r=5;r<10;++r) for (int c=5;c<10;++c) img.at(r,c,0)=1.f;
    auto kps=harris(img,0.04f,0.01f);
    // Should detect some keypoints
    EXPECT_GE(kps.size(), 0u);
}

TEST(ImageHarris, PositiveCornerResponse) {
    Image img(12,12,1,0.f);
    for (int r=6;r<12;++r) for (int c=6;c<12;++c) img.at(r,c,0)=1.f;
    auto kps=harris(img,0.04f,0.001f);
    ASSERT_GT(kps.size(), 0u);
    EXPECT_GT(kps[0].response, 0.f);
    EXPECT_GE(kps[0].x, 0.f);
    EXPECT_GE(kps[0].y, 0.f);
}

TEST(ImageShiTomasi, CornerLocalization) {
    Image img(20,20,1,0.f);
    for (int r=10;r<20;++r) for (int c=0;c<20;++c) img.at(r,c,0)=1.f;
    for (int r=0;r<10;++r) for (int c=10;c<20;++c) img.at(r,c,0)=1.f;
    auto kps=shi_tomasi(img, 5, 0.01f);
    ASSERT_GT(kps.size(), 0u);
    float best_dist=1e9f;
    for (const auto& kp:kps) {
        float dr=kp.y-10.f, dc=kp.x-10.f;
        best_dist=std::min(best_dist, std::sqrt(dr*dr+dc*dc));
    }
    EXPECT_LT(best_dist, 3.f);
}

TEST(ImageShiTomasi, FlatNoKeypoints) {
    Image img(10,10,1,0.5f);
    auto kps=shi_tomasi(img, 10, 0.01f);
    EXPECT_EQ(kps.size(), 0u);
}

TEST(ImageShiTomasi, TopNLimit) {
    Image img(24,24,1,0.f);
    for (int r=12;r<24;++r) for (int c=0;c<24;++c) img.at(r,c,0)=1.f;
    for (int r=0;r<12;++r) for (int c=12;c<24;++c) img.at(r,c,0)=1.f;
    auto kps=shi_tomasi(img, 2, 0.001f);
    EXPECT_LE(kps.size(), 2u);
    if (kps.size()>=2u) EXPECT_GE(kps[0].response, kps[1].response);
}

TEST(ImageShiTomasi, QualityThreshold) {
    Image img(16,16,1,0.f);
    for (int r=8;r<16;++r) for (int c=8;c<16;++c) img.at(r,c,0)=1.f;
    auto strict=shi_tomasi(img, 20, 0.5f);
    auto loose=shi_tomasi(img, 20, 0.001f);
    EXPECT_LE(strict.size(), loose.size());
}

// ---- Connected Components ----

TEST(ImageComponents, TwoBlobs) {
    Image bw(6,6,1,0.f);
    bw.at(0,0,0)=1.f; bw.at(0,1,0)=1.f;  // blob 1
    bw.at(5,5,0)=1.f; bw.at(4,5,0)=1.f;  // blob 2
    int n=count_components(bw);
    EXPECT_EQ(n, 2);
}

TEST(ImageComponents, LabelSimpleBlob) {
    Image bw(4,4,1,0.f);
    bw.at(1,1,0)=1.f; bw.at(1,2,0)=1.f;
    bw.at(2,1,0)=1.f; bw.at(2,2,0)=1.f;
    auto labels=label_components(bw);
    EXPECT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels[0].size(), 4u);
    int id=labels[1][1];
    EXPECT_GE(id, 0);
    EXPECT_EQ(labels[1][2], id);
    EXPECT_EQ(labels[2][1], id);
    EXPECT_EQ(labels[2][2], id);
    EXPECT_EQ(labels[0][0], -1);
}

TEST(ImageComponents, BilinearSample) {
    Image img(4,4,1,0.f);
    img.at(0,0,0)=1.f;
    float v=bilinear_sample(img,0.f,0.f,0);
    EXPECT_FLOAT_EQ(v, 1.f);
}

TEST(ImageComponents, BilinearInterp) {
    Image img(2,2,1,0.f);
    img.at(0,0,0)=0.f; img.at(0,1,0)=1.f;
    img.at(1,0,0)=0.f; img.at(1,1,0)=1.f;
    float v=bilinear_sample(img,0.5f,0.5f,0);
    EXPECT_NEAR(v, 0.5f, 0.01f);
}

// ---- Watershed ----

static Image make_two_hill_gray(int rows, int cols) {
    Image gray(rows, cols, 1, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float d1 = std::hypot(static_cast<float>(r - rows / 2),
                                  static_cast<float>(c - cols / 4));
            float d2 = std::hypot(static_cast<float>(r - rows / 2),
                                  static_cast<float>(c - 3 * cols / 4));
            gray.at(r, c, 0) = std::max(0.f, 1.f - d1 / 6.f) + std::max(0.f, 1.f - d2 / 6.f);
        }
    return gray;
}

TEST(ImageWatershed, EmptyGrayReturnsEmpty) {
    Image markers(4, 4, 1, 0.f);
    markers.at(1, 1, 0) = 1.f;
    auto out = watershed(Image{}, markers);
    EXPECT_TRUE(out.empty());
}

TEST(ImageWatershed, EmptyMarkersReturnsEmpty) {
    Image gray(4, 4, 1, 0.5f);
    auto out = watershed(gray, Image{});
    EXPECT_TRUE(out.empty());
}

TEST(ImageWatershed, SizeMismatchReturnsEmpty) {
    Image gray(4, 4, 1, 0.5f);
    Image markers(3, 4, 1, 0.f);
    auto out = watershed(gray, markers);
    EXPECT_TRUE(out.empty());
}

TEST(ImageWatershed, OutputShapeMatchesInput) {
    Image gray(12, 16, 1, 0.4f);
    Image markers(12, 16, 1, 0.f);
    markers.at(6, 4, 0) = 1.f;
    auto out = watershed(gray, markers);
    EXPECT_EQ(out.rows, 12);
    EXPECT_EQ(out.cols, 16);
    EXPECT_EQ(out.channels, 1);
}

TEST(ImageWatershed, TwoBlobSyntheticSeparates) {
    const int rows = 20, cols = 20;
    auto gray = make_two_hill_gray(rows, cols);
    Image markers(rows, cols, 1, 0.f);
    markers.at(rows / 2, cols / 4, 0) = 1.f;
    markers.at(rows / 2, 3 * cols / 4, 0) = 2.f;

    auto out = watershed(gray, markers);
    EXPECT_FLOAT_EQ(out.at(rows / 2, cols / 4, 0), 1.f);
    EXPECT_FLOAT_EQ(out.at(rows / 2, 3 * cols / 4, 0), 2.f);
    EXPECT_FLOAT_EQ(out.at(rows / 2, cols / 2, 0), 0.f);

    int left = 0, right = 0, boundary = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float l = out.at(r, c, 0);
            if (l == 1.f) ++left;
            else if (l == 2.f) ++right;
            else if (l == 0.f) ++boundary;
        }
    EXPECT_GT(left, 0);
    EXPECT_GT(right, 0);
    EXPECT_GT(boundary, 0);
}

TEST(ImageWatershed, SingleMarkerFloodsBasin) {
    Image gray(8, 8, 1, 0.2f);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            gray.at(r, c, 0) = 0.2f + 0.6f * std::max(0.f, 1.f - std::hypot(static_cast<float>(r - 4),
                                                                                 static_cast<float>(c - 4)) / 3.f);
    Image markers(8, 8, 1, 0.f);
    markers.at(4, 4, 0) = 1.f;
    auto out = watershed(gray, markers);
    EXPECT_FLOAT_EQ(out.at(4, 4, 0), 1.f);
    EXPECT_FLOAT_EQ(out.at(0, 0, 0), 1.f);
    EXPECT_FLOAT_EQ(out.at(7, 7, 0), 1.f);
}

TEST(ImageWatershed, NoMarkersAllZero) {
    Image gray(5, 5, 1, 0.5f);
    Image markers(5, 5, 1, 0.f);
    auto out = watershed(gray, markers);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
            EXPECT_FLOAT_EQ(out.at(r, c, 0), 0.f);
}

TEST(ImageWatershed, RgbGrayInputWorks) {
    Image gray(6, 6, 3, 0.f);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            float v = 0.3f + 0.5f * std::max(0.f, 1.f - std::hypot(static_cast<float>(r - 3),
                                                                     static_cast<float>(c - 3)) / 2.5f);
            gray.at(r, c, 0) = v;
            gray.at(r, c, 1) = v;
            gray.at(r, c, 2) = v;
        }
    Image markers(6, 6, 1, 0.f);
    markers.at(3, 3, 0) = 1.f;
    auto out = watershed(gray, markers);
    EXPECT_EQ(out.channels, 1);
    EXPECT_FLOAT_EQ(out.at(3, 3, 0), 1.f);
    EXPECT_FLOAT_EQ(out.at(0, 0, 0), 1.f);
}

TEST(ImageWatershed, DistinctLabelsDoNotMerge) {
    auto gray = make_two_hill_gray(18, 18);
    Image markers(18, 18, 1, 0.f);
    markers.at(9, 4, 0) = 1.f;
    markers.at(9, 13, 0) = 2.f;
    auto out = watershed(gray, markers);
    EXPECT_FLOAT_EQ(out.at(9, 2, 0), 1.f);
    EXPECT_FLOAT_EQ(out.at(9, 15, 0), 2.f);
    EXPECT_NE(out.at(9, 2, 0), out.at(9, 15, 0));
}

// ---- SLIC ----

static int count_distinct_labels(const Image& labels) {
    std::set<int> uniq;
    for (int r = 0; r < labels.rows; ++r)
        for (int c = 0; c < labels.cols; ++c) {
            const int l = static_cast<int>(std::lround(labels.at(r, c, 0)));
            if (l > 0) uniq.insert(l);
        }
    return static_cast<int>(uniq.size());
}

static Image make_checkerboard_rgb(int rows, int cols, int cell) {
    Image img(rows, cols, 3, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const bool white = ((r / cell) + (c / cell)) % 2 == 0;
            const float v = white ? 1.f : 0.f;
            img.at(r, c, 0) = v;
            img.at(r, c, 1) = v;
            img.at(r, c, 2) = v;
        }
    return img;
}

TEST(ImageSlic, EmptyInputReturnsEmpty) {
    auto out = slic(Image{}, 4);
    EXPECT_TRUE(out.empty());
}

TEST(ImageSlic, NonPositiveSuperpixelsReturnsEmpty) {
    Image rgb(8, 8, 3, 0.5f);
    EXPECT_TRUE(slic(rgb, 0).empty());
    EXPECT_TRUE(slic(rgb, -1).empty());
}

TEST(ImageSlic, OutputShapeMatchesInput) {
    Image rgb(12, 16, 3, 0.4f);
    auto out = slic(rgb, 8);
    EXPECT_EQ(out.rows, 12);
    EXPECT_EQ(out.cols, 16);
    EXPECT_EQ(out.channels, 1);
}

TEST(ImageSlic, UniformImageOneSuperpixel) {
    Image rgb(20, 20, 3, 0.6f);
    auto out = slic(rgb, 1);
    EXPECT_EQ(count_distinct_labels(out), 1);
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 20; ++c)
            EXPECT_FLOAT_EQ(out.at(r, c, 0), out.at(0, 0, 0));
}

TEST(ImageSlic, CheckerboardMultipleLabels) {
    auto rgb = make_checkerboard_rgb(32, 32, 4);
    auto out = slic(rgb, 16, 10.0);
    EXPECT_GE(count_distinct_labels(out), 2);
}

TEST(ImageSlic, EveryPixelHasPositiveLabel) {
    Image rgb(10, 10, 3, 0.3f);
    rgb.at(5, 5, 0) = 0.9f;
    rgb.at(5, 5, 1) = 0.1f;
    rgb.at(5, 5, 2) = 0.2f;
    auto out = slic(rgb, 4);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            EXPECT_GT(out.at(r, c, 0), 0.f);
}

TEST(ImageSlic, GrayscaleInputWorks) {
    Image gray(14, 14, 1, 0.2f);
    for (int r = 0; r < 14; ++r)
        for (int c = 0; c < 14; ++c)
            if (((r / 7) + (c / 7)) % 2 == 0) gray.at(r, c, 0) = 0.9f;
    auto out = slic(gray, 8);
    EXPECT_EQ(out.channels, 1);
    EXPECT_GE(count_distinct_labels(out), 2);
}

TEST(ImageSlic, TwoColorBlocksSeparate) {
    Image rgb(24, 24, 3, 0.f);
    for (int r = 0; r < 24; ++r)
        for (int c = 0; c < 12; ++c) {
            rgb.at(r, c, 0) = 0.9f;
            rgb.at(r, c, 1) = 0.1f;
            rgb.at(r, c, 2) = 0.1f;
        }
    for (int r = 0; r < 24; ++r)
        for (int c = 12; c < 24; ++c) {
            rgb.at(r, c, 0) = 0.1f;
            rgb.at(r, c, 1) = 0.8f;
            rgb.at(r, c, 2) = 0.2f;
        }
    auto out = slic(rgb, 6, 5.0);
    const float left = out.at(12, 4, 0);
    const float right = out.at(12, 20, 0);
    EXPECT_NE(left, right);
}

TEST(ImageSlic, CompactnessAffectsSegmentation) {
    auto rgb = make_checkerboard_rgb(24, 24, 6);
    auto tight = slic(rgb, 12, 1.0);
    auto loose = slic(rgb, 12, 40.0);
    EXPECT_GE(count_distinct_labels(tight), 1);
    EXPECT_GE(count_distinct_labels(loose), 1);
}

TEST(ImageSlic, SuperpixelCountBoundedByPixels) {
    Image rgb(4, 4, 3, 0.5f);
    auto out = slic(rgb, 100);
    EXPECT_LE(count_distinct_labels(out), 16);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_GT(out.at(r, c, 0), 0.f);
}
