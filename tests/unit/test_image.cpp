#include "ms/image/image.hpp"
#include <cmath>
#include <gtest/gtest.h>

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
