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
