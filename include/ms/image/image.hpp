#pragma once
#include <cstdint>
#include <vector>

namespace ms {
namespace image {

// ========================== Image struct ==========================
// Pixels stored as float [0, 1] in row-major order: data[row*cols*channels + col*channels + ch]
struct Image {
    int rows = 0, cols = 0, channels = 0;
    std::vector<float> data;

    Image() = default;
    Image(int r, int c, int ch, float fill = 0.f);
    float& at(int r, int c, int ch);
    float  at(int r, int c, int ch) const;
    bool empty() const { return data.empty(); }
};

// ========================== Color Conversions ==========================
Image rgb2gray(const Image& img);
Image gray2rgb(const Image& img);
Image rgb2hsv(const Image& img);
Image hsv2rgb(const Image& img);

// ========================== Geometric ==========================
Image imresize(const Image& img, int new_rows, int new_cols);
Image imcrop(const Image& img, int r0, int c0, int r1, int c1);
Image imflip(const Image& img, bool horizontal);
Image imrotate90(const Image& img);  // 90° CCW
Image impad(const Image& img, int pad, float val = 0.f);

// ========================== Filtering ==========================
Image imfilter(const Image& img, const std::vector<std::vector<float>>& kernel);
Image imgaussfilt(const Image& img, float sigma);
Image medfilt2(const Image& img, int ksize = 3);
Image bilateral(const Image& img, float sigma_s, float sigma_r);
Image boxfilter(const Image& img, int ksize);
Image sharpen(const Image& img);

// ========================== Edge Detection ==========================
Image sobel(const Image& img);      // returns gradient magnitude
Image sobel_x(const Image& img);
Image sobel_y(const Image& img);
Image prewitt(const Image& img);
Image laplacian(const Image& img);
Image canny(const Image& img, float low, float high, float sigma = 1.0f);
Image scharr(const Image& img);
Image roberts(const Image& img);
// LoG via sequential composition: laplacian(imgaussfilt(img, sigma))
Image laplacian_of_gaussian(const Image& img, float sigma);

// ========================== Morphology ==========================
Image imdilate(const Image& img, int ksize = 3);
Image imerode(const Image& img, int ksize = 3);
Image imopen(const Image& img, int ksize = 3);
Image imclose(const Image& img, int ksize = 3);
Image imtophat(const Image& img, int ksize = 3);
Image imbothat(const Image& img, int ksize = 3);
// Morphological gradient: imdilate(img, ksize) - imerode(img, ksize)
Image imgradient_morph(const Image& img, int ksize = 3);

// ========================== Thresholding / Segmentation ==========================
Image threshold_binary(const Image& img, float t);
Image threshold_otsu(const Image& img);   // auto threshold via Otsu

// ========================== Histogram ==========================
std::vector<int>   imhist(const Image& img, int nbins = 256);   // grayscale
Image              histeq(const Image& img);                     // histogram equalisation
Image              imadjust(const Image& img, float in_lo, float in_hi,
                             float out_lo = 0.f, float out_hi = 1.f);

// ========================== Transforms ==========================
// Returns DFT magnitude spectrum (real-valued, same size as input)
Image dft_magnitude(const Image& img);

// Radon transform: project at angles theta (degrees), returns sinogram
std::vector<std::vector<float>> radon(const Image& img,
                                       const std::vector<float>& theta_deg);
// Inverse Radon via unfiltered backprojection (matches radon angle/coordinate convention)
Image iradon(const std::vector<std::vector<float>>& sinogram,
             const std::vector<float>& theta_deg);

// ========================== Feature Detection ==========================
struct KeyPoint { float x, y, response; };
std::vector<KeyPoint> harris(const Image& img, float k = 0.04f,
                              float threshold = 0.01f);
// Shi-Tomasi corner detector: min eigenvalue of structure tensor (same as harris)
std::vector<KeyPoint> shi_tomasi(const Image& img, int n,
                                  float quality_level = 0.01f);

// Connected components labelling
std::vector<std::vector<int>> label_components(const Image& bw);
int count_components(const Image& bw);

// ========================== Utilities ==========================
Image im_from_data(int rows, int cols, int channels, const std::vector<float>& data);
float bilinear_sample(const Image& img, float r, float c, int ch);

} // namespace image
} // namespace ms
