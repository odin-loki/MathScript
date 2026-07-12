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

/// Contrast-Limited Adaptive Histogram Equalisation (CLAHE).
///
/// Unlike histeq(), which computes a single global CDF-based mapping for the
/// whole image, adapthisteq() divides the image into a grid of roughly
/// @p tile_size x @p tile_size tiles, equalises each tile's local histogram
/// independently (after clipping it, see @p clip_limit), and then blends
/// the per-tile mappings with bilinear interpolation. This brings out local
/// contrast that a global histeq() cannot, since a pattern that spans only a
/// narrow intensity band can still dominate a small tile's own histogram
/// even though it is a small, low-slope contributor to the whole image's
/// histogram.
/// @param img Input image (RGB is converted to grayscale first, matching
///            histeq()'s convention; the output is always single-channel).
/// @param tile_size Approximate tile edge length in pixels. Clamped to
///                   >= 1; the last tile in each row/column of the grid may
///                   be smaller when the image dimension is not an exact
///                   multiple of @p tile_size. A @p tile_size that is >= the
///                   image dimension collapses that axis to a single tile
///                   (equivalent to global histeq() along that axis).
/// @param clip_limit Clipping strength, in (0, 1]. Any histogram bin whose
///                    count exceeds `clip_limit * tile_pixel_count` is
///                    clipped to that limit (with a floor of 1 count so a
///                    tile is never fully zeroed out); the excess mass is
///                    redistributed uniformly over the remaining bins
///                    (iterated a bounded number of times to re-clip any
///                    bin pushed back over the limit). Smaller values clip
///                    more aggressively, giving a gentler, noise-safe local
///                    contrast boost; values close to 1 approach unclipped
///                    per-tile histogram equalisation.
/// @return A single-channel image the same size as @p img, with values in
///         [0, 1].
/// @note Naively pasting together independently-equalised tiles produces
///       visible blocky discontinuities at tile borders. To avoid this,
///       each output pixel's value is computed by bilinearly interpolating
///       between the (up to 4) nearest tile centers' mapping functions,
///       all applied to that pixel's original intensity, so the effective
///       mapping varies smoothly across the image.
/// @note Defensive: returns an empty Image for an empty input.
Image adapthisteq(const Image& img, int tile_size = 8, float clip_limit = 0.01f);

// ========================== Transforms ==========================
// Returns DFT magnitude spectrum (real-valued, same size as input)
Image dft_magnitude(const Image& img);

// Radon transform: project at angles theta (degrees), returns sinogram
std::vector<std::vector<float>> radon(const Image& img,
                                       const std::vector<float>& theta_deg);
// Inverse Radon via unfiltered backprojection (matches radon angle/coordinate convention)
Image iradon(const std::vector<std::vector<float>>& sinogram,
             const std::vector<float>& theta_deg);

// ========================== Hough Transform ==========================

/// Detected straight line from hough_lines(), in normal form
/// `x*cos(theta) + y*sin(theta) = rho` (x = column, y = row, matching this
/// module's KeyPoint convention).
struct HoughLine { double rho, theta; int votes; };

/// Standard (straight-line) Hough transform.
///
/// For every "edge" pixel (a pixel whose value exceeds @p edge_threshold),
/// casts a vote in (rho, theta) parameter space for every line -- in normal
/// form `x*cos(theta) + y*sin(theta) = rho` -- passing through that pixel,
/// across a discretised grid of @p n_theta bins spanning theta in [0, pi)
/// and @p n_rho bins spanning rho in [-rho_max, rho_max], where
/// `rho_max = sqrt(cols^2 + rows^2)` is the image diagonal (the largest
/// |rho| any line intersecting the image can have). Simple local-maximum
/// peak detection is then run on the resulting accumulator: a cell is a
/// peak if its vote count is >= @p vote_threshold AND it is a strict local
/// maximum among its immediate neighbours in the discretised (rho, theta)
/// grid, which avoids returning many near-duplicate lines clustered around
/// a single true line's peak.
/// @param img Input image (RGB is converted to grayscale first, matching
///            this module's other analysis functions).
/// @param edge_threshold Pixel intensity threshold above which a pixel is
///        treated as an "edge" pixel that votes in the accumulator.
/// @param n_theta Number of theta bins over [0, pi) (typical: 180, for
///        1-degree resolution). `n_theta <= 0` is degenerate.
/// @param n_rho Number of rho bins over [-rho_max, rho_max] (typical: a
///        few hundred, e.g. `2*ceil(rho_max)` for roughly 0.5-pixel
///        resolution). `n_rho <= 0` is degenerate.
/// @param vote_threshold Minimum accumulator vote count for a cell to be
///        considered a candidate peak/detected line.
/// @return Detected lines as {rho, theta, votes} sorted by descending vote
///         count. An empty image or degenerate parameters (`n_theta <= 0`
///         or `n_rho <= 0`) return `{}`.
/// @note This votes directly on pixels above @p edge_threshold (simple
///       thresholding), not a full Canny pipeline. Callers working with
///       noisy images will usually get cleaner peaks by running canny() (or
///       another edge detector already in this module) first and passing
///       the resulting edge map in, e.g. `hough_lines(canny(img, lo, hi), 0.5)`.
/// @note Peak detection treats theta as non-wrapping: it does not check
///       neighbours across the theta=0/theta=pi boundary (which, since a
///       line's normal form is periodic there only after flipping the sign
///       of rho, would require special-casing rather than a plain
///       neighbour lookup). This is an intentional simplification and can
///       occasionally let a true peak that sits at a theta boundary be
///       reported alongside a near-duplicate on the other side.
std::vector<HoughLine> hough_lines(const Image& img, double edge_threshold,
                                    int n_theta = 180, int n_rho = 200,
                                    int vote_threshold = 50);

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
