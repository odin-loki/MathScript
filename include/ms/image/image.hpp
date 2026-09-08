#pragma once
#include <array>
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

/// Marker-controlled watershed on a grayscale image.
///
/// Priority-flood segmentation from seed markers: each positive label in @p markers
/// grows to 4-connected neighbors in descending @p gray order (catchment basins);
/// pixels where two basins meet are labelled 0 (watershed boundary). @p gray and
/// @p markers must share the same size; multi-channel inputs are converted to
/// grayscale / single-channel first. Output is single-channel with integer-like
/// float labels (1, 2, …) and 0 for boundaries or unlabelled pixels.
/// Defensive: returns an empty Image for empty inputs or size mismatch.
Image watershed(const Image& gray, const Image& markers);

/// Simple Linear Iterative Clustering (SLIC) superpixel segmentation.
///
/// k-means on 5D Lab+xy features with a spatial compactness term. @p rgb is
/// converted to CIELab internally (grayscale inputs are replicated to RGB
/// first). Returns a single-channel label image with integer-like float labels
/// 1 … K (one label per pixel). @p compactness balances colour vs spatial
/// proximity (typical values 1–40; default 10). Defensive: returns an empty
/// Image for empty input or @p num_superpixels <= 0.
Image slic(const Image& rgb, int num_superpixels, double compactness = 10.0);

// ========================== Graph-Cut Segmentation ==========================

/// Binary foreground/background segmentation by s-t minimum cut
/// (Boykov-Jolly interactive graph cuts).
///
/// Builds a grid graph with one node per pixel plus a virtual source (S,
/// foreground) and sink (T, background), and returns the labelling that
/// globally minimises the two-label energy
///
///     E(L) = sum_p D_p(L_p)  +  sum_{(p,q) in N} [L_p != L_q] * w_pq
///
/// over all 2^(rows*cols) labellings. The pairwise term is a Potts term with
/// non-negative weights, so the energy is submodular and the minimum is
/// attained exactly by a single max-flow computation -- unlike slic() or the
/// k-means style iterations elsewhere in this module, this is not a local
/// search and does not depend on an initialisation.
///
/// **Data term.** Each pixel is scored against two intensity means by squared
/// distance: `D_p(fg) = (I_p - mu_fg)^2`, `D_p(bg) = (I_p - mu_bg)^2`. The
/// means come from @p fg_seeds / @p bg_seeds when those masks mark at least
/// one pixel each (a pixel is marked when its value is >= 0.5); a mask that
/// marks nothing, or an empty Image, contributes no seeds. When only one mask
/// marks pixels, the other mean is taken over all pixels that mask does not
/// cover. When neither marks anything, both means come from a deterministic
/// 1-D 2-means (Lloyd) split of the intensities, initialised at the image min
/// and max. Seed pixels themselves get hard constraints instead of a
/// squared-distance score, so a seed can never be labelled against the mask
/// the caller supplied.
///
/// **Smoothness term.** For each neighbour pair the contrast-sensitive Potts
/// weight is `w_pq = lambda * exp(-(I_p - I_q)^2 / (2*sigma^2)) / dist(p,q)`,
/// with `dist = 1` for the two axial offsets and `sqrt(2)` for the two
/// diagonal offsets that @p connectivity 8 adds. Large @p lambda buys shorter,
/// smoother boundaries; small @p sigma makes the boundary cling harder to
/// intensity edges.
///
/// @param gray Input image. `channels >= 3` is converted with rgb2gray();
///        1- or 2-channel input uses channel 0.
/// @param fg_seeds Foreground seed mask, same rows/cols as @p gray, or an
///        empty Image for "no foreground seeds". Pixels >= 0.5 are hard
///        foreground.
/// @param bg_seeds Background seed mask, same convention. A pixel marked in
///        both masks is treated as foreground.
/// @param lambda Smoothness weight. `lambda <= 0` drops the pairwise term
///        entirely, which reduces the result to independent per-pixel
///        thresholding at `(mu_fg + mu_bg) / 2`.
/// @param sigma Contrast scale of the smoothness term, in intensity units.
///        `sigma <= 0` uses the exact limit as sigma -> 0+: `w_pq =
///        lambda / dist(p,q)` when `I_p == I_q` exactly and `0` otherwise (a
///        hard Potts term that only smooths across flat regions).
/// @param connectivity 4 (axial neighbours) or 8 (axial + diagonal). Any
///        other value is treated as 4.
/// @return A single-channel mask the same size as @p gray, `1.f` for
///         foreground and `0.f` for background (matching threshold_binary()'s
///         convention).
/// @note Ties are resolved toward background: the returned foreground set is
///       the *smallest* of all minimum-cut source sides (the pixels reachable
///       from S in the final residual graph), which is unique and is the
///       intersection of every optimal labelling's foreground set. A pixel
///       whose two data terms are equal and which no seed reaches is therefore
///       labelled background, and an image whose all-foreground and
///       all-background labellings tie comes back all background.
/// @note Defensive: returns an empty Image if @p gray is empty, if a non-empty
///       seed mask disagrees with @p gray's rows/cols, or if the pixel count
///       exceeds the solver's index range.
/// @note Complexity: O(N) memory for N = rows*cols (roughly 120 bytes per
///       pixel at connectivity 4, 230 at connectivity 8). Boykov-Kolmogorov
///       max-flow is worst-case polynomial but near-linear on grid graphs in
///       practice; the graph build and the labelling pass are O(N + E).
Image graph_cut_segment(const Image& gray, const Image& fg_seeds,
                        const Image& bg_seeds, double lambda = 1.0,
                        double sigma = 0.1, int connectivity = 4);

/// Unseeded binary graph-cut segmentation.
///
/// Equivalent to `graph_cut_segment(gray, Image{}, Image{}, lambda, sigma,
/// connectivity)`: both intensity means are estimated by the deterministic
/// 1-D 2-means split described above, so the result is a smoothness-
/// regularised version of a two-cluster intensity threshold. A uniform image
/// has both means equal, every data term equal, and therefore returns an
/// all-background mask.
Image graph_cut_segment(const Image& gray, double lambda = 1.0,
                        double sigma = 0.1, int connectivity = 4);

/// Energy of the labelling graph_cut_segment() returns -- the value of the
/// minimum s-t cut.
///
/// Same graph, same parameters and same degenerate handling as
/// graph_cut_segment(); this returns the achieved `E(L)` (the data terms of
/// the chosen labels plus the weights of every neighbour pair the boundary
/// separates) instead of the labelling. Because the cut is a global optimum,
/// no other labelling of @p gray under these parameters has a smaller value.
///
/// @return The minimum energy, always >= 0. Returns `0.0` for any input
///         graph_cut_segment() would reject (empty image, seed-mask size
///         mismatch, pixel count out of range) and for the genuinely
///         zero-energy cases (all-foreground seeds, all-background seeds, a
///         uniform unseeded image, a 1x1 image).
/// @note The value is accumulated as a sum of augmenting-path bottlenecks plus
///       a reparametrisation offset, with no iterative refinement, so an image
///       whose intensities and parameters are exactly representable gives an
///       exactly representable answer.
/// @note Calling both min_cut_value() and graph_cut_segment() runs the solver
///       twice; there is no shared state between them.
double min_cut_value(const Image& gray, const Image& fg_seeds,
                     const Image& bg_seeds, double lambda = 1.0,
                     double sigma = 0.1, int connectivity = 4);

/// Energy of the labelling the unseeded graph_cut_segment() overload returns.
double min_cut_value(const Image& gray, double lambda = 1.0,
                     double sigma = 0.1, int connectivity = 4);

/// GrabCut: iterated graph cut with per-region Gaussian mixture intensity
/// models, initialised from a bounding rectangle.
///
/// Where graph_cut_segment() scores a pixel against two scalar means and cuts
/// once, grabcut_segment() alternates two steps: fit a K-component 1-D
/// Gaussian mixture to the intensities currently labelled foreground and
/// another to those labelled background, then re-cut with
/// `D_p(fg) = -ln p_fg(I_p)` and `D_p(bg) = -ln p_bg(I_p)`. That lets a region
/// be multi-modal (a dark object on a background that is dark in one place and
/// bright in another) where a single mean cannot.
///
/// Pixels outside the rectangle are hard background for every iteration and
/// can never become foreground; pixels inside start as foreground and are free
/// to be reassigned. Iteration stops early when a pass leaves the labelling
/// unchanged, or when either region becomes empty.
///
/// @param gray Input image; same channel handling as graph_cut_segment().
/// @param r0 First row of the initial foreground rectangle (inclusive).
/// @param c0 First column of the initial foreground rectangle (inclusive).
/// @param r1 One past the last row of the rectangle.
/// @param c1 One past the last column of the rectangle. The rectangle is the
///        half-open box `[r0, r1) x [c0, c1)` and is clamped to the image,
///        matching imcrop()'s convention.
/// @param iterations Maximum number of model-fit / min-cut passes. Values
///        below 1 are treated as 1.
/// @param n_components Gaussian components per region, clamped to [1, 8] and
///        further to the number of pixels available in that region. Fitted by
///        1-D Lloyd k-means seeded at the sorted region's (j+0.5)/K quantiles,
///        so the fit is deterministic -- no random restarts, and repeated
///        calls give bit-identical masks.
/// @param lambda Smoothness weight, as in graph_cut_segment().
/// @param sigma Contrast scale of the smoothness term, as in
///        graph_cut_segment().
/// @param connectivity 4 or 8, as in graph_cut_segment().
/// @return A single-channel mask the same size as @p gray, `1.f` foreground /
///         `0.f` background.
/// @note The mixtures are over scalar intensity rather than RGB, matching the
///       rest of this module's segmentation surface (watershed(),
///       threshold_otsu(), graph_cut_segment()), and the components are fitted
///       by hard-assignment k-means plus Gaussian moments rather than soft EM
///       -- which is what the original GrabCut formulation prescribes, and
///       what keeps the fit deterministic.
/// @note Degenerate: an empty @p gray returns an empty Image. A rectangle that
///       clamps to zero area returns an all-background mask. A rectangle
///       covering the whole image leaves no background samples for the first
///       fit, so the initial labelling stands and the result is all
///       foreground. A uniform image gives two identical mixtures, an
///       uninformative data term, and an all-background result.
/// @note Complexity O(iterations * (N*K + max-flow)), plus O(N log N) per pass
///       for the sort inside the mixture fit.
Image grabcut_segment(const Image& gray, int r0, int c0, int r1, int c1,
                      int iterations = 5, int n_components = 3,
                      double lambda = 1.0, double sigma = 0.1,
                      int connectivity = 4);

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

/// Detected circle from hough_circles() (cx = column, cy = row, matching
/// this module's KeyPoint / hough_lines coordinate convention).
struct HoughCircle { double cx, cy, r; int votes; };

/// Standard circle Hough transform.
///
/// For every "edge" pixel (a pixel whose value exceeds @p edge_threshold),
/// and for every candidate integer radius in [@p r_min, @p r_max] stepped by
/// @p r_step, casts votes for all integer centers (cx, cy) lying on the
/// circle of that radius centred on the edge pixel — i.e. all (cx, cy) such
/// that `(x-cx)^2 + (y-cy)^2 = r^2` for edge (x, y). Votes accumulate in a
/// 3D accumulator indexed by (cx, cy, r). Simple local-maximum peak detection
/// is then run on the accumulator: a cell is a peak if its vote count is >=
/// @p vote_threshold AND it is a strict local maximum among its immediate
/// neighbours in the discretised (cx, cy, r) grid, which avoids returning many
/// near-duplicate circles clustered around a single true peak.
/// @param img Input image (RGB is converted to grayscale first, matching
///            this module's other analysis functions).
/// @param edge_threshold Pixel intensity threshold above which a pixel is
///        treated as an "edge" pixel that votes in the accumulator.
/// @param r_min Minimum search radius in pixels (inclusive).
/// @param r_max Maximum search radius in pixels (inclusive).
/// @param r_step Radius discretisation step in pixels (typical: 1).
/// @param vote_threshold Minimum accumulator vote count for a cell to be
///        considered a candidate peak/detected circle.
/// @return Detected circles as {cx, cy, r, votes} sorted by descending vote
///         count. An empty image, invalid parameters (`r_step <= 0` or
///         `r_min > r_max`), or an empty radius range return `{}`.
/// @note Voting uses a full 360° sweep around each edge pixel (not gradient-
///       directed voting), which is more robust when callers pass sparse
///       synthetic outlines or Canny edge maps where gradient direction may
///       be noisy. Angular resolution scales with radius (~2*pi*r samples
///       per edge/radius pair, with a floor of 8).
/// @note Complexity is O(edge_pixels * n_radii * n_angles_per_radius) in
///       both time and accumulator memory; suitable for small-to-medium
///       images and modest radius ranges. Callers working with noisy images
///       will usually get cleaner peaks by running canny() first and passing
///       the resulting edge map in, e.g.
///       `hough_circles(canny(img, lo, hi), 0.5, 10, 40)`.
std::vector<HoughCircle> hough_circles(const Image& img, double edge_threshold,
                                        double r_min, double r_max,
                                        int r_step = 1, int vote_threshold = 30);

// ========================== Feature Detection ==========================
struct KeyPoint { float x, y, response; };
std::vector<KeyPoint> harris(const Image& img, float k = 0.04f,
                              float threshold = 0.01f);
// Shi-Tomasi corner detector: min eigenvalue of structure tensor (same as harris)
std::vector<KeyPoint> shi_tomasi(const Image& img, int n,
                                  float quality_level = 0.01f);

// ========================== Local Features (FAST / ORB / SIFT) ==========================

/// A scale- and rotation-aware keypoint produced by orb_detect_and_compute()
/// and sift_detect_and_compute().
///
/// Coordinates are always expressed in **input-image** (base) pixels even when
/// the feature was found on a coarser pyramid level, so keypoints from
/// different octaves can be compared and drawn without further conversion.
/// The coordinate convention matches KeyPoint / HoughLine / HoughCircle:
/// @c x is the column, @c y is the row, both sub-pixel.
struct FeatureKeyPoint {
    /// Column in base-image pixels (sub-pixel).
    float x = 0.f;
    /// Row in base-image pixels (sub-pixel).
    float y = 0.f;
    /// Characteristic scale in base-image pixels. For ORB this is the pyramid
    /// scale factor of the level the feature was found on (1 = full
    /// resolution), so the descriptor patch spans 31 * scale base pixels. For
    /// SIFT this is the Gaussian sigma of the scale-space extremum, already
    /// multiplied up to base-image units.
    float scale = 0.f;
    /// Dominant orientation in radians, in [-pi, pi], measured as
    /// `atan2(dI/drow, dI/dcol)` -- i.e. counter-clockwise from the +column
    /// axis in the right-handed (column, row) frame in which +row points
    /// *down* the image. Note this is the negation of OpenCV's angle
    /// convention. The wrap point is +/-pi: the angle is reduced into
    /// (-pi, pi] before being rounded to float, so a value at the wrap can
    /// come back as exactly -pi. Exactly 0 for a perfectly symmetric
    /// neighbourhood.
    float orientation = 0.f;
    /// Detector strength; larger is stronger. FAST arc score for ORB,
    /// |interpolated DoG value| for SIFT.
    float response = 0.f;
    /// Pyramid level (ORB) or octave (SIFT) the feature was detected on.
    /// 0 = full input resolution.
    int octave = 0;
};

/// A 256-bit rBRIEF descriptor, LSB-first: bit @c j lives in
/// `descriptor[j / 8]` at bit position `j % 8`.
using OrbDescriptor = std::array<std::uint8_t, 32>;

/// A 128-dimensional (4 x 4 spatial x 8 orientation) SIFT descriptor,
/// L2-normalised to unit length (see sift_detect_and_compute()).
using SiftDescriptor = std::array<float, 128>;

/// Keypoints plus their descriptors. `keypoints.size() == descriptors.size()`
/// always holds, and element @c i of one describes element @c i of the other.
struct OrbFeatures {
    std::vector<FeatureKeyPoint> keypoints;
    std::vector<OrbDescriptor>   descriptors;
};

/// Keypoints plus their descriptors, same parallel-array contract as OrbFeatures.
struct SiftFeatures {
    std::vector<FeatureKeyPoint> keypoints;
    std::vector<SiftDescriptor>  descriptors;
};

/// One descriptor correspondence from match_descriptors().
struct DescriptorMatch {
    /// Index into the query descriptor vector.
    int query_index = 0;
    /// Index into the train descriptor vector.
    int train_index = 0;
    /// Distance between the two descriptors: an integral Hamming distance in
    /// [0, 256] for OrbDescriptor, a Euclidean (L2) distance for SiftDescriptor.
    float distance = 0.f;
};

/// FAST-9 corner detection (Rosten & Drummond) with 3x3 non-maximum suppression.
///
/// A pixel p is a corner when, on the 16-pixel Bresenham circle of radius 3
/// around it, there exists a **contiguous circular arc of at least 9 pixels**
/// that are all brighter than `I(p) + threshold` or all darker than
/// `I(p) - threshold`. The corner score is the sum over that arc of
/// `|I(k) - I(p)| - threshold`, which is strictly positive for every detected
/// corner and exactly 0 everywhere else.
///
/// @param img Input image. RGB (>= 3 channels) is converted to grayscale
///        first, matching this module's other analysis functions; a 2-channel
///        image uses channel 0.
/// @param threshold Intensity difference required to call a circle pixel
///        brighter/darker, in the same [0, 1] units as the pixel data.
///        Negative values are clamped to 0. Default 0.05 (~13/255).
/// @param nonmax_suppression When true (default), of every 3x3 neighbourhood
///        only the highest-scoring corner survives; ties are broken in favour
///        of the pixel that comes first in raster order, so exactly one corner
///        survives a tie rather than all or none. When false, every pixel
///        passing the arc test is returned.
/// @return Corners as {x = column, y = row, response = arc score} in raster
///         order (increasing row, then increasing column). An empty image, or
///         an image with fewer than 7 rows or 7 columns (the circle radius is
///         3, so the outer 3-pixel frame can never be tested), returns `{}`.
///         A constant image returns `{}`, as does any image once
///         @p threshold reaches the full intensity range.
/// @note FAST-9 by construction cannot fire on an X-junction: a checkerboard
///       corner splits the circle into four ~4-pixel arcs, none of which
///       reaches 9, so a pure checkerboard yields **no** corners. Use harris()
///       or shi_tomasi() for junction-like structure. FAST is a wedge
///       detector: it fires on convex/concave corners whose wedge angle leaves
///       an arc of 9 or more like-signed circle pixels.
/// @note Complexity O(rows*cols) time and O(rows*cols) extra memory for the
///       score map; at most 16 comparisons plus a 16x16 run scan per pixel.
std::vector<KeyPoint> fast_corners(const Image& img, float threshold = 0.05f,
                                   bool nonmax_suppression = true);

/// The fixed 256-entry rBRIEF sampling pattern used by
/// orb_detect_and_compute(), as {x1, y1, x2, y2} column/row offsets relative
/// to the keypoint, every coordinate in [-15, 15] and (x1,y1) != (x2,y2) for
/// every entry.
///
/// The pattern is generated once, on first use, from the seed
/// 0x9E3779B97F4A7C15 using a SplitMix64 integer PRNG and an Irwin-Hall
/// (sum-of-12-uniforms) approximation to a Gaussian with sigma = 6.2 = 31/5,
/// rejecting quadruples that fall outside [-15, 15] or that would sample the
/// same pixel twice. Only exact IEEE-754 add / multiply / divide-by-power-of-
/// two operations are used, so the table is bit-identical on every conforming
/// platform; std::mt19937's distribution objects are deliberately NOT used
/// because their output is not specified to be portable.
/// @return A reference to the single shared table; the same object every call.
/// @note Exposed so callers (and tests) can verify descriptor reproducibility.
const std::array<std::array<int, 4>, 256>& orb_sampling_pattern();

/// ORB: oriented FAST keypoints over a scale pyramid with 256-bit rBRIEF
/// descriptors (Rublee et al. 2011).
///
/// Pipeline, per pyramid level: FAST-9 with non-maximum suppression -> keep
/// the strongest features allotted to that level -> intensity-centroid
/// orientation over a 31x31 circular patch (709 pixels) -> steered BRIEF
/// against the fixed 256-pair sampling pattern of orb_sampling_pattern(),
/// sampled from a Gaussian-blurred copy of the level so that the bits are not
/// dominated by single-pixel noise.
///
/// @param img Input image; RGB is converted to grayscale first.
/// @param max_features Upper bound on the number of returned keypoints,
///        distributed across pyramid levels in geometric proportion
///        (1, 1/s, 1/s^2, ... for scale factor s) so that coarse levels are
///        not starved by the much denser fine levels. `max_features <= 0`
///        returns an empty result.
/// @param fast_threshold FAST-9 threshold in [0, 1] intensity units, see
///        fast_corners().
/// @param n_levels Maximum number of pyramid levels; clamped to [1, 16].
///        Level construction stops early once a level would be smaller than
///        33x33, the smallest size that admits any keypoint given the
///        16-pixel detector border.
/// @param scale_factor Ratio between successive pyramid levels; clamped to
///        [1.01, 4.0]. Level i has size `1 + floor((n-1) / scale_factor^i)`
///        along each axis, so a level pixel at (r, c) maps to base-image
///        (r * scale_factor^i, c * scale_factor^i) exactly. Every level is
///        resampled from level 0 rather than from its predecessor, so
///        interpolation error does not accumulate down the pyramid.
/// @return Keypoints and descriptors in matching order, sorted by descending
///         response (ties broken by ascending octave, then row, then column,
///         so the ordering is a total order and fully reproducible). An empty
///         image, an image smaller than 33x33, or a constant image returns an
///         empty result; a constant image yields no FAST corners at all.
/// @note Deterministic: the sampling pattern is generated once from a fixed
///       seed using integer-only arithmetic (see orb_sampling_pattern()); no
///       unseeded or platform-dependent RNG is used anywhere.
/// @note A perfectly symmetric patch has a zero intensity-centroid vector, and
///       `atan2(+0, +0)` is `+0`, so such a keypoint gets orientation exactly 0
///       and an unsteered descriptor.
/// @note Complexity O(rows*cols) for the pyramid and detection (the geometric
///       series over levels contributes a constant factor of about 3.3 at the
///       default scale factor), plus O(K * 709) for orientations and
///       O(K * 256) for descriptors, K = number of retained keypoints.
OrbFeatures orb_detect_and_compute(const Image& img,
                                   int max_features = 500,
                                   float fast_threshold = 0.05f,
                                   int n_levels = 8,
                                   float scale_factor = 1.2f);

/// SIFT: scale-invariant feature transform (Lowe 2004).
///
/// Builds a Gaussian scale-space of `n_octave_layers + 3` blurred images per
/// octave, forms the difference-of-Gaussian (DoG) stack, finds extrema in the
/// 3x3x3 DoG neighbourhood, refines each to sub-pixel and sub-scale accuracy
/// with a 3D quadratic fit, rejects low-contrast and edge-like responses,
/// assigns one or more dominant orientations from a 36-bin gradient histogram,
/// and emits a 128-dimensional (4x4 spatial x 8 orientation) descriptor built
/// with trilinear interpolation, L2-normalised, clipped at 0.2 and
/// re-normalised.
///
/// @param img Input image; RGB is converted to grayscale first. The input is
///        assumed to already carry a blur of sigma 0.5 (one pixel of camera
///        smoothing), which is deconvolved out when building the octave-0 base.
/// @param max_features Cap on the number of returned keypoints, applied after
///        sorting by descending response; 0 (the default) or any negative
///        value means unlimited.
/// @param n_octave_layers Number of DoG layers per octave that are actually
///        searched for extrema (Lowe's S); clamped to [1, 8]. Default 3.
/// @param contrast_threshold A refined extremum is rejected when
///        `|D(x_hat)| * n_octave_layers < contrast_threshold`. Larger values
///        keep fewer, stronger features. Default 0.04, in [0, 1] intensity
///        units (this module's pixels are already in [0, 1], so no 1/255
///        rescaling is applied anywhere).
/// @param edge_threshold Principal-curvature ratio limit r: a response is
///        rejected when `trace(H)^2 / det(H) >= (r+1)^2 / r` for the 2x2
///        spatial Hessian, or when `det(H) <= 0`. Default 10 (ratio 12.1);
///        a value <= 0 disables the curvature-ratio test but not the
///        `det(H) <= 0` saddle rejection.
/// @param sigma The Gaussian blur of the base image of octave 0. Default 1.6;
///        clamped up to 0.01 so every scale-space blur stays well defined.
/// @return Keypoints and descriptors in matching order, sorted by descending
///         response with ties broken by (octave, row, column, scale,
///         orientation) so the ordering is total and reproducible.
/// @note This implementation does **not** up-sample the input by 2x before
///       building octave 0 (Lowe's optional "octave -1"). Octave 0 is the
///       input resolution, so the finest detectable scale is sigma ~ 1.8 base
///       pixels and roughly half as many keypoints are found as by
///       implementations that do up-sample. Octave indices stay non-negative
///       and coordinates need no half-pixel correction.
/// @note A rotationally symmetric feature legitimately produces several
///       keypoints at the same position and scale differing only in
///       orientation -- that is what the orientation histogram's 80%-of-peak
///       rule prescribes when the histogram has no unique maximum.
/// @note An image whose shorter side is below 16 pixels returns `{}`. A
///       constant image returns `{}`. A pure step edge returns `{}`: the DoG
///       is constant along the edge, so the 3D Hessian is singular and every
///       candidate is rejected by the linear solve (and, were it not, by the
///       edge test).
/// @note Complexity O(rows*cols) per octave for the pyramid and the extrema
///       scan (the octave series contributes a factor below 4/3), plus
///       O(K * radius^2) for orientations and descriptors and O(K^2) for
///       duplicate suppression, K = number of raw keypoints.
/// @note Deterministic: no random numbers are used anywhere.
SiftFeatures sift_detect_and_compute(const Image& img,
                                     int max_features = 0,
                                     int n_octave_layers = 3,
                                     float contrast_threshold = 0.04f,
                                     float edge_threshold = 10.0f,
                                     float sigma = 1.6f);

/// Hamming distance between two 256-bit ORB descriptors: the number of
/// differing bits, in [0, 256]. Symmetric, and 0 exactly when the descriptors
/// are equal. O(32) byte popcounts.
int hamming_distance(const OrbDescriptor& a, const OrbDescriptor& b);

/// Euclidean (L2) distance between two 128-dimensional SIFT descriptors.
/// Symmetric, and 0 exactly when the descriptors are equal. O(128).
float l2_distance(const SiftDescriptor& a, const SiftDescriptor& b);

/// Brute-force nearest-neighbour descriptor matching with Lowe's ratio test
/// and an optional cross-check.
///
/// For each query descriptor the two closest train descriptors are found
/// (distinct indices; ties resolved in favour of the smaller train index).
/// The match is kept only if `d1 < ratio_threshold * d2`, which rejects
/// ambiguous correspondences -- including the degenerate case of two identical
/// train descriptors, where `d1 == d2` and the strict comparison fails. When
/// the train set holds exactly one descriptor there is no second neighbour and
/// the ratio test is skipped (the single match is kept).
///
/// @param query Descriptors to find matches for.
/// @param train Descriptors to search in.
/// @param ratio_threshold Lowe ratio, typically 0.7-0.8; a value >= 1
///        effectively disables the test, a value <= 0 rejects everything.
/// @param cross_check When true, a surviving match (i -> j) is additionally
///        required to be mutual: the raw nearest neighbour of `train[j]` among
///        all queries (no ratio test in the reverse direction, ties to the
///        smaller query index) must be i. This removes many-to-one matches.
/// @return Matches in ascending query_index order. An empty @p query or
///         @p train returns `{}`.
/// @note Complexity O(|query| * |train| * D) with D = 32 byte popcounts (ORB)
///       or 128 float operations (SIFT); the cross-check adds a further
///       O(|matches| * |query| * D).
std::vector<DescriptorMatch> match_descriptors(
    const std::vector<OrbDescriptor>& query,
    const std::vector<OrbDescriptor>& train,
    float ratio_threshold = 0.75f,
    bool cross_check = false);

/// L2 overload of match_descriptors() for SIFT descriptors; identical
/// semantics with Euclidean distance in place of Hamming distance.
std::vector<DescriptorMatch> match_descriptors(
    const std::vector<SiftDescriptor>& query,
    const std::vector<SiftDescriptor>& train,
    float ratio_threshold = 0.75f,
    bool cross_check = false);

// Connected components labelling
std::vector<std::vector<int>> label_components(const Image& bw);
int count_components(const Image& bw);

// ========================== Utilities ==========================
Image im_from_data(int rows, int cols, int channels, const std::vector<float>& data);
float bilinear_sample(const Image& img, float r, float c, int ch);

} // namespace image
} // namespace ms
