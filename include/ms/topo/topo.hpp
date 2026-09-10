// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace ms {
namespace topo {

// ========================== Simplicial Complex ==========================
// A simplex is a sorted list of vertex indices
using Simplex = std::vector<int>;

class SimplicialComplex {
public:
    // Add a simplex (and all its faces)
    void add_simplex(const Simplex& s);
    // Add points by index
    void add_point(int v);

    // All simplices of a given dimension
    std::vector<Simplex> simplices(int dim) const;
    // All simplices
    const std::vector<Simplex>& all_simplices() const { return simplices_; }
    // Dimension = max simplex size - 1
    int dimension() const;
    // Number of simplices of each dimension
    std::vector<int> simplex_counts() const;
    // Euler characteristic χ = Σ (-1)^k C_k
    int euler_characteristic() const;
    // Betti numbers β_k = rank(H_k) via boundary matrix reduction
    std::vector<int> betti_numbers() const;

private:
    std::vector<Simplex> simplices_;
    // O(log n) membership index mirroring simplices_ exactly (same contents, no
    // ordering role -- simplices_ alone defines insertion order and is what every
    // accessor reads). add_point/add_simplex keep the two in lock-step so that
    // has_simplex() is a lookup instead of a linear scan; without it add_simplex
    // is O(N) per face and complex construction degrades to O(N^2), which
    // dominates every cech_complex build at max_dim >= 3.
    std::set<Simplex> index_;
    bool has_simplex(const Simplex& s) const;
    // Boundary matrix for dimension k: cols = k-simplices, rows = (k-1)-simplices
    std::vector<std::vector<int>> boundary_matrix(int k) const;
};

// ========================== Vietoris-Rips complex ==========================
// Build Vietoris-Rips complex from a distance matrix at threshold r
SimplicialComplex vietoris_rips(const std::vector<std::vector<double>>& dist_matrix,
                                 double r, int max_dim = 2);

// ========================== Čech complex ==========================
// Čech / MEB enumeration ceilings. Same defensive convention as ms::combo's kMaxEnum*
// constants: a request above a ceiling degrades silently instead of allocating without
// bound. See the @note on cech_complex for the exact semantics of each.
inline constexpr int       kMaxCechDim          = 8;      // highest dimension cech_complex will build
inline constexpr int       kMaxMebExactPoints   = 16;     // largest subset given the exact 2^m support search
inline constexpr int       kMebApproxIterations = 500;    // Badoiu-Clarkson steps above that ceiling
inline constexpr long long kMaxCechCandidates   = 50000;  // dim>=3 candidate simplices evaluated per call

// Minimum enclosing ball (MEB) radius of the points indexed by `idx`, computed from
// the distance matrix alone — no coordinates required, correct for any subset size.
//
// Method: the MEB of a finite set S is the circumball of some subset T ⊆ S whose
// circumcentre lies in conv(T) and which encloses every point of S (the standard
// optimality condition c ∈ conv(S ∩ ∂B); Carathéodory lets T be taken affinely
// independent, so such a T is always reachable by the enumeration below). Both tests
// are distance-only. Writing the circumcentre in barycentric form c = Σ λ_a q_a with
// Σ λ_a = 1, the bordered (|T|+1)x(|T|+1) Cayley-Menger system
//        [ 0  1^T ] [ v0 ]   [ 1 ]
//        [ 1   D  ] [ λ  ] = [ 0 ]      (D = squared pairwise distances of T)
// yields the barycentric circumcentre λ directly, plus R^2 = -v0/2 — equivalently, in
// determinant form, R^2 = -det(D) / (2 det(CM)). Containment of any point x then
// follows from the coordinate-free identity |c - x|^2 = Σ_a λ_a d(q_a,x)^2 - R^2, i.e.
// x is inside the circumball of T iff Σ_a λ_a d(q_a,x)^2 <= 2 R^2.
// All λ_a >= 0 means the circumcentre lies inside conv(T), so the MEB equals its
// circumball; otherwise the search moves on to larger support sets and takes the
// smallest enclosing ball that still contains every point of `idx`.
//
// Sizes 1..3 are answered by the closed forms this module already used, bit for bit:
// 0 for a single point, dist/2 for a pair, and the acute/obtuse circumradius rule for a
// triangle (the same static helper cech_complex and alpha_complex call, including its
// absolute obtuseness tolerance). From size 4 the support-set search above runs; above
// kMaxMebExactPoints a Badoiu-Clarkson core-set iteration replaces it, which does not
// under-report: its value is the covering radius of a genuine enclosing ball, hence an
// upper bound on the true MEB radius up to floating-point round-off.
//
// @param dist_matrix square, symmetric matrix of pairwise distances
// @param idx         indices into dist_matrix selecting the subset; order is irrelevant
// @return MEB radius of the indexed subset
// @note Defensive, non-throwing on malformed input (this module returns plain values,
//       not Result): an empty or single-element idx, an index outside
//       [0, dist_matrix.size()), or a ragged/non-square dist_matrix all return 0.0.
//       Affinely dependent subsets (coplanar quadruples, collinear triples, duplicated
//       points) make the Cayley-Menger system singular; those subsets are skipped and
//       the correct MEB is recovered from a smaller support set — e.g. four coplanar
//       square corners give diagonal/2, four collinear points give span/2, and
//       coincident points give 0.
double meb_radius_from_distances(const std::vector<std::vector<double>>& dist_matrix,
                                 const std::vector<int>& idx);

// Build the Čech complex of a point cloud (given as a distance matrix) at scale epsilon.
//
// Definition: the Čech complex Čech(ε) contains a k-simplex {v0,...,vk} iff the
// closed balls of radius ε centered at each v_i have a common (non-empty) intersection.
// By the Nerve Lemma / Helly-type argument this holds iff the minimum enclosing ball
// (MEB) of the k+1 points has radius ≤ ε.
//   - dim 0 (vertices): always included.
//   - dim 1 (edges {i,j}): balls of radius ε around i and j intersect iff
//     dist(i,j) ≤ 2ε, so the MEB radius of the pair is dist(i,j)/2.
//   - dim 2 (triangles {i,j,k}): MEB radius = circumradius R = (a*b*c)/(4*Area)
//     (Heron's formula for Area from side lengths a,b,c) if the triangle is acute
//     or right; if it is obtuse, the MEB is the smallest ball spanning the longest
//     side alone, i.e. radius = longest_side/2 (the opposite vertex lies inside it).
//     Obtuseness (at the vertex opposite the longest side) is detected by comparing
//     longest_side^2 against the sum of the other two sides squared.
//   - dim k >= 3 (k+1 points): MEB radius via meb_radius_from_distances (above), which
//     solves the same problem exactly from the distance matrix alone using the bordered
//     Cayley-Menger system. There is no coordinate requirement and no approximation:
//     for k+1 <= kMaxMebExactPoints the answer is the true MEB radius.
//
// @param dist_matrix square, symmetric matrix of pairwise distances (dist_matrix[i][i] == 0)
// @param epsilon      ball radius; a k-simplex is included iff its MEB radius <= epsilon
// @param max_dim      highest simplex dimension to construct (0 = vertices only,
//                      1 = vertices+edges, 2 = vertices+edges+triangles, and so on up
//                      to kMaxCechDim)
// @return SimplicialComplex containing all Čech simplices up to max_dim
// @note Unlike vietoris_rips, the Čech complex is NOT a flag complex: a triangle whose
//       three edges all satisfy the 2ε edge rule need not have MEB radius <= epsilon
//       (an obtuse triangle can have all edges short but a circumradius that exceeds
//       epsilon whenever the pairwise distances are far from acute). Each candidate
//       simplex is therefore checked independently against the true MEB condition.
// @note Dimensions >= 3 ARE supported. The historical restriction ("needs actual point
//       coordinates") was unnecessary: the circumradius R and the barycentric
//       circumcentre coordinates λ of a k-simplex both fall out of one bordered
//       Cayley-Menger linear system built purely from squared pairwise distances, and
//       Welzl's "is the centre inside the simplex / is point p inside the ball" tests
//       are then expressible in distance-only form as well. See the derivation above
//       meb_radius_from_distances in src/topo/topo.cpp.
// @note Enumeration is bounded, not unbounded. Čech(ε) considers every (k+1)-subset of
//       the n points, i.e. C(n, k+1) candidates at dimension k, so an unguarded max_dim
//       would let a stray REPL argument or a fuzzer input build an astronomically large
//       complex — the failure mode ms::combo guards with its kMaxEnum* constants. Two
//       ceilings apply, both degrading silently rather than erroring:
//         * max_dim > kMaxCechDim is clamped to kMaxCechDim (max_dim < 0 behaves as 0,
//           exactly as it does today: only vertices and the unconditional edge pass run);
//         * at most kMaxCechCandidates candidate simplices of dimension >= 3 are
//           evaluated. Dimensions 0..2 are never budgeted, and survivors are committed
//           one whole dimension at a time, so the returned complex is always COMPLETE
//           through its own top dimension — a truncated build simply stops lower than
//           requested, and its Euler characteristic and Betti numbers stay meaningful.
//       Higher dimensions are cheap in practice because the MEB radius is monotone
//       under supersets: Čech(ε) is closed under faces, so dimension k+1 is expanded
//       only from the simplices that survived dimension k, and only when every one of a
//       candidate's facets survived.
SimplicialComplex cech_complex(const std::vector<std::vector<double>>& dist_matrix,
                                double epsilon, int max_dim = 2);

// ========================== Alpha complex ==========================
// Build the alpha complex of a 2D Euclidean point cloud at scale alpha.
//
// Definition: the alpha complex Alpha(α) is the subcomplex of the Delaunay
// triangulation of `pts` containing exactly those Delaunay simplices whose
// minimum enclosing ball (MEB) radius is <= α. This is the same MEB-radius
// inclusion rule used by cech_complex, but restricted to simplices that are
// actually part of the Delaunay triangulation, rather than every possible
// k-subset of points — Alpha(α) is therefore always a subcomplex of Čech(α)
// on the same points (Alpha(α) ⊆ Čech(α) for all α, since every Delaunay
// simplex is trivially also a candidate simplex for Čech). Unlike
// cech_complex / vietoris_rips this takes explicit 2D coordinates, not a
// distance matrix, since building the Delaunay triangulation requires actual
// point positions.
//   - dim 0 (vertices): always included.
//   - dim 2 (triangles {i,j,k}): only Delaunay triangles are candidates. A
//     Delaunay triangle is included iff its circumradius (obtuse-triangle
//     MEB convention identical to cech_complex, computed here directly from
//     the 3 vertex coordinates instead of a precomputed distance matrix) is
//     <= alpha.
//   - dim 1 (edges {i,j}): only genuine Delaunay edges (edges of at least
//     one Delaunay triangle, or the single connecting edge when there are
//     only 2 points and no triangulation is possible) are candidates. An
//     edge is included iff it is a face of an included triangle, OR — for
//     edges on the convex hull that border only one Delaunay triangle (or,
//     in degenerate/collinear inputs, no triangle at all) — its own MEB
//     radius (half its length) is <= alpha.
//
// @param pts    2D point cloud; pts[i] = {x, y}
// @param alpha  ball radius; a Delaunay simplex is included iff its MEB
//                radius <= alpha
// @param max_dim highest simplex dimension to construct (0 = vertices only,
//                 1 = vertices+edges, 2 = vertices+edges+triangles)
// @return SimplicialComplex containing all alpha-complex simplices up to max_dim
// @note This is a 2D-only construction: the Delaunay triangulation is built with
//       ms::geo::delaunay_2d, so pts must be 2D and max_dim > 2 is silently
//       clamped to 2 (there is no dim-3 Delaunay structure to draw from here).
// @note Simplification (documented, standard in scope): for an *interior* Delaunay
//       edge shared by two triangles, this implementation includes it only via
//       face-closure of an included adjacent triangle — it does not separately
//       re-admit such an edge on its own MEB radius when both adjacent triangles
//       are excluded. This matches the common simplified alpha-complex treatment
//       and is exact for hull edges (which have only one adjacent triangle, so
//       their own MEB radius is the correct alpha value); a fully rigorous
//       treatment of interior edges would additionally consider the smallest
//       empty circle through the edge's two endpoints, which is not computed here.
// @note Degenerate inputs (fewer than 3 points, duplicate/coincident points,
//       collinear points) are handled defensively rather than throwing: fewer
//       than 2 points yields vertices only; exactly 2 points yields the single
//       edge if its half-length <= alpha; collinear or duplicate points may make
//       ms::geo::delaunay_2d produce degenerate (near-zero-area) triangles, whose
//       circumradius falls back to half of their longest side (same convention
//       as cech_complex's degenerate-triangle handling).
SimplicialComplex alpha_complex(const std::vector<std::vector<double>>& pts,
                                  double alpha, int max_dim = 2);

// ========================== Witness complex ==========================
// Build the weak witness complex of a point cloud at filtration scale max_epsilon,
// using a landmark subset selected by index into `points`.
//
// Definition (weak witness, de Silva–Carlsson style, practical formulation):
// Let L = { landmark_indices[i] } be the landmark vertex set (vertices of the
// complex are these original point-cloud indices, not relabelled 0..|L|-1).
// For each witness point w in the full point cloud (every point may witness,
// including landmarks themselves), order landmarks by increasing distance d(w,·).
// A k-simplex σ = {ℓ_0,…,ℓ_k} (the k+1 closest landmarks to w) is included in
// Wit(max_epsilon) iff d(w, ℓ_k) ≤ max_epsilon — i.e. the farthest vertex of σ
// lies within the filtration radius. Equivalently: ∃ witness w such that every
// landmark in σ is among the |σ| closest landmarks to w and all are within
// max_epsilon of w; no landmark outside σ may be strictly closer to w than the
// farthest landmark in σ.
//
// This is the standard "weak" (0,ν)-witness rule with ν = max_epsilon; it is
// an approximation of the Vietoris–Rips complex on L and is always a subcomplex
// of VR(L, max_epsilon) when witnesses = landmarks = all points, but can be a
// proper subset in general because the witness condition is stricter than pairwise
// proximity alone.
//
// @param points           Euclidean point cloud; points[i] = coordinate vector
// @param landmark_indices indices into `points` selecting the landmark vertex set
// @param max_epsilon      filtration radius ν; a k-simplex is included iff some
//                         witness w has its (k+1)-th nearest landmark within ν
// @param max_dim          highest simplex dimension to construct (0..2 supported;
//                         values > 2 are silently clamped to 2)
// @return SimplicialComplex on landmark vertex indices (original point indices)
// @note Defensive edge cases: empty landmarks → empty complex; invalid landmark
//       indices are skipped; n_landmarks < 2 yields vertices only (no edges).
std::vector<int> select_landmarks_maxmin(const std::vector<std::vector<double>>& points,
                                          int n_landmarks, int seed_index = 0);

SimplicialComplex witness_complex(const std::vector<std::vector<double>>& points,
                                   const std::vector<int>& landmark_indices,
                                   double max_epsilon, int max_dim = 2);

// ========================== Persistence Diagram ==========================
struct PersistencePair {
    int dim;
    double birth, death;
    bool is_essential() const { return death == std::numeric_limits<double>::infinity(); }
};

// Persistent homology from a filtered simplicial complex
// filtration: list of (simplex, birth_time) in order of appearance
std::vector<PersistencePair>
persistence_diagram(const std::vector<std::pair<Simplex,double>>& filtration);

// ========================== Distances between diagrams ==========================
// Bottleneck distance between two persistence diagrams (greedy O(n^2) matching;
// exact bottleneck requires Hungarian assignment).
double bottleneck_distance(const std::vector<PersistencePair>& dgm1,
                            const std::vector<PersistencePair>& dgm2, int dim = 0);
// Wasserstein distance (default p=2) between two persistence diagrams (greedy matching).
double wasserstein_distance(const std::vector<PersistencePair>& dgm1,
                             const std::vector<PersistencePair>& dgm2,
                             int dim = 0, int p = 2);

// ========================== Persistence Landscape ==========================
// Persistence landscape: a stable, vectorized functional summary of a persistence
// diagram (Bubenik 2015), widely used to turn topological summaries into feature
// vectors usable by standard statistical/ML methods (unlike the diagram itself,
// landscapes live in a vector space with a well-defined mean/distance, addressing
// a fundamental limitation of raw persistence diagrams).
//
// For each persistence pair (birth b, death d) -- only non-essential pairs, i.e.
// finite death, are used; essential/infinite-death pairs are excluded (see
// is_essential()), matching the convention bottleneck_distance/wasserstein_distance
// already use when consuming a persistence diagram -- define the tent/triangle
// function:
//   Lambda_{(b,d)}(t) = max(0, min(t - b, d - t))   for t in [b, d], 0 elsewhere
// (this peaks at height (d-b)/2 at the midpoint t=(b+d)/2, and is 0 at and outside [b,d]).
//
// The k-th persistence landscape function lambda_k(t) (for k = 1, 2, 3, ...) is the
// k-th LARGEST value among {Lambda_{(b,d)}(t) : (b,d) in the diagram} at each t (i.e.
// lambda_1 is the pointwise max/envelope over all tent functions, lambda_2 is the
// second-highest at each t, etc.). This function computes the first n_layers landscape
// functions, each sampled at n_samples evenly-spaced points over [t_min, t_max].
//
// @param diagram persistence pairs (e.g. from persistence_diagram). Only finite-death
//        pairs contribute; essential (infinite-death) pairs are excluded.
// @param n_layers number of landscape layers (k=1..n_layers) to compute. n_layers<=0
//        returns {}. If the diagram has fewer than k tents nonzero at some t, lambda_k
//        is simply 0 there (and possibly identically 0 everywhere) -- this is the
//        mathematically correct convention, not an error.
// @param n_samples number of evenly-spaced t-samples per layer. n_samples<=1 returns {}.
// @param t_min, t_max sampling range. If t_min>=t_max (including the sentinel default
//        of 0.0/0.0), the range is instead auto-derived from the diagram's own
//        [min birth, max death] over its finite-death pairs, as a convenience default.
// @return n_layers vectors, each of length n_samples: result[k-1][i] = lambda_k(t_i)
//         for the i-th sample point t_i in the (possibly auto-derived) range.
std::vector<std::vector<double>>
persistence_landscape(const std::vector<PersistencePair>& diagram,
                       int n_layers, int n_samples,
                       double t_min = 0.0, double t_max = 0.0);

// ========================== Utilities ==========================

// Build distance matrix from point cloud (Euclidean)
std::vector<std::vector<double>>
pairwise_distances(const std::vector<std::vector<double>>& pts);

// Betti numbers from a distance matrix at given thresholds
std::vector<std::vector<int>>
betti_curve(const std::vector<std::vector<double>>& dist_matrix,
            const std::vector<double>& thresholds, int max_dim = 2);

} // namespace topo
} // namespace ms
