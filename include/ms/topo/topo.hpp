#pragma once
#include <limits>
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
    bool has_simplex(const Simplex& s) const;
    // Boundary matrix for dimension k: cols = k-simplices, rows = (k-1)-simplices
    std::vector<std::vector<int>> boundary_matrix(int k) const;
};

// ========================== Vietoris-Rips complex ==========================
// Build Vietoris-Rips complex from a distance matrix at threshold r
SimplicialComplex vietoris_rips(const std::vector<std::vector<double>>& dist_matrix,
                                 double r, int max_dim = 2);

// ========================== Čech complex ==========================
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
//
// @param dist_matrix square, symmetric matrix of pairwise distances (dist_matrix[i][i] == 0)
// @param epsilon      ball radius; a k-simplex is included iff its MEB radius <= epsilon
// @param max_dim      highest simplex dimension to construct (0 = vertices only,
//                      1 = vertices+edges, 2 = vertices+edges+triangles)
// @return SimplicialComplex containing all Čech simplices up to max_dim
// @note Unlike vietoris_rips, the Čech complex is NOT a flag complex: a triangle whose
//       three edges all satisfy the 2ε edge rule need not have MEB radius <= epsilon
//       (an obtuse triangle can have all edges short but a circumradius that exceeds
//       epsilon whenever the pairwise distances are far from acute). Each candidate
//       simplex is therefore checked independently against the true MEB condition.
// @note max_dim > 2 is not supported: computing the minimum enclosing ball of 4+ points
//       from pairwise distances alone (without explicit coordinates) requires solving a
//       higher-dimensional Cayley-Menger / semidefinite feasibility problem (the naive
//       generalization of Welzl's algorithm needs actual point coordinates). Requests
//       with max_dim > 2 are silently clamped to 2 rather than crashing.
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
// Bottleneck distance between two persistence diagrams
double bottleneck_distance(const std::vector<PersistencePair>& dgm1,
                            const std::vector<PersistencePair>& dgm2, int dim = 0);
// Wasserstein distance (p=2) between two persistence diagrams
double wasserstein_distance(const std::vector<PersistencePair>& dgm1,
                             const std::vector<PersistencePair>& dgm2,
                             int dim = 0, int p = 2);

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
