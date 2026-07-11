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
