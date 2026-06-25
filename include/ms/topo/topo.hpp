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
