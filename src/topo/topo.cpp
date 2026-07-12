#include "ms/topo/topo.hpp"
#include "ms/geo/geo.hpp"
#include <bit>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

namespace ms {
namespace topo {

// ========================== SimplicialComplex ==========================

bool SimplicialComplex::has_simplex(const Simplex& s) const {
    for (auto& t : simplices_) if (t == s) return true;
    return false;
}

void SimplicialComplex::add_point(int v) {
    Simplex s = {v};
    if (!has_simplex(s)) simplices_.push_back(s);
}

void SimplicialComplex::add_simplex(const Simplex& s_in) {
    Simplex s = s_in;
    std::sort(s.begin(), s.end());
    if (has_simplex(s)) return;
    simplices_.push_back(s);
    // Add all faces (subsets)
    int n = static_cast<int>(s.size());
    for (int mask = 0; mask < (1<<n); ++mask) {
        if (std::popcount(static_cast<unsigned>(mask)) == n) continue;  // full set already added
        Simplex face;
        for (int i = 0; i < n; ++i) if (mask & (1<<i)) face.push_back(s[i]);
        if (!face.empty() && !has_simplex(face)) simplices_.push_back(face);
    }
}

std::vector<Simplex> SimplicialComplex::simplices(int dim) const {
    std::vector<Simplex> res;
    for (auto& s : simplices_)
        if ((int)s.size() == dim+1) res.push_back(s);
    return res;
}

int SimplicialComplex::dimension() const {
    int d = -1;
    for (auto& s : simplices_) d = std::max(d, (int)s.size()-1);
    return d;
}

std::vector<int> SimplicialComplex::simplex_counts() const {
    int D = dimension();
    if (D < 0) return {};
    std::vector<int> counts(D+1, 0);
    for (auto& s : simplices_) counts[s.size()-1]++;
    return counts;
}

int SimplicialComplex::euler_characteristic() const {
    auto counts = simplex_counts();
    int chi = 0;
    for (int k = 0; k < (int)counts.size(); ++k)
        chi += (k%2==0 ? 1 : -1) * counts[k];
    return chi;
}

// Boundary matrix for dimension k
// Columns are k-simplices, rows are (k-1)-simplices
// Entry (i,j) = 1 if (k-1)-simplex i is a face of k-simplex j (with sign ignored for Z/2)
std::vector<std::vector<int>> SimplicialComplex::boundary_matrix(int k) const {
    auto k_simps   = simplices(k);
    auto km1_simps = simplices(k-1);
    int rows = static_cast<int>(km1_simps.size());
    int cols = static_cast<int>(k_simps.size());
    std::vector<std::vector<int>> mat(rows, std::vector<int>(cols, 0));
    for (int j = 0; j < cols; ++j) {
        auto& sigma = k_simps[j];
        // Each face is obtained by deleting one vertex
        for (int del = 0; del < (int)sigma.size(); ++del) {
            Simplex face;
            for (int l = 0; l < (int)sigma.size(); ++l)
                if (l != del) face.push_back(sigma[l]);
            // Find face index in km1_simps
            for (int i = 0; i < rows; ++i)
                if (km1_simps[i] == face) { mat[i][j] = 1; break; }
        }
    }
    return mat;
}

// Smith normal form over Z/2 (F2) — reduce boundary matrix to echelon form
// Returns rank
static int f2_rank(std::vector<std::vector<int>> mat) {
    int rows = static_cast<int>(mat.size());
    int cols = rows > 0 ? static_cast<int>(mat[0].size()) : 0;
    int rank = 0;
    int pivot_row = 0;
    for (int col = 0; col < cols && pivot_row < rows; ++col) {
        // Find pivot in this column
        int pr = -1;
        for (int row = pivot_row; row < rows; ++row)
            if (mat[row][col]) { pr = row; break; }
        if (pr < 0) continue;
        std::swap(mat[pivot_row], mat[pr]);
        // Eliminate all other rows
        for (int row = 0; row < rows; ++row) {
            if (row != pivot_row && mat[row][col]) {
                for (int c = 0; c < cols; ++c)
                    mat[row][c] ^= mat[pivot_row][c];
            }
        }
        ++rank; ++pivot_row;
    }
    return rank;
}

// Betti numbers: β_k = Z_k - B_k = (C_k - rank(∂_k)) - rank(∂_{k+1})
std::vector<int> SimplicialComplex::betti_numbers() const {
    int D = dimension();
    if (D < 0) return {};
    auto counts = simplex_counts();
    std::vector<int> betti(D+1, 0);
    // Compute ranks of boundary matrices
    std::vector<int> ranks(D+2, 0);
    for (int k = 1; k <= D; ++k) {
        auto bm = boundary_matrix(k);
        if (!bm.empty() && !bm[0].empty())
            ranks[k] = f2_rank(bm);
    }
    for (int k = 0; k <= D; ++k) {
        int C_k = counts[k];
        int rank_dk   = (k <= D) ? ranks[k] : 0;   // ∂_k
        int rank_dk1  = (k+1 <= D) ? ranks[k+1] : 0; // ∂_{k+1}
        betti[k] = C_k - rank_dk - rank_dk1;
        if (betti[k] < 0) betti[k] = 0;
    }
    return betti;
}

// ========================== Vietoris-Rips ==========================

SimplicialComplex vietoris_rips(const std::vector<std::vector<double>>& dist_matrix,
                                 double r, int max_dim) {
    int n = static_cast<int>(dist_matrix.size());
    SimplicialComplex sc;
    // Add all points
    for (int i = 0; i < n; ++i) sc.add_point(i);
    // Add edges
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            if (dist_matrix[i][j] <= r) sc.add_simplex({i,j});
    // Add higher simplices
    if (max_dim >= 2) {
        // For each triple
        for (int i=0; i<n; ++i)
            for (int j=i+1; j<n; ++j) if (dist_matrix[i][j]<=r)
                for (int k=j+1; k<n; ++k)
                    if (dist_matrix[i][k]<=r && dist_matrix[j][k]<=r)
                        sc.add_simplex({i,j,k});
    }
    if (max_dim >= 3) {
        for (int i=0;i<n;++i)
            for (int j=i+1;j<n;++j) if (dist_matrix[i][j]<=r)
                for (int k=j+1;k<n;++k) if (dist_matrix[i][k]<=r && dist_matrix[j][k]<=r)
                    for (int l=k+1;l<n;++l)
                        if (dist_matrix[i][l]<=r && dist_matrix[j][l]<=r && dist_matrix[k][l]<=r)
                            sc.add_simplex({i,j,k,l});
    }
    return sc;
}

// ========================== Čech complex ==========================

// Minimum enclosing ball radius of a triangle given its three side lengths.
// If the triangle is acute/right, this is the circumradius R = (a*b*c)/(4*Area).
// If obtuse, the MEB is centered on the longest side, radius = longest_side/2.
// Degenerate/near-degenerate triangles (zero or ~zero area) fall back to
// half the longest side, which is still a valid enclosing ball radius.
static double min_enclosing_ball_radius_triangle(double a, double b, double c) {
    // Sort so that `c` is the longest side.
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);

    // Obtuse (or right) at the vertex opposite the longest side c:
    // c^2 >= a^2 + b^2  =>  MEB is just the longest side's midpoint/radius.
    if (c*c >= a*a + b*b - 1e-12) {
        return c / 2.0;
    }

    // Acute triangle: circumradius via Heron's formula.
    double s = (a + b + c) / 2.0;
    double area_sq = s * (s-a) * (s-b) * (s-c);
    if (area_sq <= 0.0) return c / 2.0;  // degenerate: collinear points
    double area = std::sqrt(area_sq);
    return (a * b * c) / (4.0 * area);
}

SimplicialComplex cech_complex(const std::vector<std::vector<double>>& dist_matrix,
                                double epsilon, int max_dim) {
    int n = static_cast<int>(dist_matrix.size());
    int dim_cap = std::min(max_dim, 2);  // see @note: k>=3 not supported from distances alone
    SimplicialComplex sc;

    // dim 0: vertices are always included (a single ball trivially "intersects itself").
    for (int i = 0; i < n; ++i) sc.add_point(i);

    // dim 1: edge {i,j} iff balls of radius epsilon around i,j intersect, i.e. dist(i,j) <= 2*epsilon.
    // (Mirrors vietoris_rips: edges are always considered regardless of max_dim.)
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            if (dist_matrix[i][j] <= 2.0 * epsilon) sc.add_simplex({i, j});

    // dim 2: triangle {i,j,k} iff its minimum enclosing ball radius <= epsilon.
    // Checked independently per triple -- NOT inferred from the 1-skeleton.
    if (dim_cap >= 2) {
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j)
                for (int k = j+1; k < n; ++k) {
                    double a = dist_matrix[i][j];
                    double b = dist_matrix[j][k];
                    double c = dist_matrix[i][k];
                    double r = min_enclosing_ball_radius_triangle(a, b, c);
                    if (r <= epsilon) sc.add_simplex({i, j, k});
                }
    }

    return sc;
}

// ========================== Alpha complex ==========================

SimplicialComplex alpha_complex(const std::vector<std::vector<double>>& pts,
                                  double alpha, int max_dim) {
    int n = static_cast<int>(pts.size());
    int dim_cap = std::min(max_dim, 2);  // see @note: only a 2D Delaunay structure exists
    SimplicialComplex sc;

    // dim 0: vertices are always included, regardless of alpha.
    for (int i = 0; i < n; ++i) sc.add_point(i);
    if (n < 2) return sc;

    auto edist = [&](int i, int j) {
        double dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
        return std::sqrt(dx*dx + dy*dy);
    };

    // With only 2 points there is no Delaunay triangulation to build; the lone
    // segment joining them is the trivial 1-dimensional "triangulation", whose
    // own MEB radius (half its length) is its alpha value.
    if (n == 2) {
        if (edist(0, 1) / 2.0 <= alpha) sc.add_simplex({0, 1});
        return sc;
    }

    std::vector<ms::geo::Point2D> gpts(n);
    for (int i = 0; i < n; ++i) gpts[i] = {pts[i][0], pts[i][1]};
    auto tris = ms::geo::delaunay_2d(gpts);

    // Per Delaunay edge: how many Delaunay triangles reference it, and whether at
    // least one of those triangles was included in the alpha complex. Edges that
    // border only one triangle (convex-hull edges) or -- in degenerate/collinear
    // inputs -- no triangle at all are the ones whose own MEB radius is checked
    // independently below (see @note on the documented interior-edge simplification).
    struct EdgeInfo { int triangle_count = 0; bool any_included = false; };
    std::map<std::pair<int,int>, EdgeInfo> edge_info;
    auto edge_key = [](int i, int j) { return i < j ? std::make_pair(i, j) : std::make_pair(j, i); };

    for (auto& t : tris) {
        double a = edist(t.a, t.b);
        double b = edist(t.b, t.c);
        double c = edist(t.a, t.c);
        double r = min_enclosing_ball_radius_triangle(a, b, c);
        bool included = (dim_cap >= 2) && (r <= alpha);
        if (included) sc.add_simplex({t.a, t.b, t.c});  // also adds its 3 edges + 3 vertices

        std::pair<int,int> keys[3] = {edge_key(t.a,t.b), edge_key(t.b,t.c), edge_key(t.a,t.c)};
        for (auto& k : keys) {
            auto& info = edge_info[k];
            info.triangle_count++;
            info.any_included = info.any_included || included;
        }
    }

    // dim 1: edges are always considered regardless of max_dim (mirrors cech_complex).
    // An edge already added as a face of an included triangle is left alone; a
    // hull/orphan edge (triangle_count <= 1) is independently tested against its
    // own MEB radius (half its length).
    for (auto& [key, info] : edge_info) {
        if (info.any_included) continue;
        if (info.triangle_count <= 1 && edist(key.first, key.second) / 2.0 <= alpha)
            sc.add_simplex({key.first, key.second});
    }

    return sc;
}

// ========================== Witness complex ==========================

// Squared Euclidean distance between two coordinate vectors (dimension-agnostic).
static double sq_dist_pts(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0;
    size_t dim = std::min(a.size(), b.size());
    for (size_t k = 0; k < dim; ++k) {
        double diff = a[k] - b[k];
        d += diff * diff;
    }
    return d;
}

static double dist_pts(const std::vector<double>& a, const std::vector<double>& b) {
    return std::sqrt(sq_dist_pts(a, b));
}

std::vector<int> select_landmarks_maxmin(const std::vector<std::vector<double>>& points,
                                          int n_landmarks, int seed_index) {
    int n = static_cast<int>(points.size());
    if (n <= 0 || n_landmarks <= 0) return {};
    n_landmarks = std::min(n_landmarks, n);

    std::vector<int> landmarks;
    landmarks.reserve(static_cast<size_t>(n_landmarks));
    std::vector<bool> selected(static_cast<size_t>(n), false);

    int seed = (n > 0) ? (seed_index % n + n) % n : 0;
    landmarks.push_back(seed);
    selected[static_cast<size_t>(seed)] = true;

    while (static_cast<int>(landmarks.size()) < n_landmarks) {
        int best = -1;
        double best_min_dist = -1.0;
        for (int i = 0; i < n; ++i) {
            if (selected[static_cast<size_t>(i)]) continue;
            double min_d = std::numeric_limits<double>::infinity();
            for (int lm : landmarks) {
                min_d = std::min(min_d, dist_pts(points[i], points[lm]));
            }
            if (min_d > best_min_dist) {
                best_min_dist = min_d;
                best = i;
            }
        }
        if (best < 0) break;
        landmarks.push_back(best);
        selected[static_cast<size_t>(best)] = true;
    }
    return landmarks;
}

SimplicialComplex witness_complex(const std::vector<std::vector<double>>& points,
                                   const std::vector<int>& landmark_indices,
                                   double max_epsilon, int max_dim) {
    int dim_cap = std::min(max_dim, 2);
    SimplicialComplex sc;

    // Collect valid, distinct landmark vertex indices (original point-cloud indices).
    std::vector<int> landmarks;
    std::vector<bool> seen;
    if (!points.empty()) seen.assign(points.size(), false);
    for (int idx : landmark_indices) {
        if (idx < 0 || idx >= static_cast<int>(points.size())) continue;
        if (seen[static_cast<size_t>(idx)]) continue;
        seen[static_cast<size_t>(idx)] = true;
        landmarks.push_back(idx);
    }
    int m = static_cast<int>(landmarks.size());
    if (m == 0) return sc;

    for (int v : landmarks) sc.add_point(v);

    if (m < 2 || max_epsilon < 0.0) return sc;

    // For each witness (every point in the cloud), derive witnessed simplices.
    int n = static_cast<int>(points.size());
    for (int w = 0; w < n; ++w) {
        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(static_cast<size_t>(m));
        for (int lm : landmarks)
            ranked.push_back({dist_pts(points[w], points[lm]), lm});
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first < b.first;
                      return a.second < b.second;
                  });

        int max_k = std::min(dim_cap, m - 1);
        for (int k = 0; k <= max_k; ++k) {
            if (ranked[static_cast<size_t>(k)].first > max_epsilon) break;
            Simplex sigma;
            sigma.reserve(static_cast<size_t>(k + 1));
            for (int j = 0; j <= k; ++j)
                sigma.push_back(ranked[static_cast<size_t>(j)].second);
            sc.add_simplex(sigma);
        }
    }

    return sc;
}

// ========================== Persistent Homology ==========================
// Simplified persistence using reduction algorithm (Z/2 coefficients)

std::vector<PersistencePair>
persistence_diagram(const std::vector<std::pair<Simplex,double>>& filtration) {
    int n = static_cast<int>(filtration.size());
    std::vector<PersistencePair> pairs;

    // Build index map from sorted simplex → filtration index
    std::map<Simplex, int> simplex_idx;
    for (int i = 0; i < n; ++i) {
        Simplex s = filtration[i].first;
        std::sort(s.begin(), s.end());
        simplex_idx[s] = i;
    }

    // For each simplex, dimension and boundary
    std::vector<int> dims(n);
    std::vector<std::vector<int>> boundaries(n);  // sorted indices
    for (int i = 0; i < n; ++i) {
        Simplex s = filtration[i].first;
        std::sort(s.begin(), s.end());
        dims[i] = static_cast<int>(s.size()) - 1;
        if (dims[i] > 0) {
            for (int del = 0; del < (int)s.size(); ++del) {
                Simplex face;
                for (int l = 0; l < (int)s.size(); ++l) if (l!=del) face.push_back(s[l]);
                auto it = simplex_idx.find(face);
                if (it != simplex_idx.end()) boundaries[i].push_back(it->second);
            }
            std::sort(boundaries[i].begin(), boundaries[i].end());
        }
    }

    // Standard reduction algorithm
    std::vector<std::vector<int>> reduced(n);  // columns as sorted lists
    for (int i = 0; i < n; ++i) reduced[i] = boundaries[i];
    std::vector<int> pivot_to_col(n, -1);

    auto low = [](const std::vector<int>& col) -> int {
        return col.empty() ? -1 : col.back();
    };
    auto add_col = [](std::vector<int>& a, const std::vector<int>& b) {
        // XOR (symmetric difference) of two sorted lists
        std::vector<int> result;
        int i=0, j=0;
        while (i<(int)a.size() && j<(int)b.size()) {
            if (a[i]<b[j]) result.push_back(a[i++]);
            else if (b[j]<a[i]) result.push_back(b[j++]);
            else { ++i; ++j; }  // cancel
        }
        while (i<(int)a.size()) result.push_back(a[i++]);
        while (j<(int)b.size()) result.push_back(b[j++]);
        a = result;
    };

    for (int i = 0; i < n; ++i) {
        while (!reduced[i].empty()) {
            int p = low(reduced[i]);
            if (pivot_to_col[p] == -1) {
                pivot_to_col[p] = i;
                break;
            }
            add_col(reduced[i], reduced[pivot_to_col[p]]);
        }
        if (!reduced[i].empty()) {
            int j = low(reduced[i]);  // j dies when i is added
            pairs.push_back({dims[j], filtration[j].second, filtration[i].second});
        }
    }
    // Essential cycles (never paired)
    std::set<int> paired;
    for (auto& p : pairs) {
        // find birth index
        for (int i=0;i<n;++i) {
            if (filtration[i].second == p.birth && dims[i] == p.dim) { paired.insert(i); break; }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (reduced[i].empty() && paired.find(i) == paired.end()) {
            // Check if this is truly a cycle (not a boundary)
            if (boundaries[i].empty() || !reduced[i].empty()) continue;
            pairs.push_back({dims[i], filtration[i].second,
                             std::numeric_limits<double>::infinity()});
        }
    }
    return pairs;
}

// ========================== Distances ==========================

double bottleneck_distance(const std::vector<PersistencePair>& dgm1,
                            const std::vector<PersistencePair>& dgm2, int dim) {
    // Filter by dimension
    std::vector<std::pair<double,double>> pts1, pts2;
    for (auto& p : dgm1) if (p.dim==dim && !p.is_essential()) pts1.push_back({p.birth,p.death});
    for (auto& p : dgm2) if (p.dim==dim && !p.is_essential()) pts2.push_back({p.birth,p.death});

    // Bottleneck: optimal matching minimising max cost
    // Distance from a point (b,d) to diagonal is (d-b)/2
    // Simple O(n^2) approximation
    double result = 0.0;
    // Cost between two persistence points
    auto cost = [](std::pair<double,double> a, std::pair<double,double> b) {
        return std::max(std::abs(a.first-b.first), std::abs(a.second-b.second));
    };
    auto diag_cost = [](std::pair<double,double> a) {
        return (a.second - a.first) / 2.0;
    };

    // Greedy matching
    std::vector<bool> used2(pts2.size(), false);
    for (auto& p : pts1) {
        double best = diag_cost(p);
        int best_j = -1;
        for (int j=0; j<(int)pts2.size(); ++j) {
            if (used2[j]) continue;
            double c = cost(p, pts2[j]);
            if (c < best) { best = c; best_j = j; }
        }
        if (best_j >= 0) used2[best_j] = true;
        result = std::max(result, best);
    }
    for (int j=0; j<(int)pts2.size(); ++j)
        if (!used2[j]) result = std::max(result, diag_cost(pts2[j]));
    return result;
}

double wasserstein_distance(const std::vector<PersistencePair>& dgm1,
                             const std::vector<PersistencePair>& dgm2,
                             int dim, int p) {
    std::vector<std::pair<double,double>> pts1, pts2;
    for (auto& pp : dgm1) if (pp.dim==dim && !pp.is_essential()) pts1.push_back({pp.birth,pp.death});
    for (auto& pp : dgm2) if (pp.dim==dim && !pp.is_essential()) pts2.push_back({pp.birth,pp.death});

    auto cost = [&](std::pair<double,double> a, std::pair<double,double> b) {
        return std::pow(std::max(std::abs(a.first-b.first), std::abs(a.second-b.second)), p);
    };
    auto diag_cost = [&](std::pair<double,double> a) {
        return std::pow((a.second - a.first) / 2.0, p);
    };

    double total = 0.0;
    std::vector<bool> used2(pts2.size(), false);
    for (auto& pt : pts1) {
        double best = diag_cost(pt);
        int best_j = -1;
        for (int j=0; j<(int)pts2.size(); ++j) {
            if (used2[j]) continue;
            double c = cost(pt, pts2[j]);
            if (c < best) { best = c; best_j = j; }
        }
        if (best_j >= 0) used2[best_j] = true;
        total += best;
    }
    for (int j=0; j<(int)pts2.size(); ++j)
        if (!used2[j]) total += diag_cost(pts2[j]);
    return std::pow(total, 1.0/p);
}

// ========================== Persistence Landscape ==========================

std::vector<std::vector<double>>
persistence_landscape(const std::vector<PersistencePair>& diagram,
                       int n_layers, int n_samples,
                       double t_min, double t_max) {
    if (n_layers <= 0 || n_samples <= 1) return {};

    // Only finite-death pairs contribute a tent function (see is_essential()),
    // matching how bottleneck_distance / wasserstein_distance already filter diagrams.
    std::vector<std::pair<double, double>> pairs;
    for (auto& p : diagram)
        if (!p.is_essential()) pairs.push_back({p.birth, p.death});

    // Auto-derive the sampling range from the diagram when the caller didn't supply
    // a valid one (t_min>=t_max, including the sentinel 0.0/0.0 default).
    if (t_min >= t_max) {
        if (pairs.empty()) {
            t_min = 0.0;
            t_max = 1.0;
        } else {
            double bmin = pairs[0].first, dmax = pairs[0].second;
            for (auto& pr : pairs) {
                bmin = std::min(bmin, pr.first);
                dmax = std::max(dmax, pr.second);
            }
            t_min = bmin;
            t_max = dmax;
            if (t_min >= t_max) t_max = t_min + 1.0;  // degenerate (e.g. b==d) guard
        }
    }

    std::vector<std::vector<double>> result(n_layers, std::vector<double>(n_samples, 0.0));
    double step = (t_max - t_min) / (n_samples - 1);

    std::vector<double> tent_values;
    tent_values.reserve(pairs.size());
    for (int i = 0; i < n_samples; ++i) {
        double t = t_min + step * i;
        tent_values.clear();
        for (auto& pr : pairs) {
            double v = std::min(t - pr.first, pr.second - t);
            if (v > 0.0) tent_values.push_back(v);
        }
        // Descending sort: k-th largest lands at index k-1 (or 0 if fewer values exist).
        std::sort(tent_values.begin(), tent_values.end(), std::greater<double>());
        for (int k = 0; k < n_layers; ++k)
            result[k][i] = (k < (int)tent_values.size()) ? tent_values[k] : 0.0;
    }
    return result;
}

// ========================== Utilities ==========================

std::vector<std::vector<double>>
pairwise_distances(const std::vector<std::vector<double>>& pts) {
    int n = static_cast<int>(pts.size());
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    for (int i=0; i<n; ++i)
        for (int j=i+1; j<n; ++j) {
            double d=0;
            for (size_t k=0; k<pts[i].size(); ++k) {
                double diff = pts[i][k] - pts[j][k]; d += diff*diff;
            }
            dist[i][j] = dist[j][i] = std::sqrt(d);
        }
    return dist;
}

std::vector<std::vector<int>>
betti_curve(const std::vector<std::vector<double>>& dist_matrix,
            const std::vector<double>& thresholds, int max_dim) {
    std::vector<std::vector<int>> curve;
    for (double r : thresholds) {
        auto sc = vietoris_rips(dist_matrix, r, max_dim);
        auto betti = sc.betti_numbers();
        while ((int)betti.size() <= max_dim) betti.push_back(0);
        curve.push_back(betti);
    }
    return curve;
}

} // namespace topo
} // namespace ms
