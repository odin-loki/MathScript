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
    // index_ mirrors simplices_ exactly; the lookup is behaviourally identical to the
    // linear scan it replaces (see the member's comment in topo.hpp).
    return index_.find(s) != index_.end();
}

void SimplicialComplex::add_point(int v) {
    Simplex s = {v};
    if (!has_simplex(s)) {
        simplices_.push_back(s);
        index_.insert(s);
    }
}

void SimplicialComplex::add_simplex(const Simplex& s_in) {
    Simplex s = s_in;
    std::sort(s.begin(), s.end());
    if (has_simplex(s)) return;
    simplices_.push_back(s);
    index_.insert(s);
    // Add all faces (subsets)
    int n = static_cast<int>(s.size());
    for (int mask = 0; mask < (1<<n); ++mask) {
        if (std::popcount(static_cast<unsigned>(mask)) == n) continue;  // full set already added
        Simplex face;
        for (int i = 0; i < n; ++i) if (mask & (1<<i)) face.push_back(s[i]);
        if (!face.empty() && !has_simplex(face)) {
            simplices_.push_back(face);
            index_.insert(face);
        }
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

// ---------------- Minimum enclosing ball from distances alone ----------------
//
// Derivation (everything below is coordinate-free; only squared pairwise distances
// D_ab = d(q_a,q_b)^2 are ever used).
//
// (I)  For a point c = Σ_a λ_a q_a of the affine hull of q_0..q_{s-1} (Σ_a λ_a = 1) and
//      ANY point x, expanding |c - x|^2 = |Σ_a λ_a (q_a - x)|^2 and substituting
//      (q_a - x)·(q_b - x) = (d(q_a,x)^2 + d(q_b,x)^2 - D_ab)/2 gives
//          |c - x|^2 = Σ_a λ_a d(q_a,x)^2 - Q,     Q = Σ_{a<b} λ_a λ_b D_ab = ½ λᵀDλ.
//
// (II) Imposing |c - q_b|^2 = R^2 for every b and using (I) with x = q_b turns the
//      circumsphere conditions into  Σ_a λ_a D_ab = R^2 + Q =: μ  for every b, which
//      together with Σ_a λ_a = 1 is exactly the bordered Cayley-Menger system
//          [ 0  1ᵀ ] [ v0 ]   [ 1 ]
//          [ 1   D ] [ λ  ] = [ 0 ]         (row 0: Σλ = 1; row b+1: v0 + Σ_a λ_a D_ab = 0)
//      so μ = -v0. Contracting the second block with λ gives λᵀDλ = μ, i.e. Q = μ/2,
//      hence  R^2 = μ - Q = μ/2 = -v0/2  and λ is the barycentric circumcentre.
//      (Equivalently, by Cramer's rule, R^2 = -det(D) / (2 det(CM)) — the classical
//      determinant form. The linear solve is one pass and better conditioned, so that
//      is what is implemented.)
//      Sanity: a pair at distance d gives R = d/2; a unit equilateral triangle gives
//      R = 1/√3; a unit regular tetrahedron gives R = √(3/8).
//
// (III) Substituting Q = R^2 back into (I): x lies in the circumball of T iff
//          Σ_{a∈T} λ_a d(q_a,x)^2 <= 2 R^2.
//
// (IV) A ball B(c,R) ⊇ S is THE minimum enclosing ball of S iff c ∈ conv(S ∩ ∂B).
//      So if a subset T ⊆ S has all λ_a >= 0 (c ∈ conv(T)) and its circumball contains
//      every point of S, then T ⊆ S ∩ ∂B and c ∈ conv(T) ⊆ conv(S ∩ ∂B), i.e. that
//      circumball IS the MEB. Every subset passing both tests therefore reports the
//      same radius, so a search by increasing |T| may stop at the first size that hits.
//      Carathéodory guarantees an affinely independent such T exists, and an affinely
//      independent T has a non-singular Cayley-Menger system — so skipping the singular
//      (affinely dependent) subsets never loses the answer.

namespace {

struct CircumSphere {
    double r2 = 0.0;             // raw squared circumradius (may be a numerical -0)
    std::vector<double> lambda;  // barycentric coordinates of the circumcentre
    bool ok = false;             // false <=> singular (affinely dependent) system
};

// Solve the bordered Cayley-Menger system (II) for the subset `t` by Gauss-Jordan with
// partial pivoting. Returns ok == false when the system is singular, which happens
// exactly when the points of `t` are affinely dependent (duplicates, collinear triples,
// coplanar quadruples, ...) and no unique circumsphere exists.
CircumSphere cayley_menger_circumsphere(const std::vector<std::vector<double>>& d,
                                        const std::vector<int>& t) {
    CircumSphere out;
    const std::size_t s = t.size();
    if (s == 0) return out;
    const std::size_t m = s + 1;

    // Augmented (m) x (m+1) system: last column is the right-hand side e_0.
    std::vector<std::vector<double>> a(m, std::vector<double>(m + 1, 0.0));
    double scale = 1.0;
    for (std::size_t j = 1; j < m; ++j) { a[0][j] = 1.0; a[j][0] = 1.0; }
    for (std::size_t i = 0; i < s; ++i) {
        const std::size_t ti = static_cast<std::size_t>(t[i]);
        for (std::size_t j = 0; j < s; ++j) {
            const double dij = d[ti][static_cast<std::size_t>(t[j])];
            const double sq = dij * dij;
            a[i + 1][j + 1] = sq;
            if (sq > scale) scale = sq;
        }
    }
    a[0][m] = 1.0;

    for (std::size_t col = 0; col < m; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < m; ++r)
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        // Scale-relative singularity test: the matrix entries are squared distances.
        if (std::abs(a[piv][col]) <= 1e-12 * scale) return out;
        std::swap(a[col], a[piv]);
        const double p = a[col][col];
        for (std::size_t k = col; k <= m; ++k) a[col][k] /= p;
        for (std::size_t r = 0; r < m; ++r) {
            if (r == col) continue;
            const double f = a[r][col];
            if (f == 0.0) continue;
            for (std::size_t k = col; k <= m; ++k) a[r][k] -= f * a[col][k];
        }
    }

    out.lambda.assign(s, 0.0);
    for (std::size_t i = 0; i < s; ++i) out.lambda[i] = a[i + 1][m];
    out.r2 = -0.5 * a[0][m];  // identity (II): R^2 = -v0/2
    out.ok = true;
    return out;
}

// |c - x|^2 from identity (I). `q` is Q = ½ λᵀDλ; for a circumcentre Q == R^2.
// Clamped at 0 so that round-off cannot produce a negative squared distance.
double bary_sq_dist_to(const std::vector<std::vector<double>>& d,
                       const std::vector<int>& t,
                       const std::vector<double>& lambda,
                       double q, int x) {
    double acc = 0.0;
    const std::size_t xs = static_cast<std::size_t>(x);
    for (std::size_t a = 0; a < t.size(); ++a) {
        const double dax = d[static_cast<std::size_t>(t[a])][xs];
        acc += lambda[a] * dax * dax;
    }
    const double v = acc - q;
    return v > 0.0 ? v : 0.0;
}

// Badoiu-Clarkson core-set iteration, run entirely in barycentric coordinates via
// identity (I) so it stays coordinate-free. Used only above kMaxMebExactPoints, where
// the exact 2^m support search would blow up. Every candidate value is the covering
// radius of a genuine enclosing ball, hence an upper bound on the true MEB radius (up
// to round-off), so this does not under-report and the Čech complex built from it stays
// a subcomplex of the true one. Badoiu-Clarkson converges to within (1 + 1/sqrt(T)) of
// the optimum after T steps; the running minimum over the iterates is taken because the
// iterates are not monotone and every one of them is individually a valid bound.
double meb_radius_approx(const std::vector<std::vector<double>>& d,
                         const std::vector<int>& idx) {
    const std::size_t m = idx.size();
    if (m == 0) return 0.0;
    std::vector<double> lambda(m, 0.0);
    lambda[0] = 1.0;
    double best = std::numeric_limits<double>::infinity();
    for (int it = 1; it <= kMebApproxIterations; ++it) {
        double q = 0.0;  // Q = Σ_{a<b} λ_a λ_b D_ab
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t b = a + 1; b < m; ++b) {
                const double dab = d[static_cast<std::size_t>(idx[a])]
                                    [static_cast<std::size_t>(idx[b])];
                q += lambda[a] * lambda[b] * dab * dab;
            }
        std::size_t far = 0;
        double far_d2 = -1.0;
        for (std::size_t a = 0; a < m; ++a) {
            const double v = bary_sq_dist_to(d, idx, lambda, q, idx[a]);
            if (v > far_d2) { far_d2 = v; far = a; }
        }
        // far_d2 is the squared covering radius of the current centre: a valid upper
        // bound on the MEB radius, so the running minimum is valid too.
        const double r = std::sqrt(far_d2 > 0.0 ? far_d2 : 0.0);
        if (r < best) best = r;
        const double w = 1.0 / static_cast<double>(it + 1);  // λ <- (1-w)λ + w·e_far
        for (std::size_t a = 0; a < m; ++a) lambda[a] *= (1.0 - w);
        lambda[far] += w;
    }
    return best == std::numeric_limits<double>::infinity() ? 0.0 : best;
}

} // namespace

double meb_radius_from_distances(const std::vector<std::vector<double>>& dist_matrix,
                                 const std::vector<int>& idx) {
    // Defensive validation (this module returns plain values, never Result).
    const std::size_t n = dist_matrix.size();
    for (int v : idx) {
        if (v < 0 || static_cast<std::size_t>(v) >= n) return 0.0;
        if (dist_matrix[static_cast<std::size_t>(v)].size() < n) return 0.0;
    }
    const std::size_t m = idx.size();
    if (m <= 1) return 0.0;

    // Sizes 2 and 3 delegate to the exact closed forms this module already used, so
    // every pre-existing threshold (including the triangle helper's absolute obtuseness
    // tolerance) reproduces bit for bit.
    const std::size_t i0 = static_cast<std::size_t>(idx[0]);
    const std::size_t i1 = static_cast<std::size_t>(idx[1]);
    if (m == 2) return dist_matrix[i0][i1] / 2.0;
    if (m == 3) {
        const std::size_t i2 = static_cast<std::size_t>(idx[2]);
        return min_enclosing_ball_radius_triangle(dist_matrix[i0][i1],
                                                  dist_matrix[i1][i2],
                                                  dist_matrix[i0][i2]);
    }
    if (m > static_cast<std::size_t>(kMaxMebExactPoints))
        return meb_radius_approx(dist_matrix, idx);

    // Exact support-set search: smallest T whose circumcentre lies in conv(T) and whose
    // circumball encloses every point of idx. By (IV) the first size that hits is the
    // MEB, and every hit at that size reports the same radius.
    const double tol = 1e-9;
    std::vector<int> t;
    t.reserve(m);
    for (std::size_t sz = 1; sz <= m; ++sz) {
        double best = std::numeric_limits<double>::infinity();
        const unsigned long long limit = 1ull << m;
        for (unsigned long long mask = 1; mask < limit; ++mask) {
            if (static_cast<std::size_t>(std::popcount(mask)) != sz) continue;
            t.clear();
            for (std::size_t a = 0; a < m; ++a)
                if (mask & (1ull << a)) t.push_back(idx[a]);
            const CircumSphere cs = cayley_menger_circumsphere(dist_matrix, t);
            if (!cs.ok) continue;          // affinely dependent: no unique circumsphere
            if (cs.r2 < -tol) continue;    // numerically meaningless solution
            const double r2 = cs.r2 > 0.0 ? cs.r2 : 0.0;
            bool inside = true;
            for (double l : cs.lambda) if (l < -tol) { inside = false; break; }
            if (!inside) continue;         // circumcentre outside conv(T)
            const double slack = tol * (1.0 + r2);
            for (int x : idx) {
                if (bary_sq_dist_to(dist_matrix, t, cs.lambda, r2, x) > r2 + slack) {
                    inside = false;        // identity (III): x outside the circumball
                    break;
                }
            }
            if (!inside) continue;
            const double r = std::sqrt(r2);
            if (r < best) best = r;
        }
        if (best != std::numeric_limits<double>::infinity()) return best;
    }

    // Unreachable for metric input (some support set always qualifies); kept so the
    // function stays total, returning a valid enclosing radius (at most 2x the MEB).
    double fallback = std::numeric_limits<double>::infinity();
    for (int i : idx) {
        double far = 0.0;
        for (int j : idx)
            far = std::max(far, dist_matrix[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(j)]);
        fallback = std::min(fallback, far);
    }
    return fallback == std::numeric_limits<double>::infinity() ? 0.0 : fallback;
}

SimplicialComplex cech_complex(const std::vector<std::vector<double>>& dist_matrix,
                                double epsilon, int max_dim) {
    int n = static_cast<int>(dist_matrix.size());
    int dim_cap = std::min(max_dim, kMaxCechDim);  // see @note on the enumeration caps
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
    std::vector<Simplex> prev;  // surviving simplices of the dimension just built
    if (dim_cap >= 2) {
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j)
                for (int k = j+1; k < n; ++k) {
                    double a = dist_matrix[i][j];
                    double b = dist_matrix[j][k];
                    double c = dist_matrix[i][k];
                    double r = min_enclosing_ball_radius_triangle(a, b, c);
                    if (r <= epsilon) {
                        sc.add_simplex({i, j, k});
                        prev.push_back({i, j, k});
                    }
                }
    }

    // dim >= 3: incremental (Apriori-style) expansion. The MEB radius is monotone under
    // taking supersets, so Čech(ε) is closed under faces and a (k+1)-subset can only
    // qualify when every one of its k-subsets already qualified. Extend each surviving
    // (k-1)-simplex by one strictly larger vertex index -- which both generates every
    // candidate exactly once and keeps it sorted -- keep the candidate only when all of
    // its facets survived, then test its true MEB radius.
    std::set<Simplex> prev_set(prev.begin(), prev.end());
    long long examined = 0;
    std::vector<Simplex> cur;
    Simplex cand;
    Simplex facet;
    for (int dim = 3; dim <= dim_cap && !prev.empty(); ++dim) {
        cur.clear();
        bool truncated = false;
        for (const Simplex& s : prev) {
            for (int v = s.back() + 1; v < n; ++v) {
                bool ok = true;
                for (std::size_t drop = 0; drop < s.size() && ok; ++drop) {
                    facet.clear();
                    for (std::size_t u = 0; u < s.size(); ++u)
                        if (u != drop) facet.push_back(s[u]);
                    facet.push_back(v);  // v > s.back(), so `facet` is already sorted
                    ok = prev_set.find(facet) != prev_set.end();
                }
                if (!ok) continue;  // s itself is the remaining facet, and s is in prev
                if (examined >= kMaxCechCandidates) { truncated = true; break; }
                ++examined;
                cand = s;
                cand.push_back(v);
                if (meb_radius_from_distances(dist_matrix, cand) <= epsilon)
                    cur.push_back(cand);
            }
            if (truncated) break;
        }
        // Commit only whole dimensions: a dimension abandoned mid-way is dropped so that
        // the returned complex is always COMPLETE through its own top dimension, keeping
        // simplex_counts()/euler_characteristic()/betti_numbers() meaningful.
        if (truncated) break;
        for (const Simplex& top : cur) sc.add_simplex(top);
        prev.swap(cur);
        prev_set.clear();
        prev_set.insert(prev.begin(), prev.end());
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

namespace {

using PersistencePoint = std::pair<double, double>;

void collect_finite_points(int dim,
                           const std::vector<PersistencePair>& dgm,
                           std::vector<PersistencePoint>& out) {
    out.clear();
    out.reserve(dgm.size());
    for (const auto& pair : dgm) {
        if (pair.dim == dim && !pair.is_essential())
            out.emplace_back(pair.birth, pair.death);
    }
}

inline double l_inf_cost(PersistencePoint a, PersistencePoint b) {
    return std::max(std::abs(a.first - b.first), std::abs(a.second - b.second));
}

inline double diag_l_inf_cost(PersistencePoint a) {
    return (a.second - a.first) * 0.5;
}

inline double raise_to_p(double x, int p) {
    if (p == 1) return x;
    if (p == 2) return x * x;
    return std::pow(x, static_cast<double>(p));
}

inline double wasserstein_point_cost(PersistencePoint a, PersistencePoint b, int p) {
    return raise_to_p(l_inf_cost(a, b), p);
}

inline double wasserstein_diag_cost(PersistencePoint a, int p) {
    return raise_to_p(diag_l_inf_cost(a), p);
}

} // namespace

double bottleneck_distance(const std::vector<PersistencePair>& dgm1,
                            const std::vector<PersistencePair>& dgm2, int dim) {
    // Bottleneck distance: L-infinity optimal matching between off-diagonal points,
    // allowing unmatched points to be projected to the diagonal at cost (d-b)/2.
    // Exact bottleneck requires the Hungarian algorithm (O(n^3)); for typical
    // diagram sizes (n < 500) we use greedy O(n^2) matching, which is fast and
    // sufficient for approximate comparisons.
    std::vector<PersistencePoint> pts1, pts2;
    collect_finite_points(dim, dgm1, pts1);
    collect_finite_points(dim, dgm2, pts2);

    if (pts1.empty() && pts2.empty()) return 0.0;
    if (pts1.empty()) {
        double result = 0.0;
        for (const auto& pt : pts2)
            result = std::max(result, diag_l_inf_cost(pt));
        return result;
    }
    if (pts2.empty()) {
        double result = 0.0;
        for (const auto& pt : pts1)
            result = std::max(result, diag_l_inf_cost(pt));
        return result;
    }

    double result = 0.0;
    std::vector<bool> used2(pts2.size(), false);

    for (const auto& pt : pts1) {
        double best = diag_l_inf_cost(pt);
        int best_j = -1;
        const int n2 = static_cast<int>(pts2.size());
        for (int j = 0; j < n2; ++j) {
            if (used2[j]) continue;
            const double c = l_inf_cost(pt, pts2[j]);
            if (c < best) {
                best = c;
                best_j = j;
                if (best == 0.0) break;
            }
        }
        if (best_j >= 0) used2[best_j] = true;
        result = std::max(result, best);
    }
    for (int j = 0; j < static_cast<int>(pts2.size()); ++j) {
        if (!used2[j])
            result = std::max(result, diag_l_inf_cost(pts2[j]));
    }
    return result;
}

double wasserstein_distance(const std::vector<PersistencePair>& dgm1,
                             const std::vector<PersistencePair>& dgm2,
                             int dim, int p) {
    // Wasserstein-p distance via greedy matching (same complexity trade-off as
    // bottleneck_distance above). For p=2 the inner loop uses squared costs and
    // a single sqrt at the end.
    std::vector<PersistencePoint> pts1, pts2;
    collect_finite_points(dim, dgm1, pts1);
    collect_finite_points(dim, dgm2, pts2);

    if (pts1.empty() && pts2.empty()) return 0.0;

    const bool p_is_two = (p == 2);
    auto point_cost = [&](PersistencePoint a, PersistencePoint b) {
        return wasserstein_point_cost(a, b, p);
    };
    auto diag_cost = [&](PersistencePoint a) {
        return wasserstein_diag_cost(a, p);
    };

    auto accumulate_empty = [&](const std::vector<PersistencePoint>& pts) {
        double total = 0.0;
        for (const auto& pt : pts)
            total += diag_cost(pt);
        return p_is_two ? std::sqrt(total) : std::pow(total, 1.0 / p);
    };

    if (pts1.empty()) return accumulate_empty(pts2);
    if (pts2.empty()) return accumulate_empty(pts1);

    double total = 0.0;
    std::vector<bool> used2(pts2.size(), false);

    for (const auto& pt : pts1) {
        double best = diag_cost(pt);
        int best_j = -1;
        const int n2 = static_cast<int>(pts2.size());
        for (int j = 0; j < n2; ++j) {
            if (used2[j]) continue;
            const double c = point_cost(pt, pts2[j]);
            if (c < best) {
                best = c;
                best_j = j;
                if (best == 0.0) break;
            }
        }
        if (best_j >= 0) used2[best_j] = true;
        total += best;
    }
    for (int j = 0; j < static_cast<int>(pts2.size()); ++j) {
        if (!used2[j])
            total += diag_cost(pts2[j]);
    }
    return p_is_two ? std::sqrt(total) : std::pow(total, 1.0 / p);
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
