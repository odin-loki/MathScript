#define _USE_MATH_DEFINES
#include "ms/geo/geo.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <queue>
#include <stack>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace geo {

// ---- Vec ops ----

Vec2D operator+(Vec2D a, Vec2D b) { return {a.x+b.x, a.y+b.y}; }
Vec2D operator-(Vec2D a, Vec2D b) { return {a.x-b.x, a.y-b.y}; }
Vec2D operator*(double s, Vec2D v) { return {s*v.x, s*v.y}; }
Vec2D vec2(Point2D a, Point2D b)   { return {b.x-a.x, b.y-a.y}; }
double dot(Vec2D a, Vec2D b)       { return a.x*b.x + a.y*b.y; }
double cross2d(Vec2D a, Vec2D b)   { return a.x*b.y - a.y*b.x; }
double length(Vec2D v)             { return std::sqrt(v.x*v.x + v.y*v.y); }
Vec2D normalise(Vec2D v) {
    double l = length(v);
    return l < 1e-15 ? v : Vec2D{v.x/l, v.y/l};
}

Vec3D operator+(Vec3D a, Vec3D b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
Vec3D operator-(Vec3D a, Vec3D b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3D operator*(double s, Vec3D v) { return {s*v.x, s*v.y, s*v.z}; }
Vec3D vec3(Point3D a, Point3D b)   { return {b.x-a.x, b.y-a.y, b.z-a.z}; }
double dot(Vec3D a, Vec3D b)       { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3D cross(Vec3D a, Vec3D b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
double length(Vec3D v)   { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
Vec3D normalise(Vec3D v) {
    double l = length(v);
    return l < 1e-15 ? v : Vec3D{v.x/l, v.y/l, v.z/l};
}

double dist(Point2D a, Point2D b) {
    double dx=a.x-b.x, dy=a.y-b.y;
    return std::sqrt(dx*dx+dy*dy);
}
double dist(Point3D a, Point3D b) {
    double dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}
double dist_sq(Point2D a, Point2D b) {
    double dx=a.x-b.x, dy=a.y-b.y;
    return dx*dx+dy*dy;
}

// ---- Convex Hull (Andrew's monotone chain) ----

static bool pt_cmp(const Point2D& a, const Point2D& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}
static double cross2d_pts(Point2D O, Point2D A, Point2D B) {
    return (A.x-O.x)*(B.y-O.y) - (A.y-O.y)*(B.x-O.x);
}

Polygon2D convex_hull_2d(std::vector<Point2D> pts) {
    int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::sort(pts.begin(), pts.end(), pt_cmp);
    Polygon2D hull;
    hull.reserve(static_cast<size_t>(n) + 1);
    // Lower hull
    for (int i = 0; i < n; ++i) {
        while (hull.size() >= 2 &&
               cross2d_pts(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    // Upper hull
    int lower_size = static_cast<int>(hull.size());
    for (int i = n-2; i >= 0; --i) {
        while ((int)hull.size() > lower_size &&
               cross2d_pts(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    hull.pop_back();
    return hull;
}

Polygon2D upper_hull(std::vector<Point2D> pts) {
    int n = static_cast<int>(pts.size());
    std::sort(pts.begin(), pts.end(), pt_cmp);
    Polygon2D hull;
    hull.reserve(static_cast<size_t>(n));
    for (int i = n-1; i >= 0; --i) {
        while (hull.size() >= 2 &&
               cross2d_pts(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    return hull;
}

Polygon2D lower_hull(std::vector<Point2D> pts) {
    int n = static_cast<int>(pts.size());
    std::sort(pts.begin(), pts.end(), pt_cmp);
    Polygon2D hull;
    hull.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        while (hull.size() >= 2 &&
               cross2d_pts(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    return hull;
}

// ---- Convex Hull 3D (brute-force face enumeration) ----

std::vector<Triangle3Di> convex_hull_3d(const std::vector<Point3D>& points) {
    int n = static_cast<int>(points.size());
    std::vector<Triangle3Di> hull;
    if (n < 4) return hull;

    // A hull face with more than 3 coplanar vertices (e.g. a cube's square
    // faces) is discovered by every 3-subset of that face's vertices, all
    // sharing the same supporting plane. Track vertex sets already turned
    // into triangles so each physical face is triangulated exactly once,
    // via a minimal fan, instead of once per combinatorial triple.
    std::vector<std::vector<int>> processed_faces;

    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            for (int k = j+1; k < n; ++k) {
                const Vec3D e1 = vec3(points[i], points[j]);
                const Vec3D e2 = vec3(points[i], points[k]);
                Vec3D nrm = cross(e1, e2);
                const double nlen_sq = dot(nrm, nrm);
                if (nlen_sq < 1e-24) continue;  // collinear triple, no plane
                const double nlen = std::sqrt(nlen_sq);

                // Tolerance scaled to the plane normal's own magnitude rather than
                // an absolute constant, since signed distances below are computed
                // with the un-normalised normal and thus scale with |n| and the
                // point cloud's coordinate range.
                const double eps = 1e-9 * nlen;

                std::vector<int> coplanar = {i, j, k};
                bool all_neg = true, all_pos = true;
                for (int p = 0; p < n; ++p) {
                    if (p == i || p == j || p == k) continue;
                    double d = dot(nrm, vec3(points[i], points[p]));
                    if (d > eps) all_neg = false;
                    else if (d < -eps) all_pos = false;
                    else coplanar.push_back(p);  // on the plane within tolerance
                }
                if (!all_neg && !all_pos) continue;  // points on both sides: not a hull face
                if (all_neg && all_pos) continue;     // nothing strictly off-plane: no volume bounded

                std::sort(coplanar.begin(), coplanar.end());
                bool seen = false;
                for (auto& f : processed_faces) if (f == coplanar) { seen = true; break; }
                if (seen) continue;
                processed_faces.push_back(coplanar);

                if (coplanar.size() == 3) {
                    if (all_neg) hull.push_back({i, j, k});
                    else         hull.push_back({i, k, j});  // outward normal is -n
                    continue;
                }

                // More than 3 coplanar points: some may be strictly interior
                // to the flat face polygon (e.g. lying at the centroid of
                // the others), not just extra polygon vertices, so a naive
                // angle-sort-and-fan over all of them can produce an invalid
                // (self-overlapping) triangulation. Project onto an in-plane
                // 2D basis and take the actual 2D convex hull (Andrew's
                // monotone chain, mirroring convex_hull_2d) to discard any
                // interior points, then fan-triangulate that polygon —
                // giving the minimal, non-overlapping triangle set, with
                // winding consistent with the outward normal.
                Vec3D outward = all_neg ? nrm : (-1.0) * nrm;
                Vec3D u = normalise(vec3(points[coplanar[0]], points[coplanar[1]]));
                Vec3D w = normalise(outward);
                Vec3D v = cross(w, u);

                Point3D c0 = points[coplanar[0]];
                struct ProjPt { double x, y; int idx; };
                std::vector<ProjPt> proj(coplanar.size());
                for (size_t t = 0; t < coplanar.size(); ++t) {
                    Vec3D d = vec3(c0, points[coplanar[t]]);
                    proj[t] = {dot(d, u), dot(d, v), coplanar[t]};
                }
                std::sort(proj.begin(), proj.end(), [](const ProjPt& a, const ProjPt& b) {
                    return a.x < b.x || (a.x == b.x && a.y < b.y);
                });
                auto cross2 = [](const ProjPt& O, const ProjPt& A, const ProjPt& B) {
                    return (A.x-O.x)*(B.y-O.y) - (A.y-O.y)*(B.x-O.x);
                };
                std::vector<ProjPt> poly;
                for (auto& pp : proj) {
                    while (poly.size() >= 2 &&
                           cross2(poly[poly.size()-2], poly.back(), pp) <= 0)
                        poly.pop_back();
                    poly.push_back(pp);
                }
                size_t lower_size = poly.size();
                for (auto it = proj.rbegin(); it != proj.rend(); ++it) {
                    while (poly.size() > lower_size &&
                           cross2(poly[poly.size()-2], poly.back(), *it) <= 0)
                        poly.pop_back();
                    poly.push_back(*it);
                }
                poly.pop_back();  // closing point duplicates poly[0]

                for (size_t t = 1; t + 1 < poly.size(); ++t)
                    hull.push_back({poly[0].idx, poly[t].idx, poly[t+1].idx});
            }
        }
    }
    return hull;
}

// ---- Delaunay (Bowyer-Watson incremental) ----

static double circumradius_sq(Point2D a, Point2D b, Point2D c) {
    double ax = a.x, ay = a.y, bx = b.x, by = b.y, cx = c.x, cy = c.y;
    double D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by));
    if (std::abs(D) < 1e-12) return std::numeric_limits<double>::infinity();
    double ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / D;
    double uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / D;
    double dx = ax - ux, dy = ay - uy;
    return dx*dx + dy*dy;
}

static Point2D circumcenter(Point2D a, Point2D b, Point2D c) {
    double ax=a.x, ay=a.y, bx=b.x, by=b.y, cx=c.x, cy=c.y;
    double D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by));
    if (std::abs(D) < 1e-12) return {(ax+bx+cx)/3, (ay+by+cy)/3};
    double ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / D;
    double uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / D;
    return {ux, uy};
}

static bool in_circumcircle(Point2D a, Point2D b, Point2D c, Point2D p) {
    double ax=a.x-p.x, ay=a.y-p.y;
    double bx=b.x-p.x, by=b.y-p.y;
    double cx=c.x-p.x, cy=c.y-p.y;
    double det = ax*(by*( ax*ax+ay*ay - cx*cx-cy*cy) - cy*(bx*bx+by*by - ax*ax-ay*ay))
               - bx*(ay*(bx*bx+by*by) - cy*(ax*ax+ay*ay));
    // More reliable:
    double mat = (ax*ax+ay*ay)*(bx*cy-by*cx)
               - (bx*bx+by*by)*(ax*cy-ay*cx)
               + (cx*cx+cy*cy)*(ax*by-ay*bx);
    return mat > 0.0;
}

std::vector<Triangle2Di> delaunay_2d(const std::vector<Point2D>& pts) {
    int n = static_cast<int>(pts.size());
    if (n < 3) return {};
    // Work with a local copy including super-triangle vertices
    std::vector<Point2D> points = pts;
    // Super-triangle bounding box
    double mn_x=pts[0].x, mn_y=pts[0].y, mx_x=mn_x, mx_y=mn_y;
    for (auto& p : pts) {
        mn_x=std::min(mn_x,p.x); mn_y=std::min(mn_y,p.y);
        mx_x=std::max(mx_x,p.x); mx_y=std::max(mx_y,p.y);
    }
    double dx=mx_x-mn_x, dy=mx_y-mn_y, delta=std::max(dx,dy);
    int si = n;
    points.push_back({mn_x - delta,     mn_y - delta});
    points.push_back({mn_x + 2*delta,   mn_y - delta});
    points.push_back({mn_x + delta/2,   mn_y + 2*delta});

    std::vector<Triangle2Di> triangles = {{si, si+1, si+2}};

    for (int pi = 0; pi < n; ++pi) {
        Point2D p = points[pi];
        std::vector<Triangle2Di> bad;
        for (auto& t : triangles)
            if (in_circumcircle(points[t.a], points[t.b], points[t.c], p))
                bad.push_back(t);

        // Find boundary of the polygonal hole
        std::vector<std::pair<int,int>> boundary;
        for (auto& t : bad) {
            std::pair<int,int> edges[3] = {{t.a,t.b},{t.b,t.c},{t.c,t.a}};
            for (auto& e : edges) {
                bool shared = false;
                for (auto& t2 : bad)
                    if (&t2 != &t)
                        if ((t2.a==e.first&&t2.b==e.second)||(t2.b==e.first&&t2.a==e.second)||
                            (t2.b==e.first&&t2.c==e.second)||(t2.c==e.first&&t2.b==e.second)||
                            (t2.a==e.first&&t2.c==e.second)||(t2.c==e.first&&t2.a==e.second))
                        { shared = true; break; }
                if (!shared) boundary.push_back(e);
            }
        }
        // Remove bad triangles
        triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
            [&](const Triangle2Di& t) {
                for (auto& b : bad) if (&b == &t) return true;
                // proper comparison
                return false;
            }), triangles.end());
        // Actually need to erase by value
        std::vector<Triangle2Di> good;
        for (auto& t : triangles) {
            bool isbad = false;
            for (auto& b : bad) if (b.a==t.a&&b.b==t.b&&b.c==t.c) { isbad=true; break; }
            if (!isbad) good.push_back(t);
        }
        triangles = good;
        // Add new triangles
        for (auto& e : boundary)
            triangles.push_back({e.first, e.second, pi});
    }
    // Remove triangles involving super-triangle vertices
    std::vector<Triangle2Di> result;
    for (auto& t : triangles)
        if (t.a < n && t.b < n && t.c < n)
            result.push_back(t);
    return result;
}

std::vector<Point2D> voronoi(const std::vector<Point2D>& pts) {
    auto tris = delaunay_2d(pts);
    std::vector<Point2D> centers;
    for (auto& t : tris)
        centers.push_back(circumcenter(pts[t.a], pts[t.b], pts[t.c]));
    return centers;
}

// ---- KD-Tree 2D ----

KDTree2D::KDTree2D(const std::vector<Point2D>& pts) : pts_(pts) {
    std::vector<int> idx(pts.size());
    for (int i=0; i<(int)pts.size(); ++i) idx[i]=i;
    root_ = build(idx, 0, static_cast<int>(idx.size()), 0);
}

KDTree2D::~KDTree2D() {
    std::stack<Node*> stk;
    if (root_) stk.push(root_);
    while (!stk.empty()) {
        auto n = stk.top(); stk.pop();
        if (n->left) stk.push(n->left);
        if (n->right) stk.push(n->right);
        delete n;
    }
}

KDTree2D::Node* KDTree2D::build(std::vector<int>& idx, int beg, int end, int depth) {
    if (beg >= end) return nullptr;
    const int axis = depth % 2;
    const int mid = beg + (end - beg) / 2;
    std::nth_element(idx.begin() + beg, idx.begin() + mid, idx.begin() + end,
        [&](int a, int b) {
            return axis == 0 ? pts_[a].x < pts_[b].x : pts_[a].y < pts_[b].y;
        });
    auto n = new Node(idx[mid]);
    n->left  = build(idx, beg, mid, depth + 1);
    n->right = build(idx, mid + 1, end, depth + 1);
    return n;
}

static void kd2_nearest(KDTree2D::Node* n, const std::vector<Point2D>& pts,
                        Point2D q, int depth, int& best, double& best_d2) {
    if (!n) return;
    double d2 = dist_sq(q, pts[n->idx]);
    if (d2 < best_d2) { best_d2 = d2; best = n->idx; }
    int axis = depth % 2;
    double diff = axis==0 ? q.x - pts[n->idx].x : q.y - pts[n->idx].y;
    auto near_child  = diff < 0 ? n->left  : n->right;
    auto far_child   = diff < 0 ? n->right : n->left;
    kd2_nearest(near_child, pts, q, depth+1, best, best_d2);
    if (diff*diff < best_d2)
        kd2_nearest(far_child, pts, q, depth+1, best, best_d2);
}

int KDTree2D::nearest(Point2D q) const {
    int best = -1; double best_d = std::numeric_limits<double>::infinity();
    kd2_nearest(root_, pts_, q, 0, best, best_d);
    return best;
}

std::vector<int> KDTree2D::knn(Point2D q, int k) const {
    // Priority queue: (dist, idx), max-heap
    std::vector<std::pair<double,int>> heap;
    std::function<void(Node*,int)> search = [&](Node* n, int depth) {
        if (!n) return;
        double d2 = dist_sq(q, pts_[n->idx]);
        if ((int)heap.size() < k || d2 < heap[0].first) {
            heap.push_back({d2, n->idx});
            std::push_heap(heap.begin(), heap.end());
            if ((int)heap.size() > k) { std::pop_heap(heap.begin(), heap.end()); heap.pop_back(); }
        }
        int axis = depth % 2;
        double diff = axis==0 ? q.x - pts_[n->idx].x : q.y - pts_[n->idx].y;
        auto near_c = diff < 0 ? n->left : n->right;
        auto far_c  = diff < 0 ? n->right : n->left;
        search(near_c, depth+1);
        if ((int)heap.size() < k || diff*diff < heap[0].first)
            search(far_c, depth+1);
    };
    search(root_, 0);
    std::vector<int> result;
    result.reserve(heap.size());
    for (auto& p : heap) result.push_back(p.second);
    return result;
}

std::vector<int> KDTree2D::range(Point2D q, double r) const {
    std::vector<int> result;
    result.reserve(16);
    const double r2 = r * r;
    std::function<void(Node*,int)> search = [&](Node* n, int depth) {
        if (!n) return;
        if (dist_sq(q, pts_[n->idx]) <= r2) result.push_back(n->idx);
        int axis = depth % 2;
        double diff = axis==0 ? q.x - pts_[n->idx].x : q.y - pts_[n->idx].y;
        if (diff < r) search(n->left, depth+1);
        if (diff > -r) search(n->right, depth+1);
    };
    search(root_, 0);
    return result;
}

// ---- KD-Tree 3D ----

KDTree3D::KDTree3D(const std::vector<Point3D>& pts) : pts_(pts) {
    std::vector<int> idx(pts.size());
    for (int i=0; i<(int)pts.size(); ++i) idx[i]=i;
    root_ = build(idx, 0, static_cast<int>(idx.size()), 0);
}

KDTree3D::~KDTree3D() {
    std::stack<Node*> stk;
    if (root_) stk.push(root_);
    while (!stk.empty()) {
        auto n = stk.top(); stk.pop();
        if (n->left) stk.push(n->left);
        if (n->right) stk.push(n->right);
        delete n;
    }
}

KDTree3D::Node* KDTree3D::build(std::vector<int>& idx, int beg, int end, int depth) {
    if (beg >= end) return nullptr;
    const int axis = depth % 3;
    const int mid = beg + (end - beg) / 2;
    std::nth_element(idx.begin() + beg, idx.begin() + mid, idx.begin() + end,
        [&](int a, int b) {
            const auto& pa = pts_[a];
            const auto& pb = pts_[b];
            return axis == 0 ? pa.x < pb.x : axis == 1 ? pa.y < pb.y : pa.z < pb.z;
        });
    auto n = new Node(idx[mid]);
    n->left = build(idx, beg, mid, depth + 1);
    n->right = build(idx, mid + 1, end, depth + 1);
    return n;
}

static double dist_sq3(Point3D a, Point3D b) {
    double dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return dx*dx+dy*dy+dz*dz;
}

static void kd3_nearest(KDTree3D::Node* n, const std::vector<Point3D>& pts,
                         Point3D q, int depth, int& best, double& best_d2) {
    if (!n) return;
    double d2 = dist_sq3(q, pts[n->idx]);
    if (d2 < best_d2) { best_d2 = d2; best = n->idx; }
    int axis = depth % 3;
    double diff = axis==0 ? q.x-pts[n->idx].x : axis==1 ? q.y-pts[n->idx].y : q.z-pts[n->idx].z;
    auto near_c = diff < 0 ? n->left : n->right;
    auto far_c  = diff < 0 ? n->right : n->left;
    kd3_nearest(near_c, pts, q, depth+1, best, best_d2);
    if (diff*diff < best_d2) kd3_nearest(far_c, pts, q, depth+1, best, best_d2);
}

int KDTree3D::nearest(Point3D q) const {
    int best=-1; double best_d=std::numeric_limits<double>::infinity();
    kd3_nearest(root_, pts_, q, 0, best, best_d);
    return best;
}

std::vector<int> KDTree3D::knn(Point3D q, int k) const {
    std::vector<std::pair<double,int>> heap;
    std::function<void(Node*,int)> search = [&](Node* n, int depth) {
        if (!n) return;
        double d2 = dist_sq3(q, pts_[n->idx]);
        if ((int)heap.size() < k || d2 < heap[0].first) {
            heap.push_back({d2, n->idx});
            std::push_heap(heap.begin(), heap.end());
            if ((int)heap.size() > k) { std::pop_heap(heap.begin(), heap.end()); heap.pop_back(); }
        }
        int axis = depth % 3;
        double diff = axis==0?q.x-pts_[n->idx].x:axis==1?q.y-pts_[n->idx].y:q.z-pts_[n->idx].z;
        auto nc = diff<0?n->left:n->right, fc = diff<0?n->right:n->left;
        search(nc, depth+1);
        if ((int)heap.size() < k || diff*diff < heap[0].first) search(fc, depth+1);
    };
    search(root_, 0);
    std::sort(heap.begin(), heap.end());
    std::vector<int> result;
    result.reserve(heap.size());
    for (auto& p : heap) result.push_back(p.second);
    return result;
}

std::vector<int> KDTree3D::range(Point3D q, double r) const {
    std::vector<int> result;
    result.reserve(16);
    const double r2 = r * r;
    std::function<void(Node*,int)> search = [&](Node* n, int depth) {
        if (!n) return;
        if (dist_sq3(q, pts_[n->idx]) <= r2) result.push_back(n->idx);
        int axis = depth%3;
        double diff = axis==0?q.x-pts_[n->idx].x:axis==1?q.y-pts_[n->idx].y:q.z-pts_[n->idx].z;
        if (diff<r) search(n->left, depth+1);
        if (diff>-r) search(n->right, depth+1);
    };
    search(root_, 0); return result;
}

// ---- Intersections ----

bool intersect_seg_seg(Segment2D s1, Segment2D s2, Point2D* out) {
    Vec2D d1 = vec2(s1.a, s1.b), d2 = vec2(s2.a, s2.b);
    Vec2D d3 = vec2(s1.a, s2.a);
    double denom = cross2d(d1, d2);
    if (std::abs(denom) < 1e-12) return false;
    double t = cross2d(d3, d2) / denom;
    double u = cross2d(d3, d1) / denom;
    if (t < 0 || t > 1 || u < 0 || u > 1) return false;
    if (out) *out = {s1.a.x + t*d1.x, s1.a.y + t*d1.y};
    return true;
}

bool intersect_ray_tri(Ray3D ray, Triangle3D tri, double* t_out) {
    // Möller-Trumbore
    Vec3D e1 = vec3(tri.a, tri.b), e2 = vec3(tri.a, tri.c);
    Vec3D h = cross(ray.dir, e2);
    double det = dot(e1, h);
    if (std::abs(det) < 1e-12) return false;
    double inv_det = 1.0 / det;
    Vec3D s = vec3(tri.a, ray.origin);
    double u = dot(s, h) * inv_det;
    if (u < 0 || u > 1) return false;
    Vec3D q = cross(s, e1);
    double v = dot(ray.dir, q) * inv_det;
    if (v < 0 || u+v > 1) return false;
    double t = dot(e2, q) * inv_det;
    if (t < 0) return false;
    if (t_out) *t_out = t;
    return true;
}

bool intersect_ray_sphere(Ray3D ray, Sphere3D sphere, double* t_out) {
    Vec3D oc = vec3(sphere.center, ray.origin);
    double b = dot(oc, ray.dir);
    double c = dot(oc, oc) - sphere.radius * sphere.radius;
    double disc = b*b - c;
    if (disc < 0) return false;
    double sq = std::sqrt(disc);
    double t = -b - sq;
    if (t < 0) t = -b + sq;
    if (t < 0) return false;
    if (t_out) *t_out = t;
    return true;
}

bool intersect_ray_aabb(Ray3D ray, AABB3D box, double* t_out) {
    double tmin = 0.0, tmax = std::numeric_limits<double>::infinity();
    double dirs[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
    double ori[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    double mn[3] = {box.min.x, box.min.y, box.min.z};
    double mx[3] = {box.max.x, box.max.y, box.max.z};
    for (int i = 0; i < 3; ++i) {
        if (std::abs(dirs[i]) < 1e-12) {
            if (ori[i] < mn[i] || ori[i] > mx[i]) return false;
        } else {
            double t1 = (mn[i] - ori[i]) / dirs[i];
            double t2 = (mx[i] - ori[i]) / dirs[i];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    if (t_out) *t_out = tmin;
    return true;
}

bool overlap_aabb_aabb(AABB3D a, AABB3D b) {
    return a.min.x<=b.max.x && a.max.x>=b.min.x &&
           a.min.y<=b.max.y && a.max.y>=b.min.y &&
           a.min.z<=b.max.z && a.max.z>=b.min.z;
}

bool overlap_circle_circle(Circle2D a, Circle2D b) {
    double r = a.radius + b.radius;
    return dist_sq(a.center, b.center) <= r*r;
}

bool point_in_polygon(Point2D p, const Polygon2D& poly) {
    const int n = static_cast<int>(poly.size());
    int winding = 0;
    for (int i = 0; i < n; ++i) {
        const Point2D& a = poly[i];
        const Point2D& b = poly[(i + 1) % n];
        const double abx = b.x - a.x;
        const double aby = b.y - a.y;
        const double apx = p.x - a.x;
        const double apy = p.y - a.y;
        const double cross = abx * apy - aby * apx;
        if (a.y <= p.y) {
            if (b.y > p.y && cross > 0.0) ++winding;
        } else if (b.y <= p.y && cross < 0.0) {
            --winding;
        }
    }
    return winding != 0;
}

bool point_in_aabb(Point2D p, AABB2D box) {
    return p.x >= box.min.x && p.x <= box.max.x &&
           p.y >= box.min.y && p.y <= box.max.y;
}

// ---- Distances ----

double dist_point_segment(Point2D p, Segment2D s) {
    const double abx = s.b.x - s.a.x;
    const double aby = s.b.y - s.a.y;
    const double apx = p.x - s.a.x;
    const double apy = p.y - s.a.y;
    const double dab = abx * abx + aby * aby;
    if (dab < 1e-15) return std::sqrt(apx * apx + apy * apy);
    double t = (apx * abx + apy * aby) / dab;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    const double px = apx - t * abx;
    const double py = apy - t * aby;
    return std::sqrt(px * px + py * py);
}

double dist_point_line(Point2D p, Line2D l) {
    return std::abs(l.a*p.x + l.b*p.y + l.c) / std::sqrt(l.a*l.a + l.b*l.b);
}

double dist_point_plane(Point3D p, Plane3D pl) {
    return std::abs(dot(pl.normal, {p.x, p.y, p.z}) + pl.d) / length(pl.normal);
}

double dist_point_segment3(Point3D p, Segment3D s) {
    const double abx = s.b.x - s.a.x;
    const double aby = s.b.y - s.a.y;
    const double abz = s.b.z - s.a.z;
    const double apx = p.x - s.a.x;
    const double apy = p.y - s.a.y;
    const double apz = p.z - s.a.z;
    const double dab = abx * abx + aby * aby + abz * abz;
    if (dab < 1e-15) return std::sqrt(apx * apx + apy * apy + apz * apz);
    double t = (apx * abx + apy * aby + apz * abz) / dab;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    const double px = apx - t * abx;
    const double py = apy - t * aby;
    const double pz = apz - t * abz;
    return std::sqrt(px * px + py * py + pz * pz);
}

// ---- Bezier curves ----

static Point2D lerp2(Point2D a, Point2D b, double t) {
    return {a.x + t*(b.x-a.x), a.y + t*(b.y-a.y)};
}

Point2D bezier_eval(const std::vector<Point2D>& ctrl, double t) {
    auto pts = ctrl;
    while (pts.size() > 1) {
        std::vector<Point2D> next(pts.size()-1);
        for (size_t i = 0; i < next.size(); ++i) next[i] = lerp2(pts[i], pts[i+1], t);
        pts = next;
    }
    return pts[0];
}

Vec2D bezier_deriv(const std::vector<Point2D>& ctrl, double t) {
    int n = static_cast<int>(ctrl.size()) - 1;
    std::vector<Point2D> diff(n);
    for (int i = 0; i < n; ++i)
        diff[i] = {(double)n*(ctrl[i+1].x - ctrl[i].x), (double)n*(ctrl[i+1].y - ctrl[i].y)};
    Point2D d = bezier_eval(diff, t);
    return {d.x, d.y};
}

std::pair<std::vector<Point2D>, std::vector<Point2D>>
bezier_subdivide(const std::vector<Point2D>& ctrl, double t) {
    std::vector<Point2D> left, right;
    auto pts = ctrl;
    left.push_back(pts.front());
    right.push_back(pts.back());
    while (pts.size() > 1) {
        std::vector<Point2D> next(pts.size()-1);
        for (size_t i = 0; i < next.size(); ++i) next[i] = lerp2(pts[i], pts[i+1], t);
        left.push_back(next.front());
        right.push_back(next.back());
        pts = next;
    }
    std::reverse(right.begin(), right.end());
    return {left, right};
}

Point2D bspline_eval(const std::vector<Point2D>& ctrl,
                     const std::vector<double>& knots, int degree, double t) {
    int n = static_cast<int>(ctrl.size()) - 1;
    // de Boor's algorithm
    int k = degree;
    // Find knot span
    int span = k;
    for (int i = k; i <= n; ++i)
        if (t < knots[i+1]) { span = i; break; }
    // Init with control points in the window
    std::vector<Point2D> d(k+1);
    for (int j = 0; j <= k; ++j) {
        int idx = span - k + j;
        if (idx < 0 || idx > n) d[j] = {0,0};
        else d[j] = ctrl[idx];
    }
    for (int r = 1; r <= k; ++r) {
        for (int j = k; j >= r; --j) {
            int i = span - k + j;
            double denom = knots[i+k-r+1] - knots[i];
            double alpha = (std::abs(denom) < 1e-12) ? 0.0 : (t - knots[i]) / denom;
            d[j] = lerp2(d[j-1], d[j], alpha);
        }
    }
    return d[k];
}

Point2D catmull_rom(const std::vector<Point2D>& ctrl, double t) {
    // Parameterised over [0, n-1]
    int n = static_cast<int>(ctrl.size());
    if (n < 2) return ctrl.empty() ? Point2D{0,0} : ctrl[0];
    double tt = t * (n - 1);
    int i = static_cast<int>(tt);
    i = std::max(0, std::min(i, n-2));
    double u = tt - i;
    // Catmull-Rom: need p_{i-1}, p_i, p_{i+1}, p_{i+2}
    int i0 = std::max(0, i-1), i1 = i, i2 = std::min(n-1, i+1), i3 = std::min(n-1, i+2);
    auto& p0=ctrl[i0]; auto& p1=ctrl[i1]; auto& p2=ctrl[i2]; auto& p3=ctrl[i3];
    double u2=u*u, u3=u2*u;
    double c0 = -u3+2*u2-u, c1 = 3*u3-5*u2+2, c2 = -3*u3+4*u2+u, c3 = u3-u2;
    return {0.5*(c0*p0.x + c1*p1.x + c2*p2.x + c3*p3.x),
            0.5*(c0*p0.y + c1*p1.y + c2*p2.y + c3*p3.y)};
}

Point2D hermite_curve(Point2D p0, Vec2D m0, Point2D p1, Vec2D m1, double t) {
    double t2=t*t, t3=t2*t;
    double h00=2*t3-3*t2+1, h10=t3-2*t2+t, h01=-2*t3+3*t2, h11=t3-t2;
    return {h00*p0.x + h10*m0.x + h01*p1.x + h11*m1.x,
            h00*p0.y + h10*m0.y + h01*p1.y + h11*m1.y};
}

// ---- Measurements ----

double signed_area(const Polygon2D& poly) {
    int n = static_cast<int>(poly.size());
    double area = 0.0;
    for (int i = 0; i < n; ++i) {
        auto& a = poly[i]; auto& b = poly[(i+1)%n];
        area += a.x * b.y - b.x * a.y;
    }
    return area * 0.5;
}

double area(const Polygon2D& poly) { return std::abs(signed_area(poly)); }

double perimeter(const Polygon2D& poly) {
    int n = static_cast<int>(poly.size());
    double p = 0.0;
    for (int i = 0; i < n; ++i) p += dist(poly[i], poly[(i+1)%n]);
    return p;
}

Point2D centroid(const Polygon2D& poly) {
    int n = static_cast<int>(poly.size());
    double cx=0, cy=0, area=0;
    for (int i = 0; i < n; ++i) {
        auto& a=poly[i]; auto& b=poly[(i+1)%n];
        double cross = a.x*b.y - b.x*a.y;
        cx += (a.x + b.x) * cross;
        cy += (a.y + b.y) * cross;
        area += cross;
    }
    area *= 0.5;
    return {cx / (6*area), cy / (6*area)};
}

double moment_of_inertia(const Polygon2D& poly) {
    auto c = centroid(poly);
    int n = static_cast<int>(poly.size());
    double I = 0.0;
    for (int i = 0; i < n; ++i) {
        auto a = poly[i]; auto b = poly[(i+1)%n];
        a.x -= c.x; a.y -= c.y; b.x -= c.x; b.y -= c.y;
        double cross = a.x*b.y - b.x*a.y;
        I += cross * (a.x*a.x + a.x*b.x + b.x*b.x + a.y*a.y + a.y*b.y + b.y*b.y);
    }
    return std::abs(I) / 12.0;
}

double area(Point2D a, Point2D b, Point2D c) {
    return 0.5 * std::abs(cross2d(vec2(a,b), vec2(a,c)));
}

double area(Triangle3D t) {
    Vec3D e1 = vec3(t.a, t.b), e2 = vec3(t.a, t.c);
    return 0.5 * length(cross(e1, e2));
}

double volume_tetrahedron(Point3D a, Point3D b, Point3D c, Point3D d) {
    Vec3D ab=vec3(a,b), ac=vec3(a,c), ad=vec3(a,d);
    return std::abs(dot(ab, cross(ac, ad))) / 6.0;
}

// ---- Bounding Rectangles ----

std::array<Point2D, 4> MinBoundingRect::corners() const {
    double c = std::cos(angle), s = std::sin(angle);
    double hw = width * 0.5, hh = height * 0.5;
    std::array<Point2D, 4> local = {{ {-hw,-hh}, {hw,-hh}, {hw,hh}, {-hw,hh} }};
    std::array<Point2D, 4> out;
    for (int i = 0; i < 4; ++i) {
        out[i] = { center.x + c*local[i].x - s*local[i].y,
                   center.y + s*local[i].x + c*local[i].y };
    }
    return out;
}

// Axis-aligned bounding box of the raw points, used whenever the input is too degenerate
// (fewer than 3 points, or fully collinear) for convex_hull_2d to yield a proper hull.
static MinBoundingRect aabb_fallback(const std::vector<Point2D>& points) {
    MinBoundingRect r;
    r.center = {0.0, 0.0};
    if (points.empty()) return r;
    double minx = points[0].x, maxx = points[0].x;
    double miny = points[0].y, maxy = points[0].y;
    for (const auto& p : points) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    r.center = {(minx+maxx)*0.5, (miny+maxy)*0.5};
    r.width  = maxx - minx;
    r.height = maxy - miny;
    r.angle  = 0.0;
    return r;
}

MinBoundingRect min_bounding_rect(const std::vector<Point2D>& points) {
    Polygon2D hull = convex_hull_2d(points);
    if (hull.size() < 3) return aabb_fallback(points);

    const int h = static_cast<int>(hull.size());
    MinBoundingRect best;
    double best_area = std::numeric_limits<double>::infinity();

    for (int i = 0; i < h; ++i) {
        const Point2D& a = hull[i];
        const Point2D& b = hull[(i+1) % h];
        double theta = std::atan2(b.y - a.y, b.x - a.x);
        double c = std::cos(theta), s = std::sin(theta);

        double minx = std::numeric_limits<double>::infinity();
        double maxx = -std::numeric_limits<double>::infinity();
        double miny = std::numeric_limits<double>::infinity();
        double maxy = -std::numeric_limits<double>::infinity();
        for (const auto& p : hull) {
            // Rotate by -theta so the current hull edge becomes axis-aligned.
            double rx =  c*p.x + s*p.y;
            double ry = -s*p.x + c*p.y;
            minx = std::min(minx, rx); maxx = std::max(maxx, rx);
            miny = std::min(miny, ry); maxy = std::max(maxy, ry);
        }
        double w = maxx - minx, ht = maxy - miny;
        double ar = w * ht;
        if (ar < best_area) {
            best_area = ar;
            double cx_rot = (minx+maxx)*0.5, cy_rot = (miny+maxy)*0.5;
            // Rotate the rotated-frame center back by +theta into the original frame.
            best.center = { c*cx_rot - s*cy_rot, s*cx_rot + c*cy_rot };
            best.width  = w;
            best.height = ht;
            best.angle  = theta;
        }
    }
    return best;
}

// ---- Minkowski Sum ----

// Index of the "bottom-most" vertex: lowest y, tie-broken by lowest x. Used as the
// canonical start of the angular edge walk (see minkowski_sum_convex).
static int bottom_most_index(const Polygon2D& poly) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(poly.size()); ++i) {
        if (poly[i].y < poly[best].y ||
            (poly[i].y == poly[best].y && poly[i].x < poly[best].x))
            best = i;
    }
    return best;
}

// Brute-force Minkowski sum: every pairwise vertex sum, then convex_hull_2d over the
// candidates. Correct for any convex a/b (not just the >=3-vertex case the merge-by-angle
// walk requires), so it doubles as the fallback for degenerate/tiny inputs.
static Polygon2D minkowski_sum_bruteforce(const Polygon2D& a, const Polygon2D& b) {
    std::vector<Point2D> sums;
    sums.reserve(a.size() * b.size());
    for (const auto& pa : a)
        for (const auto& pb : b)
            sums.push_back({pa.x + pb.x, pa.y + pb.y});
    if (sums.size() < 3) return sums;  // convex_hull_2d itself no-ops below 3 points anyway
    return convex_hull_2d(sums);
}

Polygon2D minkowski_sum_convex(const Polygon2D& a, const Polygon2D& b) {
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    if (na == 0 || nb == 0) return {};
    if (na < 3 || nb < 3) return minkowski_sum_bruteforce(a, b);

    constexpr double kEps = 1e-9;
    const int ia0 = bottom_most_index(a);
    const int ib0 = bottom_most_index(b);

    Polygon2D result;
    result.reserve(static_cast<size_t>(na) + static_cast<size_t>(nb));

    int ia = ia0, ib = ib0;
    Point2D cur = { a[ia0].x + b[ib0].x, a[ia0].y + b[ib0].y };

    for (int i = 0, j = 0; i < na || j < nb; ) {
        result.push_back(cur);

        const bool have_a = i < na, have_b = j < nb;
        const Vec2D ea = have_a ? vec2(a[ia], a[(ia + 1) % na]) : Vec2D{0.0, 0.0};
        const Vec2D eb = have_b ? vec2(b[ib], b[(ib + 1) % nb]) : Vec2D{0.0, 0.0};
        const double cr = (have_a && have_b) ? cross2d(ea, eb) : 0.0;

        // Advance whichever edge has the smaller polar angle; a near-zero cross product means
        // the two edges are collinear, in which case both advance and their vectors combine
        // into a single merged output edge (avoiding a redundant near-duplicate vertex).
        const bool advance_a = have_a && (!have_b || cr >= -kEps);
        const bool advance_b = have_b && (!have_a || cr <= kEps);

        Vec2D step{0.0, 0.0};
        if (advance_a) { step = step + ea; ia = (ia + 1) % na; ++i; }
        if (advance_b) { step = step + eb; ib = (ib + 1) % nb; ++j; }
        cur = { cur.x + step.x, cur.y + step.y };
    }
    return result;
}

// ---- Polygon Triangulation (ear clipping) ----

static bool pt_eq(const Point2D& a, const Point2D& b, double eps = 1e-12) {
    return std::abs(a.x - b.x) < eps && std::abs(a.y - b.y) < eps;
}

static bool pt_on_seg(const Point2D& p, const Point2D& a, const Point2D& b, double eps = 1e-9) {
    if (std::abs(cross2d_pts(a, b, p)) > eps) return false;
    double minx = std::min(a.x, b.x), maxx = std::max(a.x, b.x);
    double miny = std::min(a.y, b.y), maxy = std::max(a.y, b.y);
    return p.x >= minx - eps && p.x <= maxx + eps &&
           p.y >= miny - eps && p.y <= maxy + eps;
}

static bool pt_in_triangle_strict(const Point2D& p,
                                  const Point2D& a, const Point2D& b, const Point2D& c,
                                  bool ccw, double eps = 1e-9) {
    double s1 = cross2d_pts(a, b, p);
    double s2 = cross2d_pts(b, c, p);
    double s3 = cross2d_pts(c, a, p);
    if (ccw) return s1 > eps && s2 > eps && s3 > eps;
    return s1 < -eps && s2 < -eps && s3 < -eps;
}

static bool pt_in_ear(const Point2D& p,
                      const Point2D& a, const Point2D& b, const Point2D& c,
                      bool ccw) {
    if (pt_eq(p, a) || pt_eq(p, b) || pt_eq(p, c)) return false;
    if (pt_on_seg(p, a, b) || pt_on_seg(p, b, c) || pt_on_seg(p, c, a)) return false;
    return pt_in_triangle_strict(p, a, b, c, ccw);
}

std::vector<Triangle2Di> triangulate_polygon(const Polygon2D& poly) {
    std::vector<Triangle2Di> result;
    const int n = static_cast<int>(poly.size());
    if (n < 3) return result;

    std::vector<int> ring;
    ring.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (ring.empty() || !pt_eq(poly[i], poly[ring.back()]))
            ring.push_back(i);
    }
    if (ring.size() >= 2 && pt_eq(poly[ring.front()], poly[ring.back()]))
        ring.pop_back();
    if (ring.size() < 3) return result;

    const bool ccw = signed_area(poly) >= 0.0;
    constexpr double kEps = 1e-9;

    auto is_convex = [&](int prev, int curr, int next) {
        double cr = cross2d_pts(poly[prev], poly[curr], poly[next]);
        return ccw ? cr > kEps : cr < -kEps;
    };

    auto is_collinear = [&](int prev, int curr, int next) {
        return std::abs(cross2d_pts(poly[prev], poly[curr], poly[next])) <= kEps;
    };

    auto is_ear = [&](int i) {
        const int m = static_cast<int>(ring.size());
        const int prev = ring[(i + m - 1) % m];
        const int curr = ring[i];
        const int next = ring[(i + 1) % m];
        if (!is_convex(prev, curr, next)) return false;
        for (int j = 0; j < m; ++j) {
            if (j == i || j == (i + m - 1) % m || j == (i + 1) % m) continue;
            if (pt_in_ear(poly[ring[j]], poly[prev], poly[curr], poly[next], ccw))
                return false;
        }
        return true;
    };

    int guard = 0;
    const int max_steps = static_cast<int>(ring.size()) * static_cast<int>(ring.size()) + 8;
    while (ring.size() > 3 && guard++ < max_steps) {
        bool clipped = false;
        for (int i = 0; i < static_cast<int>(ring.size()); ++i) {
            const int m = static_cast<int>(ring.size());
            const int prev = ring[(i + m - 1) % m];
            const int curr = ring[i];
            const int next = ring[(i + 1) % m];

            if (is_collinear(prev, curr, next)) {
                ring.erase(ring.begin() + i);
                clipped = true;
                break;
            }
            if (!is_ear(i)) continue;

            result.push_back({prev, curr, next});
            ring.erase(ring.begin() + i);
            clipped = true;
            break;
        }
        if (!clipped) break;
    }

    if (ring.size() == 3)
        result.push_back({ring[0], ring[1], ring[2]});
    else if (ring.size() > 3)
        return {};

    return result;
}

// ---- Polygon Clipping (Sutherland-Hodgman) ----

// True if `p` lies on the clip window's interior side of the directed edge (a -> b). `ccw`
// is the clip window's own winding (per signed_area, matching the ccw convention already used
// by triangulate_polygon above): for a CCW window the interior is to the left of each directed
// edge (cross2d_pts(a,b,p) >= 0), and for a CW window it is to the right, so the comparison is
// flipped. Points exactly on the edge count as inside (non-strict), matching the standard
// Sutherland-Hodgman convention of retaining boundary points rather than dropping them.
static bool clip_inside(const Point2D& a, const Point2D& b, const Point2D& p, bool ccw) {
    double cr = cross2d_pts(a, b, p);
    return ccw ? cr >= -1e-12 : cr <= 1e-12;
}

// Intersection of the infinite line through clip edge (a, b) with segment (s, e). Only called
// when s and e fall on opposite sides of that line (so a genuine crossing exists between
// them); falls back to `s` if the two lines come out (near-)parallel regardless, which should
// not occur given the caller's opposite-side precondition outside of degenerate input.
static Point2D clip_line_intersection(const Point2D& a, const Point2D& b,
                                      const Point2D& s, const Point2D& e) {
    double A1 = b.y - a.y, B1 = a.x - b.x, C1 = A1 * a.x + B1 * a.y;
    double A2 = e.y - s.y, B2 = s.x - e.x, C2 = A2 * s.x + B2 * s.y;
    double det = A1 * B2 - A2 * B1;
    if (std::abs(det) < 1e-15) return s;
    return { (B2 * C1 - B1 * C2) / det, (A1 * C2 - A2 * C1) / det };
}

Polygon2D clip_polygon(const Polygon2D& subject, const Polygon2D& clip_window) {
    if (subject.empty() || clip_window.size() < 3) return {};

    const bool ccw = signed_area(clip_window) >= 0.0;
    const int m = static_cast<int>(clip_window.size());

    Polygon2D output = subject;
    for (int i = 0; i < m && !output.empty(); ++i) {
        const Point2D a = clip_window[i];
        const Point2D b = clip_window[(i + 1) % m];

        Polygon2D input = std::move(output);
        output.clear();
        const int n = static_cast<int>(input.size());
        for (int j = 0; j < n; ++j) {
            const Point2D curr = input[j];
            const Point2D next = input[(j + 1) % n];
            bool curr_in = clip_inside(a, b, curr, ccw);
            bool next_in = clip_inside(a, b, next, ccw);

            if (curr_in) output.push_back(curr);
            if (curr_in != next_in)
                output.push_back(clip_line_intersection(a, b, curr, next));
        }
    }
    return output;
}

// ---- Polygon Union (convex MVP) ----

static bool point_strictly_inside(const Point2D& p, const Polygon2D& poly) {
    if (!point_in_polygon(p, poly)) return false;
    const int n = static_cast<int>(poly.size());
    constexpr double kEps = 1e-9;
    for (int i = 0; i < n; ++i)
        if (pt_on_seg(p, poly[i], poly[(i + 1) % n], kEps)) return false;
    return true;
}

static void collect_segment_intersections(std::vector<Point2D>& out,
                                          const Point2D& a, const Point2D& b,
                                          const Point2D& c, const Point2D& d) {
    Vec2D r = vec2(a, b), s = vec2(c, d);
    double denom = cross2d(r, s);
    constexpr double kEps = 1e-9;
    if (std::abs(denom) < 1e-12) {
        if (pt_on_seg(a, c, d, kEps)) out.push_back(a);
        if (pt_on_seg(b, c, d, kEps)) out.push_back(b);
        if (pt_on_seg(c, a, b, kEps)) out.push_back(c);
        if (pt_on_seg(d, a, b, kEps)) out.push_back(d);
        return;
    }
    // a + t*r = c + u*s  =>  t = ((c - a) x s) / (r x s), u = ((c - a) x r) / (r x s).
    Vec2D ac = vec2(a, c);
    double t = cross2d(ac, s) / denom;
    double u = cross2d(ac, r) / denom;
    if (t >= -kEps && t <= 1.0 + kEps && u >= -kEps && u <= 1.0 + kEps)
        out.push_back({a.x + t * r.x, a.y + t * r.y});
}

static Polygon2D poly_union_hull_fallback(const Polygon2D& a, const Polygon2D& b) {
    std::vector<Point2D> pts;
    pts.reserve(a.size() + b.size());
    pts.insert(pts.end(), a.begin(), a.end());
    pts.insert(pts.end(), b.begin(), b.end());
    return convex_hull_2d(std::move(pts));
}

static Polygon2D poly_union_from_candidates(Polygon2D a, Polygon2D b) {
    std::vector<Point2D> candidates;
    candidates.reserve(a.size() + b.size() + 8);

    for (const auto& p : a)
        if (!point_strictly_inside(p, b)) candidates.push_back(p);
    for (const auto& p : b)
        if (!point_strictly_inside(p, a)) candidates.push_back(p);

    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    for (int i = 0; i < na; ++i) {
        const Point2D& s1 = a[i];
        const Point2D& e1 = a[(i + 1) % na];
        for (int j = 0; j < nb; ++j)
            collect_segment_intersections(candidates, s1, e1, b[j], b[(j + 1) % nb]);
    }

    if (candidates.size() < 3) return poly_union_hull_fallback(a, b);
    return convex_hull_2d(std::move(candidates));
}

static bool polygons_overlap(const Polygon2D& a, const Polygon2D& b) {
    for (const auto& p : a)
        if (point_in_polygon(p, b)) return true;
    for (const auto& p : b)
        if (point_in_polygon(p, a)) return true;

    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    for (int i = 0; i < na; ++i) {
        const Point2D& s1 = a[i];
        const Point2D& e1 = a[(i + 1) % na];
        for (int j = 0; j < nb; ++j) {
            Point2D ip;
            if (intersect_seg_seg({s1, e1}, {b[j], b[(j + 1) % nb]}, &ip))
                return true;
        }
    }
    return false;
}

Polygon2D poly_union(const Polygon2D& a, const Polygon2D& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.size() < 3 || b.size() < 3) return poly_union_hull_fallback(a, b);

    auto all_not_outside = [](const Polygon2D& inner, const Polygon2D& outer) {
        for (const auto& p : inner)
            if (!point_in_polygon(p, outer)) return false;
        return true;
    };
    if (all_not_outside(a, b)) return b;
    if (all_not_outside(b, a)) return a;

    if (!polygons_overlap(a, b)) return poly_union_hull_fallback(a, b);

    return poly_union_from_candidates(a, b);
}

// ---- Polygon Intersection (convex MVP) ----

Polygon2D poly_intersect(const Polygon2D& a, const Polygon2D& b) {
    if (a.empty() || b.empty()) return {};
    if (a.size() < 3 || b.size() < 3) return {};
    return clip_polygon(a, b);
}

// ---- Polygon Difference (convex MVP) ----

// Clip `subject` against one directed half-plane through edge (a -> b). When `keep_left`
// is true, retain the left side (cross >= 0); otherwise retain the right side — matching
// the interior/exterior convention of `clip_inside` for CCW/CW windows.
static Polygon2D clip_halfplane(const Polygon2D& subject, const Point2D& a, const Point2D& b,
                                bool keep_left) {
    if (subject.empty()) return {};
    Polygon2D output;
    const int n = static_cast<int>(subject.size());
    for (int j = 0; j < n; ++j) {
        const Point2D curr = subject[j];
        const Point2D next = subject[(j + 1) % n];
        const bool curr_in = clip_inside(a, b, curr, keep_left);
        const bool next_in = clip_inside(a, b, next, keep_left);
        if (curr_in) output.push_back(curr);
        if (curr_in != next_in)
            output.push_back(clip_line_intersection(a, b, curr, next));
    }
    return output;
}

Polygon2D poly_diff(const Polygon2D& a, const Polygon2D& b) {
    if (a.empty() || a.size() < 3) return {};
    if (b.empty() || b.size() < 3) return a;

    // Use intersection area for containment / empty-overlap: winding point-in-polygon is
    // unreliable on vertices, so vertex-subset tests mis-detect A ⊆ B (e.g. identical).
    const Polygon2D ix = poly_intersect(a, b);
    const double area_a = area(a);
    const double area_ix = area(ix);
    if (area_ix < 1e-12) return a;
    if (area_a - area_ix < 1e-9) return {};

    // A \ B = ∪_i (A ∩ exterior(edge_i of B)). When the difference is a single convex
    // piece, one of those half-plane clips has area ≈ area(A) − area(A∩B); pick that.
    // If no clip matches (non-convex difference / hole), fall back to a convex hull
    // over-approximation of candidate boundary points.
    const bool b_ccw = signed_area(b) >= 0.0;
    const bool exterior_keep_left = !b_ccw;
    const int m = static_cast<int>(b.size());
    const double target = area_a - area_ix;

    Polygon2D best;
    double best_err = 1e300;
    std::vector<Point2D> candidates;
    candidates.reserve(a.size() + b.size() + 8);

    for (int i = 0; i < m; ++i) {
        Polygon2D piece = clip_halfplane(a, b[i], b[(i + 1) % m], exterior_keep_left);
        if (piece.size() < 3) continue;
        const double ar = area(piece);
        if (ar < 1e-12) continue;
        const double err = std::abs(ar - target);
        if (err < best_err) {
            best_err = err;
            best = piece;
        }
        for (const auto& p : piece)
            if (!point_strictly_inside(p, b)) candidates.push_back(p);
    }

    if (best_err < 1e-6) return best;

    for (const auto& p : a)
        if (!point_strictly_inside(p, b)) candidates.push_back(p);
    for (const auto& p : b)
        if (point_strictly_inside(p, a) || point_in_polygon(p, a)) candidates.push_back(p);

    const int na = static_cast<int>(a.size());
    for (int i = 0; i < na; ++i) {
        const Point2D& s1 = a[i];
        const Point2D& e1 = a[(i + 1) % na];
        for (int j = 0; j < m; ++j)
            collect_segment_intersections(candidates, s1, e1, b[j], b[(j + 1) % m]);
    }

    if (candidates.size() < 3) return {};
    return convex_hull_2d(std::move(candidates));
}


// ---- General (non-convex) polygon booleans ----
//
// Edge-split / classify / trace. Both operands are oriented CCW so "interior on the left"
// holds; every edge is cut at its crossings with the other operand so that no sub-edge
// straddles the other boundary; each sub-edge is then kept, dropped or reversed by the rule
// for the requested operation; and the surviving directed edges are welded and traced into
// closed contours. Tracing takes the first edge clockwise from the reversed incoming
// direction at every branch, which is the planar-subdivision face rule and is what keeps a
// shell and a hole that touch at a single vertex as two separate contours.

namespace {

// How a sub-edge sits relative to the other operand's boundary.
enum class EdgeSide {
    Outside,        // midpoint strictly outside
    Inside,         // midpoint strictly inside
    SharedForward,  // midpoint on the boundary, both edges run the same way
    SharedReverse,  // midpoint on the boundary, the edges run opposite ways
};

struct BoolDirEdge {
    Point2D a, b;
};

double boolean_scale(const Polygon2D& a, const Polygon2D& b) {
    double lo_x = 0.0, hi_x = 0.0, lo_y = 0.0, hi_y = 0.0;
    bool seen = false;
    const auto sweep = [&](const Polygon2D& p) {
        for (const auto& q : p) {
            if (!seen) {
                lo_x = hi_x = q.x;
                lo_y = hi_y = q.y;
                seen = true;
                continue;
            }
            lo_x = std::min(lo_x, q.x);
            hi_x = std::max(hi_x, q.x);
            lo_y = std::min(lo_y, q.y);
            hi_y = std::max(hi_y, q.y);
        }
    };
    sweep(a);
    sweep(b);
    if (!seen) return 1.0;
    return std::max({1.0, hi_x - lo_x, hi_y - lo_y, std::abs(lo_x), std::abs(hi_x),
                     std::abs(lo_y), std::abs(hi_y)});
}

Polygon2D as_ccw(Polygon2D p) {
    if (signed_area(p) < 0.0) std::reverse(p.begin(), p.end());
    return p;
}

// Parameters along [p, q] at which the segment [r, s] touches it, appended to `ts`.
// Collinear overlaps contribute the projections of r and s, so a shared stretch of boundary
// is cut at both of its ends and classified as one sub-edge.
void edge_split_params(std::vector<double>& ts, const Point2D& p, const Point2D& q,
                       const Point2D& r, const Point2D& s, double eps) {
    const Vec2D d = vec2(p, q);
    const Vec2D e = vec2(r, s);
    const double d_len2 = d.x * d.x + d.y * d.y;
    if (d_len2 <= eps * eps) return;
    const double denom = cross2d(d, e);
    const double e_len2 = e.x * e.x + e.y * e.y;

    if (std::abs(denom) <= eps * std::sqrt(d_len2 * std::max(e_len2, eps * eps))) {
        // Parallel. Only a collinear overlap can contribute split points.
        if (std::abs(cross2d(d, vec2(p, r))) > eps * std::sqrt(d_len2)) return;
        for (const Point2D& x : {r, s}) {
            const double t = (vec2(p, x).x * d.x + vec2(p, x).y * d.y) / d_len2;
            if (t > 0.0 && t < 1.0) ts.push_back(t);
        }
        return;
    }

    // p + t*d = r + u*e  =>  t = ((r - p) x e) / (d x e), u = ((r - p) x d) / (d x e).
    const Vec2D pr = vec2(p, r);
    const double t = cross2d(pr, e) / denom;
    const double u = cross2d(pr, d) / denom;
    const double tol = eps / std::sqrt(d_len2);
    const double utol = (e_len2 > 0.0) ? eps / std::sqrt(e_len2) : 0.0;
    if (t > 0.0 && t < 1.0 && u >= -utol && u <= 1.0 + utol) ts.push_back(t);
}

// Split every edge of `subject` at its crossings with `other`, in order.
std::vector<BoolDirEdge> split_against(const Polygon2D& subject, const Polygon2D& other,
                                       double eps) {
    std::vector<BoolDirEdge> out;
    const std::size_t n = subject.size();
    const std::size_t m = other.size();
    out.reserve(n + m);
    std::vector<double> ts;
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D& p = subject[i];
        const Point2D& q = subject[(i + 1) % n];
        ts.clear();
        for (std::size_t j = 0; j < m; ++j) {
            edge_split_params(ts, p, q, other[j], other[(j + 1) % m], eps);
        }
        ts.push_back(0.0);
        ts.push_back(1.0);
        std::sort(ts.begin(), ts.end());
        const Vec2D d = vec2(p, q);
        const double len = length(d);
        if (len <= eps) continue;
        const double t_eps = eps / len;
        Point2D prev = p;
        double prev_t = 0.0;
        for (std::size_t k = 1; k < ts.size(); ++k) {
            const double t = std::min(1.0, std::max(0.0, ts[k]));
            if (t - prev_t <= t_eps) continue;
            const Point2D cur{p.x + t * d.x, p.y + t * d.y};
            out.push_back({prev, cur});
            prev = cur;
            prev_t = t;
        }
    }
    return out;
}

EdgeSide classify_edge(const BoolDirEdge& e, const Polygon2D& other, double eps) {
    const Point2D mid{0.5 * (e.a.x + e.b.x), 0.5 * (e.a.y + e.b.y)};
    const std::size_t m = other.size();
    const Vec2D d = vec2(e.a, e.b);
    for (std::size_t j = 0; j < m; ++j) {
        const Point2D& r = other[j];
        const Point2D& s = other[(j + 1) % m];
        if (!pt_on_seg(mid, r, s, eps)) continue;
        const Vec2D o = vec2(r, s);
        return (dot(d, o) > 0.0) ? EdgeSide::SharedForward : EdgeSide::SharedReverse;
    }
    return point_in_polygon(mid, other) ? EdgeSide::Inside : EdgeSide::Outside;
}

// Welds coincident endpoints onto shared vertex ids using a quantised grid plus a
// neighbourhood probe, so points that straddle a cell boundary still merge.
class VertexWelder {
  public:
    explicit VertexWelder(double cell) : cell_(cell > 0.0 ? cell : 1e-12) {}

    int intern(const Point2D& p) {
        const long long cx = static_cast<long long>(std::floor(p.x / cell_));
        const long long cy = static_cast<long long>(std::floor(p.y / cell_));
        for (long long dx = -1; dx <= 1; ++dx) {
            for (long long dy = -1; dy <= 1; ++dy) {
                const auto it = cells_.find({cx + dx, cy + dy});
                if (it == cells_.end()) continue;
                for (int id : it->second) {
                    if (std::abs(points_[static_cast<std::size_t>(id)].x - p.x) <= cell_ &&
                        std::abs(points_[static_cast<std::size_t>(id)].y - p.y) <= cell_) {
                        return id;
                    }
                }
            }
        }
        const int id = static_cast<int>(points_.size());
        points_.push_back(p);
        cells_[{cx, cy}].push_back(id);
        return id;
    }

    const std::vector<Point2D>& points() const { return points_; }

  private:
    double cell_;
    std::vector<Point2D> points_;
    std::map<std::pair<long long, long long>, std::vector<int>> cells_;
};

// Trace welded directed edges into closed contours.
PolygonSet trace_contours(const std::vector<Point2D>& pts, std::vector<std::pair<int, int>> edges,
                          double area_eps) {
    const std::size_t ne = edges.size();
    std::vector<char> used(ne, 0);
    std::map<int, std::vector<std::size_t>> outgoing;
    for (std::size_t i = 0; i < ne; ++i) {
        if (edges[i].first == edges[i].second) {
            used[i] = 1;  // zero-length after welding
            continue;
        }
        outgoing[edges[i].first].push_back(i);
    }

    PolygonSet contours;
    for (std::size_t seed = 0; seed < ne; ++seed) {
        if (used[seed]) continue;
        const int start = edges[seed].first;
        std::size_t cur = seed;
        Polygon2D contour;
        bool closed = false;
        for (std::size_t guard = 0; guard <= ne; ++guard) {
            used[cur] = 1;
            contour.push_back(pts[static_cast<std::size_t>(edges[cur].first)]);
            const int node = edges[cur].second;
            if (node == start) {
                closed = true;
                break;
            }
            const auto it = outgoing.find(node);
            if (it == outgoing.end()) break;

            // First edge clockwise from the reversed incoming direction. A delta of 0 is the
            // exact reversal, which we rank last by mapping it to a full turn.
            const Point2D& from = pts[static_cast<std::size_t>(edges[cur].first)];
            const Point2D& at = pts[static_cast<std::size_t>(node)];
            const double back = std::atan2(from.y - at.y, from.x - at.x);
            std::size_t best = ne;
            double best_delta = -1.0;
            for (std::size_t cand : it->second) {
                if (used[cand]) continue;
                const Point2D& to = pts[static_cast<std::size_t>(edges[cand].second)];
                const double ang = std::atan2(to.y - at.y, to.x - at.x);
                double delta = back - ang;
                while (delta <= 1e-12) delta += 2.0 * M_PI;
                while (delta > 2.0 * M_PI) delta -= 2.0 * M_PI;
                if (best == ne || delta < best_delta) {
                    best = cand;
                    best_delta = delta;
                }
            }
            if (best == ne) break;
            cur = best;
        }
        if (!closed || contour.size() < 3) continue;
        if (std::abs(signed_area(contour)) <= area_eps) continue;
        contours.push_back(std::move(contour));
    }
    return contours;
}

}  // namespace

PolygonSet poly_boolean(const Polygon2D& a_in, const Polygon2D& b_in, BooleanOp op) {
    const bool a_ok = a_in.size() >= 3 && std::abs(signed_area(a_in)) > 0.0;
    const bool b_ok = b_in.size() >= 3 && std::abs(signed_area(b_in)) > 0.0;

    // Degenerate operands have no interior, so the result follows from set algebra alone.
    if (!a_ok || !b_ok) {
        PolygonSet out;
        const auto emit = [&out](const Polygon2D& p, bool ok) {
            if (ok) out.push_back(as_ccw(p));
        };
        switch (op) {
            case BooleanOp::Union:
            case BooleanOp::SymmetricDifference:
                emit(a_in, a_ok);
                emit(b_in, b_ok);
                break;
            case BooleanOp::Intersection:
                break;
            case BooleanOp::Difference:
                emit(a_in, a_ok);
                break;
        }
        return out;
    }

    const Polygon2D a = as_ccw(a_in);
    const Polygon2D b = as_ccw(b_in);
    const double scale = boolean_scale(a, b);
    const double eps = 1e-9 * scale;

    const std::vector<BoolDirEdge> a_edges = split_against(a, b, eps);
    const std::vector<BoolDirEdge> b_edges = split_against(b, a, eps);

    std::vector<BoolDirEdge> kept;
    kept.reserve(a_edges.size() + b_edges.size());

    for (const auto& e : a_edges) {
        const EdgeSide side = classify_edge(e, b, eps);
        switch (op) {
            case BooleanOp::Union:
                if (side == EdgeSide::Outside || side == EdgeSide::SharedForward) kept.push_back(e);
                break;
            case BooleanOp::Intersection:
                if (side == EdgeSide::Inside || side == EdgeSide::SharedForward) kept.push_back(e);
                break;
            case BooleanOp::Difference:
                if (side == EdgeSide::Outside || side == EdgeSide::SharedReverse) kept.push_back(e);
                break;
            case BooleanOp::SymmetricDifference:
                if (side == EdgeSide::Outside) kept.push_back(e);
                else if (side == EdgeSide::Inside) kept.push_back({e.b, e.a});
                break;
        }
    }
    for (const auto& e : b_edges) {
        const EdgeSide side = classify_edge(e, a, eps);
        switch (op) {
            case BooleanOp::Union:
                if (side == EdgeSide::Outside) kept.push_back(e);
                break;
            case BooleanOp::Intersection:
                if (side == EdgeSide::Inside) kept.push_back(e);
                break;
            case BooleanOp::Difference:
                if (side == EdgeSide::Inside) kept.push_back({e.b, e.a});
                break;
            case BooleanOp::SymmetricDifference:
                if (side == EdgeSide::Outside) kept.push_back(e);
                else if (side == EdgeSide::Inside) kept.push_back({e.b, e.a});
                break;
        }
    }

    VertexWelder welder(eps);
    std::vector<std::pair<int, int>> welded;
    welded.reserve(kept.size());
    for (const auto& e : kept) {
        welded.emplace_back(welder.intern(e.a), welder.intern(e.b));
    }
    return trace_contours(welder.points(), std::move(welded), eps * scale);
}

PolygonSet poly_union_general(const Polygon2D& a, const Polygon2D& b) {
    return poly_boolean(a, b, BooleanOp::Union);
}

PolygonSet poly_intersect_general(const Polygon2D& a, const Polygon2D& b) {
    return poly_boolean(a, b, BooleanOp::Intersection);
}

PolygonSet poly_diff_general(const Polygon2D& a, const Polygon2D& b) {
    return poly_boolean(a, b, BooleanOp::Difference);
}

PolygonSet poly_symmetric_diff_general(const Polygon2D& a, const Polygon2D& b) {
    return poly_boolean(a, b, BooleanOp::SymmetricDifference);
}

double poly_set_area(const PolygonSet& set) {
    double total = 0.0;
    for (const auto& c : set) total += signed_area(c);
    return total;
}

bool point_in_polygon_set(Point2D p, const PolygonSet& set) {
    int winding = 0;
    for (const auto& poly : set) {
        const int n = static_cast<int>(poly.size());
        for (int i = 0; i < n; ++i) {
            const Point2D& u = poly[i];
            const Point2D& v = poly[(i + 1) % n];
            const double cr = (v.x - u.x) * (p.y - u.y) - (v.y - u.y) * (p.x - u.x);
            if (u.y <= p.y) {
                if (v.y > p.y && cr > 0.0) ++winding;
            } else if (v.y <= p.y && cr < 0.0) {
                --winding;
            }
        }
    }
    return winding != 0;
}

// ---- Marching cubes / marching squares ----

// Local corner offsets of the 8 cube vertices, in Lorensen-Cline order.
static constexpr int kCornerOffset[8][3] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

// For each of the 12 local edges: the offset of its lower-index grid endpoint plus the axis
// (0=x, 1=y, 2=z) it runs along. Canonicalising a local edge to a global grid edge this way
// makes neighbouring cells interpolate from the identical endpoint pair, so they produce
// bit-identical crossing points and share one vertex in the indexed mesh.
static constexpr int kEdgeBase[12][4] = {
    {0,0,0,0}, {1,0,0,1}, {0,1,0,0}, {0,0,0,1},
    {0,0,1,0}, {1,0,1,1}, {0,1,1,0}, {0,0,1,1},
    {0,0,0,2}, {1,0,0,2}, {1,1,0,2}, {0,1,0,2}
};

// 256-entry edge table: bit e of kEdgeTable[cube_index] is set when local edge e is crossed by
// the isosurface, i.e. when its two corners fall on opposite sides of the iso level.
static constexpr int kEdgeTable[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

// 256 x 16 triangle table (Lorensen-Cline). Each row lists local edge indices in groups of
// three, terminated by -1. Rows are emitted with their last two entries swapped (see
// marching_cubes_core) so the resulting normals point out of the sub-level set.
static constexpr int kTriTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1},
    { 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1},
    { 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1},
    { 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1},
    {10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1},
    { 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1},
    { 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1},
    { 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1},
    {11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1},
    { 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1},
    {11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1},
    {11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1},
    { 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1},
    { 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1},
    { 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1},
    { 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1},
    { 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1},
    { 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1},
    { 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1},
    { 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1},
    { 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1},
    { 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1},
    {10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1},
    {10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1},
    { 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1},
    { 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1},
    { 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1},
    { 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1},
    { 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1},
    {10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1},
    {10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1},
    { 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1},
    { 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1},
    {11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1},
    { 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1},
    { 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1},
    { 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1},
    { 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1},
    {10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1},
    { 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1},
    { 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1},
    { 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1},
    { 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1},
    {10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1},
    { 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1},
    { 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1},
    {10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1},
    {10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1},
    { 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1},
    { 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1},
    { 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1},
    { 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1},
    { 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1},
    { 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1},
    { 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1},
    { 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1},
    { 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1},
    { 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1},
    { 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1},
    { 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1},
    { 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1},
    { 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1},
    {11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1},
    { 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1},
    { 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1},
    { 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1},
    { 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1},
    {10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1},
    { 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1},
    { 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1},
    {11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1},
    { 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1},
    { 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1},
    { 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1},
    { 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1},
    { 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1},
    {10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1},
    { 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1},
    { 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1},
    { 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1},
    { 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1},
    { 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1},
    { 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1},
    { 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1},
    { 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1},
    { 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1},
    { 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1},
    { 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1},
    {11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1},
    { 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1},
    { 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1},
    { 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1},
    { 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1},
    { 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1},
    { 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1},
    { 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// 2D: corner offsets and the canonical base/axis of each of the 4 cell edges.
static constexpr int kCornerOffset2[4][2] = { {0,0}, {1,0}, {1,1}, {0,1} };
static constexpr int kEdgeBase2[4][3] = { {0,0,0}, {1,0,1}, {0,1,0}, {0,0,1} };

// The 16 marching-squares cases as directed edge pairs (from, to), -1 terminated. Directions
// put the sub-level set {f < iso} on the left of each segment. The two diagonally ambiguous
// cases (5 and 10) are resolved at run time by the asymptotic decider and are left empty here.
static constexpr int kSegTable[16][5] = {
    {-1, -1, -1, -1, -1},  //  0  none below
    { 0,  3, -1, -1, -1},  //  1  corner 0
    { 1,  0, -1, -1, -1},  //  2  corner 1
    { 1,  3, -1, -1, -1},  //  3  corners 0,1
    { 2,  1, -1, -1, -1},  //  4  corner 2
    {-1, -1, -1, -1, -1},  //  5  corners 0,2  (ambiguous)
    { 2,  0, -1, -1, -1},  //  6  corners 1,2
    { 2,  3, -1, -1, -1},  //  7  corners 0,1,2
    { 3,  2, -1, -1, -1},  //  8  corner 3
    { 0,  2, -1, -1, -1},  //  9  corners 0,3
    {-1, -1, -1, -1, -1},  // 10  corners 1,3  (ambiguous)
    { 1,  2, -1, -1, -1},  // 11  corners 0,1,3
    { 3,  1, -1, -1, -1},  // 12  corners 2,3
    { 0,  1, -1, -1, -1},  // 13  corners 0,2,3
    { 3,  0, -1, -1, -1},  // 14  corners 1,2,3
    {-1, -1, -1, -1, -1}   // 15  all below
};

// The two resolutions of each ambiguous case, as two directed edge pairs.
static constexpr int kSeg5Separated[4]  = {0, 3, 2, 1};
static constexpr int kSeg5Joined[4]     = {0, 1, 2, 3};
static constexpr int kSeg10Separated[4] = {1, 0, 3, 2};
static constexpr int kSeg10Joined[4]    = {3, 0, 1, 2};

// Linear crossing point along one cell edge. Only called for a genuinely cut edge, where one
// endpoint is strictly below the level and the other is not, so the denominator cannot be zero
// and t naturally lands in [0,1]; the guard and the clamp are belt and braces against a
// non-finite denominator surviving the caller's finiteness screen. The !(t >= 0.0) form also
// rejects NaN.
static Point3D interp_edge3(double iso, Point3D p0, Point3D p1, double v0, double v1) {
    const double denom = v1 - v0;
    double t = 0.5;
    if (std::abs(denom) > 0.0) t = (iso - v0) / denom;
    if (!(t >= 0.0)) t = 0.0;
    if (t > 1.0) t = 1.0;
    return { p0.x + t * (p1.x - p0.x),
             p0.y + t * (p1.y - p0.y),
             p0.z + t * (p1.z - p0.z) };
}

static Point2D interp_edge2(double iso, Point2D p0, Point2D p1, double v0, double v1) {
    const double denom = v1 - v0;
    double t = 0.5;
    if (std::abs(denom) > 0.0) t = (iso - v0) / denom;
    if (!(t >= 0.0)) t = 0.0;
    if (t > 1.0) t = 1.0;
    return { p0.x + t * (p1.x - p0.x), p0.y + t * (p1.y - p0.y) };
}

// Shared core: builds the de-duplicated vertex array and the index triples. Both public
// marching-cubes entry points go through this, so the soup and the indexed mesh are guaranteed
// to carry identical vertex positions in identical triangle order.
static TriMesh3D marching_cubes_core(const std::vector<double>& field,
                                     int nx, int ny, int nz, double iso,
                                     Point3D origin, Vec3D spacing) {
    TriMesh3D mesh;
    if (nx < 2 || ny < 2 || nz < 2) return mesh;
    const std::size_t total = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    if (field.size() != total) return mesh;

    const std::size_t snx = static_cast<std::size_t>(nx);
    const std::size_t sny = static_cast<std::size_t>(ny);

    auto sample = [&](int i, int j, int k) {
        return field[static_cast<std::size_t>(i) +
                     snx * (static_cast<std::size_t>(j) +
                            sny * static_cast<std::size_t>(k))];
    };
    auto position = [&](int i, int j, int k) {
        return Point3D{ origin.x + i * spacing.x,
                        origin.y + j * spacing.y,
                        origin.z + k * spacing.z };
    };

    // vertex_of_edge[3 * node_index + axis] is the index in mesh.vertices of the crossing point
    // on the grid edge leaving node (i,j,k) along `axis`, or -1 when it has not been cut yet.
    // This is what welds the shared edges of neighbouring cells onto one vertex.
    std::vector<int> vertex_of_edge(3 * total, -1);

    auto vertex_for = [&](int i, int j, int k, int axis) {
        const std::size_t key =
            3 * (static_cast<std::size_t>(i) +
                 snx * (static_cast<std::size_t>(j) +
                        sny * static_cast<std::size_t>(k))) +
            static_cast<std::size_t>(axis);
        if (vertex_of_edge[key] >= 0) return vertex_of_edge[key];
        const int di = (axis == 0) ? 1 : 0;
        const int dj = (axis == 1) ? 1 : 0;
        const int dk = (axis == 2) ? 1 : 0;
        const Point3D p = interp_edge3(iso, position(i, j, k),
                                       position(i + di, j + dj, k + dk),
                                       sample(i, j, k),
                                       sample(i + di, j + dj, k + dk));
        const int id = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back(p);
        vertex_of_edge[key] = id;
        return id;
    };

    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                double vals[8] = {};
                bool finite = true;
                for (int c = 0; c < 8; ++c) {
                    vals[c] = sample(i + kCornerOffset[c][0],
                                     j + kCornerOffset[c][1],
                                     k + kCornerOffset[c][2]);
                    if (!std::isfinite(vals[c])) finite = false;
                }
                if (!finite) continue;   // corrupted cell contributes nothing

                int cube_index = 0;
                for (int c = 0; c < 8; ++c)
                    if (vals[c] < iso) cube_index |= (1 << c);

                const int cut = kEdgeTable[cube_index];
                if (cut == 0) continue;  // wholly inside or wholly outside

                int edge_vertex[12] = {};
                for (int e = 0; e < 12; ++e) {
                    edge_vertex[e] = -1;
                    if ((cut & (1 << e)) == 0) continue;
                    edge_vertex[e] = vertex_for(i + kEdgeBase[e][0],
                                                j + kEdgeBase[e][1],
                                                k + kEdgeBase[e][2],
                                                kEdgeBase[e][3]);
                }
                // Last two entries swapped: the published table winds triangles with the normal
                // pointing into the sub-level set, and this module's convention is outward.
                for (int t = 0; kTriTable[cube_index][t] != -1; t += 3) {
                    mesh.triangles.push_back({ edge_vertex[kTriTable[cube_index][t]],
                                               edge_vertex[kTriTable[cube_index][t + 2]],
                                               edge_vertex[kTriTable[cube_index][t + 1]] });
                }
            }
        }
    }
    return mesh;
}

std::vector<Triangle3D> marching_cubes(const std::vector<double>& field,
                                       int nx, int ny, int nz, double iso,
                                       Point3D origin, Vec3D spacing) {
    const TriMesh3D mesh = marching_cubes_core(field, nx, ny, nz, iso, origin, spacing);
    std::vector<Triangle3D> soup;
    soup.reserve(mesh.triangles.size());
    for (const auto& t : mesh.triangles)
        soup.push_back({ mesh.vertices[static_cast<std::size_t>(t.a)],
                         mesh.vertices[static_cast<std::size_t>(t.b)],
                         mesh.vertices[static_cast<std::size_t>(t.c)] });
    return soup;
}

TriMesh3D marching_cubes_mesh(const std::vector<double>& field,
                              int nx, int ny, int nz, double iso,
                              Point3D origin, Vec3D spacing) {
    return marching_cubes_core(field, nx, ny, nz, iso, origin, spacing);
}

std::vector<Segment2D> marching_squares(const std::vector<double>& field,
                                        int nx, int ny, double iso,
                                        Point2D origin, Vec2D spacing) {
    std::vector<Segment2D> out;
    if (nx < 2 || ny < 2) return out;
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (field.size() != total) return out;

    const std::size_t snx = static_cast<std::size_t>(nx);
    auto sample = [&](int i, int j) {
        return field[static_cast<std::size_t>(i) + snx * static_cast<std::size_t>(j)];
    };
    auto position = [&](int i, int j) {
        return Point2D{ origin.x + i * spacing.x, origin.y + j * spacing.y };
    };

    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            double vals[4] = {};
            bool finite = true;
            for (int c = 0; c < 4; ++c) {
                vals[c] = sample(i + kCornerOffset2[c][0], j + kCornerOffset2[c][1]);
                if (!std::isfinite(vals[c])) finite = false;
            }
            if (!finite) continue;

            int square_index = 0;
            for (int c = 0; c < 4; ++c)
                if (vals[c] < iso) square_index |= (1 << c);
            if (square_index == 0 || square_index == 15) continue;

            // Crossing points are always interpolated from the lexicographically lower grid node
            // of the edge (kEdgeBase2), so the cell on the other side of a shared edge computes
            // the identical point and the contour has no gaps.
            Point2D edge_point[4] = {};
            for (int e = 0; e < 4; ++e) {
                const int a = e;
                const int b = (e + 1) % 4;
                if ((vals[a] < iso) == (vals[b] < iso)) continue;
                const int bi = i + kEdgeBase2[e][0];
                const int bj = j + kEdgeBase2[e][1];
                const int axis = kEdgeBase2[e][2];
                const int di = (axis == 0) ? 1 : 0;
                const int dj = (axis == 1) ? 1 : 0;
                edge_point[e] = interp_edge2(iso, position(bi, bj),
                                             position(bi + di, bj + dj),
                                             sample(bi, bj), sample(bi + di, bj + dj));
            }

            if (square_index == 5 || square_index == 10) {
                // Asymptotic decider: the saddle value of the cell's bilinear interpolant.
                const double denom = vals[0] - vals[1] + vals[2] - vals[3];
                double saddle = 0.25 * (vals[0] + vals[1] + vals[2] + vals[3]);
                if (std::abs(denom) > 0.0) {
                    const double s = (vals[0] * vals[2] - vals[1] * vals[3]) / denom;
                    if (std::isfinite(s)) saddle = s;
                }
                const bool joined = saddle < iso;
                const int* pairs = (square_index == 5)
                    ? (joined ? kSeg5Joined : kSeg5Separated)
                    : (joined ? kSeg10Joined : kSeg10Separated);
                for (int p = 0; p < 4; p += 2)
                    out.push_back({ edge_point[pairs[p]], edge_point[pairs[p + 1]] });
                continue;
            }

            for (int p = 0; kSegTable[square_index][p] != -1; p += 2)
                out.push_back({ edge_point[kSegTable[square_index][p]],
                                edge_point[kSegTable[square_index][p + 1]] });
        }
    }
    return out;
}

// ---- Mesh measurements ----

double mesh_surface_area(const std::vector<Triangle3D>& tris) {
    double total = 0.0;
    for (const auto& t : tris) total += area(t);
    return total;
}

double mesh_surface_area(const TriMesh3D& mesh) {
    double total = 0.0;
    const int nv = static_cast<int>(mesh.vertices.size());
    for (const auto& t : mesh.triangles) {
        if (t.a < 0 || t.b < 0 || t.c < 0 || t.a >= nv || t.b >= nv || t.c >= nv) continue;
        total += area(Triangle3D{ mesh.vertices[static_cast<std::size_t>(t.a)],
                                  mesh.vertices[static_cast<std::size_t>(t.b)],
                                  mesh.vertices[static_cast<std::size_t>(t.c)] });
    }
    return total;
}

double mesh_volume(const std::vector<Triangle3D>& tris) {
    double six_v = 0.0;
    for (const auto& t : tris) {
        const Vec3D bc = cross(Vec3D{t.b.x, t.b.y, t.b.z}, Vec3D{t.c.x, t.c.y, t.c.z});
        six_v += t.a.x * bc.x + t.a.y * bc.y + t.a.z * bc.z;
    }
    return six_v / 6.0;
}

double mesh_volume(const TriMesh3D& mesh) {
    double six_v = 0.0;
    const int nv = static_cast<int>(mesh.vertices.size());
    for (const auto& t : mesh.triangles) {
        if (t.a < 0 || t.b < 0 || t.c < 0 || t.a >= nv || t.b >= nv || t.c >= nv) continue;
        const Point3D& pa = mesh.vertices[static_cast<std::size_t>(t.a)];
        const Point3D& pb = mesh.vertices[static_cast<std::size_t>(t.b)];
        const Point3D& pc = mesh.vertices[static_cast<std::size_t>(t.c)];
        const Vec3D bc = cross(Vec3D{pb.x, pb.y, pb.z}, Vec3D{pc.x, pc.y, pc.z});
        six_v += pa.x * bc.x + pa.y * bc.y + pa.z * bc.z;
    }
    return six_v / 6.0;
}

} // namespace geo
} // namespace ms
