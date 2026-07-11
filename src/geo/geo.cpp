#define _USE_MATH_DEFINES
#include "ms/geo/geo.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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
                Vec3D nrm = cross(vec3(points[i], points[j]), vec3(points[i], points[k]));
                double nlen = length(nrm);
                if (nlen < 1e-12) continue;  // collinear triple, no plane

                // Tolerance scaled to the plane normal's own magnitude rather than
                // an absolute constant, since signed distances below are computed
                // with the un-normalised normal and thus scale with |n| and the
                // point cloud's coordinate range.
                double eps = 1e-9 * nlen;

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
;


KDTree2D::KDTree2D(const std::vector<Point2D>& pts) : pts_(pts) {
    std::vector<int> idx(pts.size());
    for (int i=0; i<(int)pts.size(); ++i) idx[i]=i;
    root_ = build(idx, 0);
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

KDTree2D::Node* KDTree2D::build(std::vector<int>& idx, int depth) {
    if (idx.empty()) return nullptr;
    int axis = depth % 2;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return axis==0 ? pts_[a].x < pts_[b].x : pts_[a].y < pts_[b].y;
    });
    int mid = (int)idx.size() / 2;
    auto n = new Node(idx[mid]);
    std::vector<int> left_idx(idx.begin(), idx.begin()+mid);
    std::vector<int> right_idx(idx.begin()+mid+1, idx.end());
    n->left  = build(left_idx, depth+1);
    n->right = build(right_idx, depth+1);
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
    for (auto& p : heap) result.push_back(p.second);
    return result;
}

std::vector<int> KDTree2D::range(Point2D q, double r) const {
    std::vector<int> result;
    double r2 = r * r;
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
;


KDTree3D::KDTree3D(const std::vector<Point3D>& pts) : pts_(pts) {
    std::vector<int> idx(pts.size());
    for (int i=0; i<(int)pts.size(); ++i) idx[i]=i;
    root_ = build(idx, 0);
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

KDTree3D::Node* KDTree3D::build(std::vector<int>& idx, int depth) {
    if (idx.empty()) return nullptr;
    int axis = depth % 3;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        auto& pa = pts_[a]; auto& pb = pts_[b];
        return axis==0 ? pa.x<pb.x : axis==1 ? pa.y<pb.y : pa.z<pb.z;
    });
    int mid = (int)idx.size()/2;
    auto n = new Node(idx[mid]);
    std::vector<int> l(idx.begin(), idx.begin()+mid);
    std::vector<int> r(idx.begin()+mid+1, idx.end());
    n->left = build(l, depth+1); n->right = build(r, depth+1);
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
    std::vector<int> result;
    for (auto& p : heap) result.push_back(p.second);
    return result;
}

std::vector<int> KDTree3D::range(Point3D q, double r) const {
    std::vector<int> result; double r2=r*r;
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
    int n = static_cast<int>(poly.size());
    int winding = 0;
    for (int i = 0; i < n; ++i) {
        Point2D a = poly[i], b = poly[(i+1)%n];
        if (a.y <= p.y) {
            if (b.y > p.y && cross2d(vec2(a,b), vec2(a,p)) > 0) ++winding;
        } else {
            if (b.y <= p.y && cross2d(vec2(a,b), vec2(a,p)) < 0) --winding;
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
    Vec2D ab = vec2(s.a, s.b), ap = vec2(s.a, p);
    double t = dot(ap, ab) / dot(ab, ab);
    t = std::max(0.0, std::min(1.0, t));
    Point2D proj = {s.a.x + t*ab.x, s.a.y + t*ab.y};
    return dist(p, proj);
}

double dist_point_line(Point2D p, Line2D l) {
    return std::abs(l.a*p.x + l.b*p.y + l.c) / std::sqrt(l.a*l.a + l.b*l.b);
}

double dist_point_plane(Point3D p, Plane3D pl) {
    return std::abs(dot(pl.normal, {p.x, p.y, p.z}) + pl.d) / length(pl.normal);
}

double dist_point_segment3(Point3D p, Segment3D s) {
    Vec3D ab = vec3(s.a, s.b), ap = vec3(s.a, p);
    double dab = dot(ab, ab);
    if (dab < 1e-15) return dist(p, s.a);
    double t = std::max(0.0, std::min(1.0, dot(ap, ab) / dab));
    Point3D proj = {s.a.x + t*ab.x, s.a.y + t*ab.y, s.a.z + t*ab.z};
    return dist(p, proj);
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

} // namespace geo
} // namespace ms
