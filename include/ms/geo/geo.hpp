#pragma once
#include <array>
#include <functional>
#include <optional>
#include <vector>

namespace ms {
namespace geo {

// ========================== 2D Primitives ==========================

struct Point2D { double x, y; };
struct Vec2D   { double x, y; };

struct Segment2D { Point2D a, b; };
struct Ray2D     { Point2D origin; Vec2D dir; };
struct Line2D    { double a, b, c; };  // ax + by + c = 0

struct Circle2D  { Point2D center; double radius; };
struct AABB2D    { Point2D min, max; };

using Polygon2D = std::vector<Point2D>;

// ========================== 3D Primitives ==========================

struct Point3D { double x, y, z; };
struct Vec3D   { double x, y, z; };

struct Segment3D  { Point3D a, b; };
struct Ray3D      { Point3D origin; Vec3D dir; };
struct Plane3D    { Vec3D normal; double d; };  // n·p + d = 0
struct Triangle3D { Point3D a, b, c; };
struct Sphere3D   { Point3D center; double radius; };
struct AABB3D     { Point3D min, max; };

// ========================== Vec ops ==========================

Vec2D  operator+(Vec2D a, Vec2D b);
Vec2D  operator-(Vec2D a, Vec2D b);
Vec2D  operator*(double s, Vec2D v);
Vec2D  vec2(Point2D a, Point2D b);   // b - a
double dot(Vec2D a, Vec2D b);
double cross2d(Vec2D a, Vec2D b);    // scalar cross product
double length(Vec2D v);
Vec2D  normalise(Vec2D v);

Vec3D  operator+(Vec3D a, Vec3D b);
Vec3D  operator-(Vec3D a, Vec3D b);
Vec3D  operator*(double s, Vec3D v);
Vec3D  vec3(Point3D a, Point3D b);
double dot(Vec3D a, Vec3D b);
Vec3D  cross(Vec3D a, Vec3D b);
double length(Vec3D v);
Vec3D  normalise(Vec3D v);

double dist(Point2D a, Point2D b);
double dist(Point3D a, Point3D b);
double dist_sq(Point2D a, Point2D b);

// ========================== Convex Hull ==========================

// Graham scan (2D): returns CCW hull
Polygon2D convex_hull_2d(std::vector<Point2D> points);
// Andrew's monotone chain — upper hull
Polygon2D upper_hull(std::vector<Point2D> points);
Polygon2D lower_hull(std::vector<Point2D> points);

// ========================== Delaunay / Voronoi ==========================

struct Triangle2Di { int a, b, c; };  // indices into point array

// Returns triangulation as list of index triples
std::vector<Triangle2Di> delaunay_2d(const std::vector<Point2D>& pts);

// Voronoi: returns Voronoi vertices for each triangle circumcircle
std::vector<Point2D> voronoi(const std::vector<Point2D>& pts);

// ========================== KD-Tree (2D) ==========================

class KDTree2D {
public:
    struct Node {
        int idx; Node* left=nullptr; Node* right=nullptr;
        explicit Node(int i) : idx(i) {}
    };

    explicit KDTree2D(const std::vector<Point2D>& pts);
    ~KDTree2D();

    // Returns index of nearest point to query
    int nearest(Point2D q) const;
    // Returns indices of k nearest points
    std::vector<int> knn(Point2D q, int k) const;
    // Returns indices of all points within radius r
    std::vector<int> range(Point2D q, double r) const;

    const std::vector<Point2D>& points() const { return pts_; }

private:
    Node* root_ = nullptr;
    std::vector<Point2D> pts_;
    Node* build(std::vector<int>& idx, int depth);
};

// ========================== KD-Tree (3D) ==========================

class KDTree3D {
public:
    struct Node {
        int idx; Node* left=nullptr; Node* right=nullptr;
        explicit Node(int i) : idx(i) {}
    };

    explicit KDTree3D(const std::vector<Point3D>& pts);
    ~KDTree3D();
    int nearest(Point3D q) const;
    std::vector<int> knn(Point3D q, int k) const;
    std::vector<int> range(Point3D q, double r) const;

    const std::vector<Point3D>& points() const { return pts_; }

private:
    Node* root_ = nullptr;
    std::vector<Point3D> pts_;
    Node* build(std::vector<int>& idx, int depth);
};

// ========================== Intersection tests ==========================

bool intersect_seg_seg(Segment2D s1, Segment2D s2, Point2D* out = nullptr);
bool intersect_ray_tri(Ray3D ray, Triangle3D tri, double* t_out = nullptr);
bool intersect_ray_sphere(Ray3D ray, Sphere3D sphere, double* t_out = nullptr);
bool intersect_ray_aabb(Ray3D ray, AABB3D box, double* t_out = nullptr);
bool overlap_aabb_aabb(AABB3D a, AABB3D b);
bool overlap_circle_circle(Circle2D a, Circle2D b);
bool point_in_polygon(Point2D p, const Polygon2D& poly);  // winding number
bool point_in_aabb(Point2D p, AABB2D box);

// ========================== Distances ==========================

double dist_point_segment(Point2D p, Segment2D s);
double dist_point_line(Point2D p, Line2D l);
double dist_point_plane(Point3D p, Plane3D pl);
double dist_point_segment3(Point3D p, Segment3D s);

// ========================== Curves ==========================

Point2D bezier_eval(const std::vector<Point2D>& ctrl, double t);
Vec2D   bezier_deriv(const std::vector<Point2D>& ctrl, double t);
std::pair<std::vector<Point2D>, std::vector<Point2D>>
        bezier_subdivide(const std::vector<Point2D>& ctrl, double t);

Point2D bspline_eval(const std::vector<Point2D>& ctrl,
                     const std::vector<double>& knots, int degree, double t);
Point2D catmull_rom(const std::vector<Point2D>& ctrl, double t);
Point2D hermite_curve(Point2D p0, Vec2D m0, Point2D p1, Vec2D m1, double t);

// ========================== Measurements ==========================

double area(const Polygon2D& poly);
double signed_area(const Polygon2D& poly);
double perimeter(const Polygon2D& poly);
Point2D centroid(const Polygon2D& poly);
double moment_of_inertia(const Polygon2D& poly);  // about centroid

// Triangle area
double area(Point2D a, Point2D b, Point2D c);
double area(Triangle3D t);
double volume_tetrahedron(Point3D a, Point3D b, Point3D c, Point3D d);

} // namespace geo
} // namespace ms
