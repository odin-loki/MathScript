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

struct Triangle3Di { int a, b, c; };  // indices into point array

// Brute-force face enumeration (O(n^4)): for every triple of points, checks
// whether all remaining points lie on one side of the plane through them.
// Suited to the dozens-to-low-hundreds point counts this library targets,
// not large point clouds. Faces with more than 3 coplanar hull vertices
// (e.g. a cube's square faces) are fan-triangulated into their minimal
// triangle count rather than emitted once per combinatorial triple. Returns
// outward-facing triangles with consistent winding (normal via
// cross(b-a, c-a) points away from the point cloud). Fewer than 4 points, or
// a coplanar/collinear input, yields an empty result since no plane through
// any triple then has every other point strictly on one side bounding a 3D
// volume.
std::vector<Triangle3Di> convex_hull_3d(const std::vector<Point3D>& points);

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
    Node* build(std::vector<int>& idx, int beg, int end, int depth);
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
    Node* build(std::vector<int>& idx, int beg, int end, int depth);
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

// ========================== Bounding Rectangles ==========================

// Minimum-area oriented bounding rectangle of a 2D point set, computed via the rotating
// calipers technique over the point set's convex hull: since the minimum-area bounding
// rectangle of any point set always has at least one side flush with a convex hull edge,
// this only needs to test each hull edge's orientation as a rectangle candidate (rather than
// searching over all possible angles), giving an O(h) algorithm after O(n log n) hull
// construction (h = hull vertex count).
struct MinBoundingRect {
    Point2D center;       // rectangle center
    double width = 0.0;   // extent along `angle` direction
    double height = 0.0;  // extent along the direction perpendicular to `angle`
    double angle = 0.0;   // orientation of the `width` axis, in radians, measured from +x axis
    double area() const { return width * height; }
    // Returns the 4 corners of the rectangle, starting from the local corner
    // (-width/2, -height/2) and proceeding counter-clockwise: (-w/2,-h/2), (+w/2,-h/2),
    // (+w/2,+h/2), (-w/2,+h/2). Each local corner is rotated about the origin by `angle`
    // (standard CCW rotation matrix [[cos,-sin],[sin,cos]]) and then translated by `center`.
    std::array<Point2D, 4> corners() const;
};

// @param points input 2D point set. Fewer than 3 points (or fully degenerate/collinear input
//        that convex_hull_2d cannot form a proper hull from) returns a MinBoundingRect with
//        width/height derived defensively from the points' axis-aligned bounding box instead
//        (still a valid, sensible non-crashing result, just not from the rotating-calipers path).
MinBoundingRect min_bounding_rect(const std::vector<Point2D>& points);

// ========================== Minkowski Sum ==========================

// Minkowski sum of two convex polygons A and B: the set {a + b : a in A, b in B}, which is
// itself always a convex polygon when A and B are both convex. Computed via the efficient
// O(|A|+|B|) "merge by angle" algorithm (valid ONLY because both inputs are convex -- this is
// NOT a general-purpose Minkowski sum for arbitrary/non-convex polygons, which is a much
// harder problem out of scope here):
//   1. Both inputs are required to be in CCW order (matching this module's convention, e.g.
//      convex_hull_2d's output) -- this is a precondition, NOT defensively checked/corrected,
//      since silently reversing a caller's polygon would hide bugs; a CW-wound input is
//      documented as producing unreliable/undefined results.
//   2. Find the starting vertex of each polygon: the lowest-y, then lowest-x, vertex (the
//      "bottom-most" vertex), a standard choice that ensures a clean angular merge.
//   3. Walk around both polygons simultaneously starting from their respective starting
//      vertices, at each step comparing the polar angle of the current edge vector of A
//      against the current edge vector of B, and advancing whichever polygon's current edge
//      has the smaller angle (adding that edge vector to a running sum point and appending the
//      result to the output), until both polygons have been fully traversed (standard "merge
//      two sorted-by-angle edge lists" pattern, analogous to merging two sorted arrays).
//
// @param a, b convex polygons in CCW order. Fewer than 3 vertices in either input is a
//        degenerate case (no well-defined "edge list" to angularly merge) and is handled via
//        a brute-force fallback instead: every pairwise sum of a vertex of `a` and a vertex of
//        `b`, followed by convex_hull_2d over those candidate points. Note this fallback still
//        assumes convex inputs; it does not generalise this function to non-convex polygons.
//        Both empty returns an empty polygon.
// @return the Minkowski sum polygon, convex and in CCW order.
Polygon2D minkowski_sum_convex(const Polygon2D& a, const Polygon2D& b);

// ========================== Polygon Triangulation ==========================

// Ear-clipping triangulation of a simple (non-self-intersecting) 2D polygon. Repeatedly
// finds a valid "ear" — a triangle formed by three consecutive vertices where the middle
// vertex is convex and no other remaining polygon vertex lies strictly inside that triangle
// — and clips it off until three vertices remain. Works for both convex and concave simple
// polygons (unlike naive fan triangulation from a single vertex, which fails on reflex
// vertices). Standard O(n^2) textbook implementation.
//
// @param poly ordered vertex ring of a simple polygon. Fewer than 3 vertices, fewer than 3
//        distinct vertices after removing duplicate consecutive points, or a polygon where
//        no valid ear can be found (e.g. self-intersecting input violating the simple-polygon
//        precondition) returns an empty result rather than crashing or looping indefinitely.
//        Collinear consecutive vertices are skipped as non-ears; if the walk stalls with only
//        collinear/degenerate ears left, remaining collinear vertices may be removed without
//        emitting a triangle.
// @return triangle index triples referencing the original `poly` vertex array (`n`-gon yields
//         exactly `n-2` triangles when successful).
// @note Requires a SIMPLE polygon as a precondition — self-intersecting input is undefined
//       and typically yields an empty result once ear detection stalls.
std::vector<Triangle2Di> triangulate_polygon(const Polygon2D& poly);

// ========================== Polygon Clipping ==========================

// Sutherland-Hodgman polygon clipping: intersects an arbitrary simple `subject` polygon against
// a CONVEX `clip_window`, returning the clipped result's vertices in order. Standard textbook
// algorithm: for each edge of the clip window (taken in the order given), the current working
// polygon is clipped against the half-plane bounded by that edge — keeping only the side the
// clip window's interior lies on, and inserting an edge-intersection point wherever the working
// polygon crosses the boundary — producing a new intermediate polygon; this repeats for every
// clip-window edge in turn, and the final intermediate polygon (after all edges) is the result.
//
// @param subject an arbitrary simple polygon (convex OR concave — e.g. an L-shape is fine).
//        An empty subject returns an empty result.
// @param clip_window a CONVEX polygon. Either winding order is accepted: orientation is
//        detected via `signed_area` and the interior half-plane test is flipped to match, so
//        both CCW and CW clip windows produce correct results. Fewer than 3 vertices is
//        degenerate (no well-defined interior half-plane to clip against) and returns an empty
//        result rather than crashing.
// @return the clipped polygon's vertices in order, or an empty polygon if `subject` and
//         `clip_window` do not overlap at all.
// @note Requires the clip window to be CONVEX as a precondition — this is NOT checked. A
//       non-convex clip window is out of scope: clipping against each edge's half-plane
//       independently cannot represent a concave region as an intersection of half-planes, so
//       results for a concave `clip_window` are undefined (typically some spurious extra area
//       that should have been excluded survives). The `subject` polygon has no such restriction.
Polygon2D clip_polygon(const Polygon2D& subject, const Polygon2D& clip_window);

// ========================== Polygon Union ==========================

// Union of two CONVEX polygons. Overlapping/touching operands are merged by collecting
// exterior vertices and edge–edge intersections, then `convex_hull_2d` on that set.
// This is exact when the union region is itself convex (e.g. two overlapping axis-aligned
// rectangles); when the union region is non-convex — which can occur even with convex
// inputs — the hull step returns a documented convex over-approximation of the union.
//
// Disjoint operands cannot be represented as one polygon; the implementation falls back
// to `convex_hull_2d` over all vertices (also an over-approximation bridging the gap).
//
// Degenerate inputs (<3 vertices in either polygon, or either operand empty) also use the
// hull fallback over all vertices.
//
// @param a, b convex polygons in any winding (both CCW is the usual convention in this
//        module). Fewer than 3 vertices in either operand uses the hull fallback above.
//        Both empty returns an empty polygon.
// @return the union polygon in CCW order (via `convex_hull_2d`), or empty when both inputs
//         are empty / produce no area.
// @note Requires CONVEX inputs as a precondition — non-convex polygons are out of scope and
//       yield undefined/over-approximated results. This is an MVP boolean-union helper, not a
//       general simple-polygon clipper (cf. `clip_polygon` for intersection against a convex
//       window).
Polygon2D poly_union(const Polygon2D& a, const Polygon2D& b);

// ========================== Polygon Intersection ==========================

// Intersection of two CONVEX polygons. Computed via Sutherland-Hodgman clipping: `a` is
// clipped against each half-plane of convex `b` in turn (reusing `clip_polygon`), yielding
// the overlap region A ∩ B. The result is always convex when non-empty.
//
// @param a, b convex polygons in any winding (CCW is the usual convention in this module).
//        Fewer than 3 vertices in either operand is degenerate (no well-defined polygon
//        interior) and returns an empty result. Either operand empty also returns empty.
// @return the intersection polygon's vertices in order, or empty when the operands do not
//         overlap (including edge-only or point-only contact with zero area).
// @note Requires CONVEX inputs as a precondition — non-convex polygons are out of scope.
//       For concave subjects against a convex clip window, use `clip_polygon` directly.
Polygon2D poly_intersect(const Polygon2D& a, const Polygon2D& b);

// ========================== Polygon Difference ==========================

// Set difference A \ B of two CONVEX polygons. Computed by clipping `a` against each
// exterior half-plane of convex `b` (complement of the Sutherland-Hodgman interior
// half-planes used by `clip_polygon` / `poly_intersect`) and selecting the clip whose
// area matches area(A) − area(A ∩ B). When no single half-plane clip matches — i.e. the
// difference is non-convex — falls back to `convex_hull_2d` over candidate boundary
// points (documented convex over-approximation).
//
// Exact when A \ B is itself convex (e.g. two overlapping axis-aligned rectangles that
// leave a rectangular remnant). When the difference region is non-convex — which can
// occur even with convex inputs, e.g. B cutting a corner bite from A — the hull fallback
// returns a documented convex over-approximation of A \ B.
//
// @param a, b convex polygons in any winding (CCW is the usual convention in this module).
//        Fewer than 3 vertices in `a` is degenerate and returns empty. Empty or degenerate
//        `b` leaves `a` unchanged (subtracting nothing). If `a` is contained in `b`,
//        returns empty.
// @return the difference polygon in CCW order (via `convex_hull_2d`), or empty when
//         A ⊆ B / A has no area after subtraction.
// @note Requires CONVEX inputs as a precondition — non-convex polygons are out of scope
//       and yield undefined/over-approximated results. This is an MVP boolean-difference
//       helper, not a general simple-polygon clipper with holes.
Polygon2D poly_diff(const Polygon2D& a, const Polygon2D& b);

// ========================== Isosurface Extraction ==========================

// Indexed triangle mesh: a shared vertex array plus index triples into it. Produced by
// `marching_cubes_mesh`, which merges the two (or more) cells that share a grid edge onto a
// single vertex, so a closed isosurface comes out watertight (every triangle edge shared by
// exactly two triangles) rather than as an unwelded triangle soup.
struct TriMesh3D {
    std::vector<Point3D> vertices;
    std::vector<Triangle3Di> triangles;  // indices into `vertices`
};

// Marching cubes (Lorensen & Cline 1987): extracts the isosurface f = iso from a scalar field
// sampled on a regular 3D grid, as a triangle soup.
//
// The field is a flat array of nx*ny*nz samples indexed x-fastest,
//     field[i + nx * (j + ny * k)]   is the sample at grid node (i, j, k),
// which sits at world position origin + (i*spacing.x, j*spacing.y, k*spacing.z). The grid is
// swept cell by cell ((nx-1)*(ny-1)*(nz-1) cells); each cell's 8 corners are classified with
// the strict test `value < iso` into an 8-bit case index using the standard corner numbering
//     0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
// (bit c set means corner c is below the iso level). The 256-entry edge table then names the
// cut edges of the standard 12-edge numbering
//     0:0-1 1:1-2 2:2-3 3:3-0  4:4-5 5:5-6 6:6-7 7:7-4  8:0-4 9:1-5 10:2-6 11:3-7
// and the 256-row triangle table names the triangles as edge triples. The crossing point on a
// cut edge is placed by linear interpolation of the two endpoint samples,
//     t = (iso - v0) / (v1 - v0),   p = p0 + t * (p1 - p0),
// always evaluated from the lexicographically lower grid endpoint of the edge, so two cells
// sharing an edge compute bit-identical crossing points and the surface has no cracks.
//
// The extracted surface is the boundary of the open sub-level set {f < iso} ("the solid"), and
// triangles are wound so that cross(b - a, c - a) points OUT of that solid, i.e. towards
// increasing field. A sample exactly equal to `iso` counts as above the level (outside), which
// is what makes an iso level at or below the field minimum, or strictly above its maximum,
// yield an empty result rather than a degenerate sheet.
//
// @param field nx*ny*nz samples, x-fastest. A size other than nx*ny*nz returns empty.
// @param nx, ny, nz grid dimensions. Fewer than 2 in any dimension means there is no cell to
//        march at all and returns an empty result (this includes the empty field).
// @param iso the level to extract. An iso level at or below min(field), or strictly above
//        max(field), leaves every cell uncut and returns an empty result.
// @param origin, spacing world placement of grid node (0,0,0) and the node pitch along each
//        axis. Zero or negative spacing is not rejected; it simply degenerates or mirrors the
//        output geometry the same way it degenerates the grid.
// @return the isosurface as an unindexed triangle soup, in cell-sweep order (k, then j, then
//         i). Empty when no cell is cut.
// @note Cells containing a non-finite sample (NaN or infinity) are skipped entirely, so a
//       corrupted region of the field punches a hole in the surface instead of poisoning the
//       output with non-finite coordinates.
// @note Zero-area triangles can appear where a sample sits exactly on the iso level (two cut
//       edges then interpolate to the same grid node). They are kept rather than filtered, so
//       the triangle count stays a pure function of the case table; they contribute nothing to
//       `mesh_surface_area` or `mesh_volume`.
// @note Faces of the sampled box are NOT capped: an isosurface that reaches the outer boundary
//       comes out as an open sheet there. See the precondition on `mesh_volume`.
// @note These are the crack-free Lorensen-Cline tables, not Marching Cubes 33. Within a single
//       cell they can pick a topology that the trilinear interpolant would not (the classic
//       ambiguous-face objection); this affects interior topology on under-resolved features,
//       never watertightness of the result.
// Complexity: O(nx*ny*nz) time; O(nx*ny*nz) auxiliary memory (3 ints per grid node for the
// edge-to-vertex map, i.e. about 1.5x the size of the input field), plus the output.
std::vector<Triangle3D> marching_cubes(const std::vector<double>& field,
                                       int nx, int ny, int nz, double iso,
                                       Point3D origin = {0.0, 0.0, 0.0},
                                       Vec3D spacing = {1.0, 1.0, 1.0});

// Marching cubes producing an indexed mesh with de-duplicated vertices: identical algorithm,
// identical vertex positions and identical triangle order as `marching_cubes`, but each grid
// edge contributes exactly one shared vertex instead of one copy per incident triangle. For a
// closed isosurface (one that does not reach the outer boundary of the sampled box) the result
// is watertight and consistently oriented: every directed triangle edge appears exactly once
// and its reverse exactly once, and V - E + F = 2 for a genus-0 component.
//
// @param field, nx, ny, nz, iso, origin, spacing as `marching_cubes`.
// @return the indexed mesh; both `vertices` and `triangles` are empty when no cell is cut, and
//         for the degenerate inputs listed on `marching_cubes`.
// Complexity: O(nx*ny*nz), same auxiliary memory as `marching_cubes`.
TriMesh3D marching_cubes_mesh(const std::vector<double>& field,
                              int nx, int ny, int nz, double iso,
                              Point3D origin = {0.0, 0.0, 0.0},
                              Vec3D spacing = {1.0, 1.0, 1.0});

// Marching squares: the 2D analogue of `marching_cubes`, extracting the contour f = iso from a
// scalar field sampled on a regular 2D grid as a list of directed line segments.
//
// The field is a flat array of nx*ny samples indexed x-fastest, field[i + nx * j] at grid node
// (i, j), placed at origin + (i*spacing.x, j*spacing.y). Each cell's 4 corners are classified
// with the strict test `value < iso` into a 4-bit case index using the corner numbering
// 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1) and the edge numbering 0:0-1 1:1-2 2:2-3 3:3-0, giving 16
// cases; the crossing point on a cut edge is linearly interpolated exactly as in 3D, from the
// lexicographically lower grid endpoint so neighbouring cells agree bit for bit.
//
// Segments are directed so that the sub-level region {f < iso} lies to the LEFT of a -> b: a
// closed contour around a simply-connected region comes out counter-clockwise, so the shoelace
// sum 0.5 * sum(a.x*b.y - b.x*a.y) over the returned segments is the positive enclosed area.
//
// Saddle-point disambiguation: the two diagonal cases (case 5, corners 0 and 2 below the level;
// case 10, corners 1 and 3 below it) cut all four edges and admit two topologies -- the two
// like-signed corners joined through the middle of the cell, or separated into two corner cuts.
// They are resolved with the ASYMPTOTIC DECIDER (Nielson & Hamann 1991): the bilinear
// interpolant over the cell has a single saddle point whose value is
//     f_saddle = (f0*f2 - f1*f3) / (f0 - f1 + f2 - f3)
// with f0..f3 the corner values in cyclic order. If f_saddle < iso the cell centre belongs to
// the sub-level set, so the two below-level corners are JOINED through it; otherwise they are
// SEPARATED. This is the topology of the actual bilinear interpolant, not a coin toss, and it
// agrees with what the neighbouring cells see because edge crossings depend only on the shared
// edge. When the denominator is exactly zero (the bilinear patch is degenerate and has no
// saddle) the rule falls back to the cell-centre value, the mean of the four corners.
//
// @param field nx*ny samples, x-fastest. A size other than nx*ny returns empty.
// @param nx, ny grid dimensions. Fewer than 2 in either dimension returns an empty result.
// @param iso the level to extract; at or below min(field), or strictly above max(field),
//        returns an empty result.
// @param origin, spacing world placement of grid node (0,0) and the node pitch per axis.
// @return the contour as directed segments, in cell-sweep order (j, then i). Empty when no cell
//         is cut. Cells with a non-finite sample are skipped, as in 3D.
// @note The boundary of the sampled box is not closed off, so a contour that runs off the edge
//       of the grid comes out as an open chain there.
// Complexity: O(nx*ny) time, O(1) auxiliary memory beyond the output (no vertex table is needed
// because the segment list is not indexed).
std::vector<Segment2D> marching_squares(const std::vector<double>& field,
                                        int nx, int ny, double iso,
                                        Point2D origin = {0.0, 0.0},
                                        Vec2D spacing = {1.0, 1.0});

// ========================== Mesh Measurements ==========================

// Total surface area of a triangle mesh: the sum of `area(Triangle3D)` over every triangle.
// Winding and orientation are irrelevant (triangle area is unsigned), so this is meaningful for
// open sheets, closed surfaces and unstructured soups alike.
//
// @param tris / mesh the triangles. An empty mesh returns 0.0. In the indexed overload,
//        triangles whose indices fall outside `vertices` are skipped rather than dereferenced.
// Complexity: O(number of triangles).
double mesh_surface_area(const std::vector<Triangle3D>& tris);
double mesh_surface_area(const TriMesh3D& mesh);

// Signed volume enclosed by a closed triangle mesh, via the divergence theorem: integrating
// div(p/3) = 1 over the enclosed region turns into a surface integral that reduces, for a
// triangulated boundary, to the sum of signed tetrahedron volumes from the origin,
//     V = (1/6) * sum over triangles of  a . (b x c).
// The origin cancels out for a closed surface, so the result does not depend on where the mesh
// sits. With the outward orientation that `marching_cubes` produces (normal cross(b-a, c-a)
// pointing out of the solid {f < iso}) the result is positive.
//
// @param tris / mesh a CLOSED, consistently oriented triangle mesh. An empty mesh returns 0.0.
//        In the indexed overload, out-of-range index triples are skipped.
// @return the enclosed volume, positive for an outward-oriented closed mesh and negative for an
//         inward-oriented one.
// @note Requires a closed surface as a precondition -- this is NOT checked. For an open sheet
//       (for instance an isosurface that runs off the edge of the sampled box, which this module
//       does not cap) the sum is the volume of the cone from the origin over the sheet: a finite
//       number, but origin-dependent and not an enclosed volume.
// Complexity: O(number of triangles).
double mesh_volume(const std::vector<Triangle3D>& tris);
double mesh_volume(const TriMesh3D& mesh);

} // namespace geo
} // namespace ms
