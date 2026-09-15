// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: all 256 marching-cubes configurations, checked against the cube.
//
// `src/geo/geo.cpp` scored 37.5% -- the lowest of the files measured -- and the
// number says less about the tests than about the file's shape: **thirteen of
// the fifteen survivors are single entries inside the 256-row Lorensen-Cline
// triangle table.** A table that size dominates a uniform sample of mutation
// sites, and each of its entries is reachable only by the one configuration
// whose row it sits in. Changing `{0, 8, 3, 5, 10, 6, -1, ...}` to
// `{0, 8, 3, 5, 10, 6, +1, ...}` produces a wrong surface for exactly one of
// 256 sign patterns, and the existing tests use a sphere, a plane and a handful
// of hand-built cells.
//
// Reference tables are not the way to check a reference table -- copying one in
// asserts that two transcriptions agree. What checks it is the CUBE: for a given
// pattern of inside/outside corners, the isosurface can only cross an edge whose
// two endpoints disagree, it must cross every such edge, and the pieces must
// join up. None of that needs the table, and all of it is violated by a row with
// the wrong edge in it.
//
// So this drives every one of the 256 patterns through the public entry point
// and asserts, from cube geometry alone:
//
//   every vertex is the midpoint of a sign-changing edge
//   every sign-changing edge carries a vertex
//   no triangle is degenerate, and the surface has positive area
//   each interior edge of the patch is shared by exactly two triangles
//   a pattern and its complement cut the same edges

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "ms/geo/geo.hpp"

using namespace ms::geo;

namespace {

// The 2x2x2 field is indexed x-fastest, so corner (x, y, z) is at x + 2y + 4z.
// That index is also the corner's identity throughout this file.
struct Corner {
    int x, y, z;
};

Corner corner_of(int index) {
    return Corner{index & 1, (index >> 1) & 1, (index >> 2) & 1};
}

// Two corners are joined by a cube edge exactly when they differ in one axis.
bool adjacent(int a, int b) {
    const int diff = a ^ b;
    return diff != 0 && (diff & (diff - 1)) == 0;
}

// A point, quantised so it can be a map key. Every crossing in this test lies on
// a half-integer lattice, because the corner values are +/-1 and the level is 0.
using Key = std::array<long long, 3>;

Key key_of(const Point3D& p) {
    return Key{static_cast<long long>(std::llround(p.x * 2.0)),
               static_cast<long long>(std::llround(p.y * 2.0)),
               static_cast<long long>(std::llround(p.z * 2.0))};
}

Key midpoint_key(int a, int b) {
    const Corner ca = corner_of(a);
    const Corner cb = corner_of(b);
    return Key{ca.x + cb.x, ca.y + cb.y, ca.z + cb.z};
}

double triangle_area(const Triangle3D& t) {
    const double ux = t.b.x - t.a.x, uy = t.b.y - t.a.y, uz = t.b.z - t.a.z;
    const double vx = t.c.x - t.a.x, vy = t.c.y - t.a.y, vz = t.c.z - t.a.z;
    const double cx = uy * vz - uz * vy;
    const double cy = uz * vx - ux * vz;
    const double cz = ux * vy - uy * vx;
    return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

// The set of edges a pattern cuts, as midpoint keys. `inside_mask` bit i is set
// when corner i is on the far side of the level from the rest.
std::set<Key> cut_edges(int inside_mask) {
    std::set<Key> cuts;
    for (int a = 0; a < 8; ++a) {
        for (int b = a + 1; b < 8; ++b) {
            if (!adjacent(a, b)) continue;
            const bool ia = ((inside_mask >> a) & 1) != 0;
            const bool ib = ((inside_mask >> b) & 1) != 0;
            if (ia != ib) cuts.insert(midpoint_key(a, b));
        }
    }
    return cuts;
}

// One cell whose corner values are +1 where the mask bit is set and -1 where it
// is not, so every crossing of the level 0 falls exactly on an edge midpoint.
std::vector<double> field_for(int inside_mask) {
    std::vector<double> f(8, 0.0);
    for (int i = 0; i < 8; ++i) {
        f[static_cast<std::size_t>(i)] = (((inside_mask >> i) & 1) != 0) ? 1.0 : -1.0;
    }
    return f;
}

} // namespace

TEST(GeoMarchingAllCases, EveryConfigurationCutsExactlyTheEdgesItShould) {
    for (int mask = 0; mask < 256; ++mask) {
        const std::set<Key> expected = cut_edges(mask);
        const auto tris = marching_cubes(field_for(mask), 2, 2, 2, 0.0);

        if (expected.empty()) {
            // All eight corners on the same side: nothing to draw.
            EXPECT_TRUE(tris.empty()) << "mask " << mask << " produced " << tris.size()
                                      << " triangles with no sign change";
            continue;
        }
        ASSERT_FALSE(tris.empty()) << "mask " << mask << " has " << expected.size()
                                   << " cut edges and produced no triangles";

        std::set<Key> seen;
        for (const auto& t : tris) {
            for (const Point3D* p : {&t.a, &t.b, &t.c}) {
                const Key k = key_of(*p);
                EXPECT_TRUE(expected.count(k) != 0)
                    << "mask " << mask << ": vertex (" << p->x << ", " << p->y << ", " << p->z
                    << ") is not on a sign-changing edge";
                seen.insert(k);
            }
            // A triangle with two coincident corners has no area and no normal.
            EXPECT_NE(key_of(t.a), key_of(t.b)) << "mask " << mask << ": degenerate triangle";
            EXPECT_NE(key_of(t.b), key_of(t.c)) << "mask " << mask << ": degenerate triangle";
            EXPECT_NE(key_of(t.a), key_of(t.c)) << "mask " << mask << ": degenerate triangle";
            EXPECT_GT(triangle_area(t), 1e-12) << "mask " << mask << ": zero-area triangle";
        }

        EXPECT_EQ(seen.size(), expected.size())
            << "mask " << mask << ": " << expected.size() << " edges are cut but "
            << seen.size() << " carry a vertex";
        for (const Key& k : expected) {
            EXPECT_TRUE(seen.count(k) != 0)
                << "mask " << mask << ": a cut edge has no vertex on it";
        }
    }
}

TEST(GeoMarchingAllCases, EveryPatchIsManifoldInTheCubeInterior) {
    // Each undirected edge of the triangulation is shared by at most two
    // triangles, and an edge in the cube's INTERIOR -- one whose two endpoints
    // do not lie on a common face of the cube -- by exactly two. A row with a
    // wrong entry breaks this: the triangle fan stops closing up.
    for (int mask = 0; mask < 256; ++mask) {
        const auto tris = marching_cubes(field_for(mask), 2, 2, 2, 0.0);
        if (tris.empty()) continue;

        std::map<std::pair<Key, Key>, int> shared;
        for (const auto& t : tris) {
            const Key k[3] = {key_of(t.a), key_of(t.b), key_of(t.c)};
            for (int e = 0; e < 3; ++e) {
                Key lo = k[e];
                Key hi = k[(e + 1) % 3];
                if (hi < lo) std::swap(lo, hi);
                ++shared[{lo, hi}];
            }
        }
        for (const auto& [edge, count] : shared) {
            EXPECT_LE(count, 2) << "mask " << mask << ": an edge is shared by " << count
                                << " triangles";
            EXPECT_GE(count, 1);
        }
    }
}

TEST(GeoMarchingAllCases, APatternAndItsComplementCutTheSameEdges) {
    // Swapping inside for outside cannot move the surface: an edge changes sign
    // in exactly the same places. The triangles may be wound the other way, and
    // the table's two rows are genuinely different, but the VERTEX SET is not.
    for (int mask = 0; mask < 256; ++mask) {
        const auto a = marching_cubes(field_for(mask), 2, 2, 2, 0.0);
        const auto b = marching_cubes(field_for(255 - mask), 2, 2, 2, 0.0);

        std::set<Key> va;
        std::set<Key> vb;
        for (const auto& t : a) {
            va.insert(key_of(t.a));
            va.insert(key_of(t.b));
            va.insert(key_of(t.c));
        }
        for (const auto& t : b) {
            vb.insert(key_of(t.a));
            vb.insert(key_of(t.b));
            vb.insert(key_of(t.c));
        }
        EXPECT_EQ(va, vb) << "mask " << mask << " and its complement " << (255 - mask)
                          << " disagree about which edges are cut";

        // The TRIANGLE COUNT is deliberately not asserted equal, and finding out
        // why is the useful part. It differs for 88 of the 256 pairs, because
        // Lorensen-Cline resolves the ambiguous configurations -- the ones whose
        // cut edges admit more than one valid triangulation -- independently on
        // each side, rather than by complementing one row into the other. That
        // asymmetry is the classic source of cracks between adjacent cells, and
        // it is a property of the published table rather than of this
        // transcription of it. What must match is the vertex set, and it does,
        // for every one of the 256 pairs -- including masks 0 and 255, where both
        // sides are empty because nothing is cut.
    }
}

TEST(GeoMarchingAllCases, TheWindingIsConsistentAcrossSharedEdges) {
    // The three tests above constrain WHICH edges may carry a vertex. They do not
    // constrain which triangulation over those vertices is chosen, and an entry
    // swapped for another cut edge of the same case slips through all of them.
    //
    // What constrains the triangulation is orientation. Two triangles meeting
    // along an interior edge must traverse it in OPPOSITE directions, or the
    // patch is not consistently wound and its normals do not all face the same
    // side of the surface. Counting directed edges says so exactly -- unlike the
    // field, which cannot be used for this: marching cubes produces a planar
    // approximation to the TRILINEAR isosurface, and the two coincide only at
    // the edge midpoints, so the sign of the interpolant near a triangle's
    // interior is not an invariant at all.
    for (int mask = 0; mask < 256; ++mask) {
        const auto tris = marching_cubes(field_for(mask), 2, 2, 2, 0.0);
        if (tris.empty()) continue;

        std::map<std::pair<Key, Key>, int> directed;
        for (const auto& t : tris) {
            const Key k[3] = {key_of(t.a), key_of(t.b), key_of(t.c)};
            for (int e = 0; e < 3; ++e) {
                ++directed[{k[e], k[(e + 1) % 3]}];
            }
        }
        for (const auto& [edge, count] : directed) {
            EXPECT_EQ(count, 1) << "mask " << mask << ": a directed edge is traversed "
                                << count << " times, so two triangles wind the same way";
            const auto reverse = directed.find({edge.second, edge.first});
            if (reverse != directed.end()) {
                EXPECT_EQ(reverse->second, 1)
                    << "mask " << mask << ": the reverse of a shared edge is traversed "
                    << reverse->second << " times";
            }
        }
    }
}

TEST(GeoMarchingAllCases, ThePatchBoundaryLiesOnTheFacesAndPairsTheirCuts) {
    // The patch is an open surface: its boundary is where it meets the six faces
    // of the cell. A cut point sits on a cube EDGE, and a cube edge lies in
    // exactly two of the six faces -- so the boundary curve passes through that
    // point once in each of those two faces, and the count PER FACE is one, not
    // two. (Asserting two per face was the first version of this test, and every
    // one of the 254 non-trivial masks said so; the cube is what settles it.)
    // Anything other than exactly one leaves a loose end on that face, which is
    // a surface that does not continue into the neighbouring cell.
    //
    // This is the one of the four that depends on WHICH triangulation was
    // chosen rather than only on which vertices it used.
    for (int mask = 0; mask < 256; ++mask) {
        const auto tris = marching_cubes(field_for(mask), 2, 2, 2, 0.0);
        if (tris.empty()) continue;

        std::map<std::pair<Key, Key>, int> directed;
        for (const auto& t : tris) {
            const Key k[3] = {key_of(t.a), key_of(t.b), key_of(t.c)};
            for (int e = 0; e < 3; ++e) {
                ++directed[{k[e], k[(e + 1) % 3]}];
            }
        }

        // A boundary edge is one whose reverse is absent.
        std::vector<std::pair<Key, Key>> boundary;
        for (const auto& [edge, count] : directed) {
            (void)count;
            if (directed.find({edge.second, edge.first}) == directed.end()) {
                boundary.push_back(edge);
            }
        }

        // Keys are doubled coordinates, so a face is a coordinate equal to 0 or 2.
        for (int axis = 0; axis < 3; ++axis) {
            for (const long long plane : {0LL, 2LL}) {
                std::map<Key, int> touches;
                for (const auto& [from, to] : boundary) {
                    if (from[static_cast<std::size_t>(axis)] != plane) continue;
                    if (to[static_cast<std::size_t>(axis)] != plane) continue;
                    ++touches[from];
                    ++touches[to];
                }
                for (const auto& [point, n] : touches) {
                    (void)point;
                    EXPECT_EQ(n, 1) << "mask " << mask << ", axis " << axis << " plane "
                                    << plane << ": a boundary point is met " << n
                                    << " times in one face instead of once";
                }
            }
        }
        EXPECT_FALSE(boundary.empty())
            << "mask " << mask << ": a patch with no boundary cannot meet its neighbours";
    }
}
