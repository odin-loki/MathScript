// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Arguments that used to end the process.
//
// Every command here takes a count, an order or a grid extent, converted it with
//
//     const int n_i = static_cast<int>(n_d);
//     if (n_i < 0 || n_d != n_i) { ... }
//
// and handed the result straight to an allocation. Three things are wrong with that
// and the guard caught none of them:
//
//   - the cast IS the check. `static_cast<int>` of a double outside `int`'s range is
//     undefined behaviour, not a wrap, so by the time the guard reads `n_i` there is
//     no value there to test;
//   - a count that fits in an `int` is not a count that is affordable --
//     `fem_poisson1d(100000000)` is a perfectly ordinary `int` and asks for 800 MB;
//   - a cap on each extent alone is not a cap on the allocation. What gets allocated
//     is the PRODUCT, and `imresize(A, 100000, 100000)` names two extents that each
//     look like a resolution.
//
// The library is built with `-fno-exceptions`, so the `std::bad_alloc` that came out
// of `std::vector` reached `std::terminate`: the process was gone, with nothing on
// either stream, before it could say what had happened. Which is why the strongest
// assertion in this file is that it runs to the end at all -- a regression here does
// not fail a test, it takes the executable down with it.
//
// Each command is asserted twice: once at a value that used to be fatal, and once at
// a small one, because a guard that refuses everything is not a fix.

#include <string>

#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"

#include "repl/repl_test_helpers.hpp"

using ms::interp::Interpreter;

namespace {

/// A 2x2 to hand the image and ML commands, so the only large number in the call is
/// the one under test.
void seed(Interpreter& interp) {
    expect_ok(interp, "A = [1, 2; 3, 4]");
}

}  // namespace

TEST(ReplSizeArguments, FemPoissonRefusesAGridItCannotAllocate) {
    Interpreter interp;
    expect_error_contains(interp, "fem_poisson1d(100000000)", "n 100000000 is too large");
    // The no-assignment form of fem_poisson1d goes through the matrix-call registry
    // and prints under `_`; fem_poisson2d and 3d keep their own hand-written branches
    // and print under `u`. Both paths had the defect and both are guarded.
    expect_contains(interp, "fem_poisson1d(4)", "_ =");

    // Neither extent is large. Their product is ten billion, which is the case a
    // per-extent cap is blind to.
    expect_error_contains(interp, "fem_poisson2d(100000, 100000)", "is too large");
    expect_contains(interp, "fem_poisson2d(3, 3)", "u =");

    expect_error_contains(interp, "fem_poisson3d(5000, 5000, 5000, 0, 0, 0)", "is too large");
    expect_contains(interp, "fem_poisson3d(2, 2, 2, 0, 0, 0)", "u =");
}

TEST(ReplSizeArguments, CfdAdvectionRefusesAGridItCannotAllocate) {
    Interpreter interp;
    expect_error_contains(interp, "cfd_advection2d(100000, 100000, 1, 0, 0.1, 0.01)",
                          "is too large");
    expect_contains(interp, "cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)", "u =");
    expect_error_contains(interp, "x = cfd_advection3d(2000, 2000, 2000, 1, 0, 0, 0.1, 0.01)",
                          "is too large");
    expect_ok(interp, "x = cfd_advection3d(4, 4, 4, 1, 0, 0, 0.1, 0.01)");
    // One extent, so no product to catch it -- 1e8 alone is 800 MB.
    expect_error_contains(interp, "cfd_advection1d(100000000, 1, 0.5, 0.01)",
                          "nx 100000000 is too large");
    expect_contains(interp, "cfd_advection1d(8, 1, 0.5, 0.01)", "_ =");
}

TEST(ReplSizeArguments, FareyCountsItsSequenceBeforeBuildingIt) {
    Interpreter interp;
    // |F_n| is about 0.304 n^2, so the length is quadratic in an argument that reads
    // as an ordinary count: order 1000000 is three hundred billion fractions. The
    // message quotes the true count rather than an estimate, because an estimate
    // would have to be conservative and would then refuse an order that fits.
    expect_error_contains(interp, "y = numthy_farey(1000000)", "the Farey sequence of order");
    expect_error_contains(interp, "y = numthy_farey(1000000)", "the result is limited to");
    expect_ok(interp, "y = numthy_farey(5)");
    // The order just past the budget is refused with its exact length, not with a
    // guess -- 131072 rows is the cap, and F_658 is the first order over it.
    expect_error_contains(interp, "y = numthy_farey(2000)", "fractions in it");
}

TEST(ReplSizeArguments, ImageResizingRefusesAResultItCannotAllocate) {
    Interpreter interp;
    seed(interp);
    // The padded result grows on all FOUR sides, so its element count goes as the
    // square of the padding: a pad of a million is four trillion elements.
    expect_error_contains(interp, "B = impad(A, 1000000)", "too large");
    expect_ok(interp, "B = impad(A, 1)");

    expect_error_contains(interp, "C = imresize(A, 100000, 100000)", "is too large");
    expect_ok(interp, "C = imresize(A, 4, 4)");
    // imresize never checked integrality either, so a fractional extent was silently
    // truncated rather than reported.
    expect_error_contains(interp, "C = imresize(A, 2.5, 4)",
                          "expected non-negative integer rows");
}

TEST(ReplSizeArguments, HoughAccumulatorsAreBoundedByTheirProduct) {
    Interpreter interp;
    seed(interp);
    // One cell per (theta, rho) pair: each resolution alone reads as a plausible
    // number and together they are ten quadrillion cells.
    expect_error_contains(interp, "G = hough_lines(A, 0.5, 100000000, 100000000, 1)",
                          "is too large");
    expect_ok(interp, "G = hough_lines(A, 0.5, 8, 8, 1)");

    // One cell per (radius, row, column): the radius count multiplies a bounded image
    // rather than adding to it.
    expect_error_contains(interp, "H = hough_circles(A, 1, 100000000)", "radii");
    expect_ok(interp, "H = hough_circles(A, 1, 3)");
}

TEST(ReplSizeArguments, TheMlFitsAskForAShapeRatherThanASize) {
    Interpreter interp;
    seed(interp);
    // These two are not size caps and should not be. A principal component is a
    // direction in feature space and there are only min(samples, features) of them;
    // k clusters need k points to put in them. Asking for a hundred million of either
    // is not an expensive request, it is a request with no answer -- and it was
    // treated as the former, sized an allocation from the count, and ended the
    // process for a matrix with two rows in it.
    expect_error_contains(interp, "D = ml_pca_fit(A, 100000000)", "1 <= n_components <= 2");
    expect_error_contains(interp, "D = ml_pca_fit(A, 3)", "1 <= n_components <= 2");
    expect_ok(interp, "D = ml_pca_fit(A, 1)");

    expect_error_contains(interp, "F = ml_pca_fit_transform(A, 100000000)",
                          "1 <= n_components <= 2");
    expect_ok(interp, "F = ml_pca_fit_transform(A, 1)");

    expect_error_contains(interp, "E = ml_kmeans_fit(A, 100000000)", "1 <= k <= 2");
    expect_error_contains(interp, "E = ml_kmeans_fit(A, 3)", "the number of rows");
    expect_ok(interp, "E = ml_kmeans_fit(A, 1)");
}

TEST(ReplSizeArguments, AnExtentIsRangeCheckedOnTheDoubleRatherThanAfterTheCast) {
    Interpreter interp;
    seed(interp);
    // `static_cast<int>(1e18)` is undefined behaviour. On x86-64 it happens to yield
    // INT_MIN, so the `n_i < 0` that followed it rejected these inputs by accident --
    // an accident that reads exactly like a guard and is not one. The range is decided
    // on the double now, before any conversion.
    for (const char* huge : {"1e18", "1e300", "1e9999"}) {
        const std::string call = std::string("fem_poisson1d(") + huge + ")";
        expect_error_contains(interp, call, "fem_poisson1d");
    }
    expect_error_contains(interp, "fem_poisson1d(2.5)", "expected non-negative integer n");
    expect_error_contains(interp, "fem_poisson1d(-1)", "expected non-negative integer n");
    expect_error_contains(interp, "C = imresize(A, 1e18, 2)", "rows");
}
