// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Arguments the LINEAR cap admits and the command cannot survive.
//
// `test_repl_commands_size_arguments.cpp` next door covers the arguments that were
// fatal because the conversion was undefined or because one extent bought a huge
// allocation. This file covers the ones that get through all of that: an argument
// that is an ordinary integer, well inside `kMaxReplIntegerArgument`, and that still
// ends the process or the afternoon -- because the command's cost is not linear in it.
//
//   - `finance_binomial_call(S,K,T,r,sigma,10000000)` asks for 5e13 node visits,
//     roughly twelve days.
//   - `pde_heat_1d(ones(200,1), 0.1, 0.1, 0.001, 10000000)` keeps one grid per step
//     and asks for a 16 GB history in order to return 200 numbers. Under
//     `-fno-exceptions` that `std::bad_alloc` reaches `std::terminate` and the process
//     is gone with nothing on either stream. Nine of these were measured aborting.
//   - `gria_alpha_ca(30, 10000000, 10000000)` reserves the product as doubles: 800 TB.
//
// None of them was reachable by the oversized-argument sweep, and the reason is worth
// keeping written down: that sweep probes 3000000000 and 1e18, which the linear cap
// REJECTS. A sweep built out of values the guard turns away cannot find a command that
// dies on a value the guard lets through.
//
// Every command is asserted twice -- once at the value that was fatal, once at a value
// anyone would actually type -- because a guard that refuses everything is not a fix.

#include <string>

#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"

#include "repl/repl_test_helpers.hpp"

using ms::interp::Interpreter;

TEST(ReplSuperlinearArguments, ABinomialTreeIsBoundedBelowTheLinearCap) {
    Interpreter interp;
    // 1e7 is a fine integer and a fine LINEAR budget. The tree visits steps^2/2 nodes.
    for (const auto* call : {"finance_binomial_call(100,100,1,0.05,0.2,10000000)",
                             "finance_binomial_put(100,100,1,0.05,0.2,10000000)",
                             "finance_american_option(100,100,1,0.05,0.2,1,10000000)",
                             "finance_trinomial_option(100,100,1,0.05,0.2,10000000,1,1)"}) {
        expect_error_contains(interp, call, "is too large");
        expect_error_contains(interp, call, "work proportional to");
    }
    // The tree converges like 1/steps and is done long before the bound, so nothing a
    // user would write is refused.
    expect_ok(interp, "finance_binomial_call(100,100,1,0.05,0.2,500)");
    expect_ok(interp, "finance_binomial_put(100,100,1,0.05,0.2,500)");
    expect_ok(interp, "finance_american_option(100,100,1,0.05,0.2,1,500)");
    expect_ok(interp, "finance_trinomial_option(100,100,1,0.05,0.2,500,1,1)");
}

TEST(ReplSuperlinearArguments, TheBoundFollowsTheCostRatherThanTheExponent) {
    // Both of these are quadratic, and a single shared "quadratic arguments" cap would
    // be wrong for both: a binomial node is a multiply-add at about 11 ns, a
    // Gauss-Bonnet grid point is a numerical quadrature of a curvature tensor at about
    // 1970 ns. Measured, diffgeo_sphere_gauss_bonnet(3000) took 17.4 s -- a value the
    // tree's bound would have admitted without comment.
    Interpreter interp;
    expect_ok(interp, "finance_binomial_call(100,100,1,0.05,0.2,3000)");
    expect_error_contains(interp, "diffgeo_sphere_gauss_bonnet(3000)", "is too large");
    expect_error_contains(interp, "diffgeo_sphere_gauss_bonnet_residual(3000)", "is too large");
    // The grid that actually demonstrates Gauss-Bonnet on a sphere is a small one.
    expect_ok(interp, "diffgeo_sphere_gauss_bonnet(100)");
    expect_ok(interp, "diffgeo_sphere_gauss_bonnet_residual(100)");
}

TEST(ReplSuperlinearArguments, ATimeSteppingSolverCannotBeAskedForItsWholeHistory) {
    Interpreter interp;
    expect_ok(interp, "U = ones(200,1)");
    expect_ok(interp, "V = zeros(200,1)");
    expect_ok(interp, "M = ones(30,30)");
    expect_ok(interp, "W = zeros(30,30)");
    // Each of these was measured aborting the process at steps = 1e7.
    for (const auto* call : {"pde_heat_1d(U,0.1,0.1,0.001,10000000)",
                             "pde_heat_1d_cn(U,0.1,0.1,0.001,10000000)",
                             "pde_advection_1d(U,1,0.1,0.001,10000000)",
                             "pde_advection_1d_lax_wendroff(U,1,0.1,0.001,10000000)",
                             "pde_reaction_diffusion_1d(U,0.1,1,0.1,0.001,10000000)",
                             "pde_burgers_1d(U,0.1,0.1,0.001,10000000)",
                             "pde_wave_1d(U,V,1,0.1,0.001,10000000)",
                             "pde_heat_2d(M,0.1,0.1,0.1,0.001,10000000)",
                             "pde_heat_2d_cn_adi(M,0.1,0.1,0.1,0.001,10000000)",
                             "pde_wave_2d(M,W,1,0.1,0.1,0.001,10000000)"}) {
        expect_error_contains(interp, call, "is too large");
        expect_error_contains(interp, call, "the size of what it is given");
    }
    // And a run anyone would ask for still returns.
    expect_contains(interp, "pde_heat_1d(U,0.1,0.1,0.001,100)", "u =");
    expect_contains(interp, "pde_heat_2d(M,0.1,0.1,0.1,0.001,100)", "u =");
    expect_contains(interp, "pde_wave_1d(U,V,1,0.1,0.001,100)", "u =");
}

TEST(ReplSuperlinearArguments, TheSolverBoundShrinksAsTheGridGrows) {
    // The relationship the bound has to encode is that steps and the grid multiply. A
    // per-argument cap cannot say that, which is why the same steps count is fine on a
    // small grid and refused on a large one.
    Interpreter interp;
    expect_ok(interp, "SMALL = ones(20,1)");
    expect_ok(interp, "BIG = ones(400,400)");
    expect_contains(interp, "pde_heat_1d(SMALL,0.1,0.1,0.001,20000)", "u =");
    expect_error_contains(interp, "pde_heat_2d(BIG,0.1,0.1,0.1,0.001,20000)", "is too large");
}

TEST(ReplSuperlinearArguments, AMonteCarloIsBoundedOnItsProductAndNotOnEitherFactor) {
    Interpreter interp;
    // n_paths and n_steps are each ordinary at 1e7; together they are 1e14 paths-steps.
    expect_error_contains(interp,
                          "finance_mc_asian_call(100,100,1,0.05,0.2,10000000,10000000,42)",
                          "is too large");
    // A large path count is a REQUEST, not a slip -- Monte Carlo error falls like
    // 1/sqrt(n_paths), so this is the command doing its job and it still runs.
    expect_ok(interp, "finance_mc_european_call(100,100,1,0.05,0.2,100000,42)");
    expect_ok(interp, "finance_mc_asian_call(100,100,1,0.05,0.2,20000,100,42)");
}

TEST(ReplSuperlinearArguments, ACellularAutomatonIsBoundedOnStepsTimesWidth) {
    Interpreter interp;
    // alpha_ca reserves steps*width doubles for the history it measures the entropy of.
    expect_error_contains(interp, "gria_alpha_ca(30,10000000,10000000)", "is too large");
    expect_ok(interp, "gria_alpha_ca(30,200,200)");
}

TEST(ReplSuperlinearArguments, APropagatorIsBoundedOnStepsTimesTheOperator) {
    Interpreter interp;
    expect_ok(interp, "H = eye(8)");
    expect_ok(interp, "PSI = ones(8,1)");
    expect_error_contains(interp, "quantum_schrodinger_final(H,PSI,0,1,10000000)",
                          "is too large");
    expect_contains(interp, "quantum_schrodinger_final(H,PSI,0,1,100)", "psi =");
}

TEST(ReplSuperlinearArguments, AFactorisationRankIsBoundedByCostAndNotOnlyByShape) {
    Interpreter interp;
    expect_ok(interp, "T = ones(80,80)");
    // The ceiling here used to be "rank <= the tensor's element count", which is a
    // statement about whether the decomposition is informative and says nothing about
    // what it costs. Under it, `tensorops_decompose_nmf(h, ones(80,80), 200)` -- an
    // entirely ordinary request at rank 200 of 6400 -- had not finished after 45 s,
    // because NMF runs all max_iter sweeps of a rank x numel factor pair.
    expect_error_contains(interp, "tensorops_decompose_nmf(nmf_big,T,200)", "is too large");
    expect_error_contains(interp, "tensorops_decompose_cp(cp_big,T,2000)", "is too large");
    // A rank anyone would factor at still runs, and still creates its handle.
    expect_contains(interp, "tensorops_decompose_nmf(nmf_ok,T,3)", "created NMFDecomposition");
    expect_contains(interp, "tensorops_decompose_cp(cp_ok,T,3)", "created CPDecomposition");
    // Tucker is deliberately not bounded this way: its ALS converges out of max_iter
    // rather than running it, so 1e7 iterations of a 40x40 returns in 0.02 s and the
    // mode-dimension ceiling it already has is the right one.
    expect_ok(interp, "S = ones(40,40)");
    expect_contains(interp, "tensorops_decompose_tucker(tk_ok,S,[4,4],10000000,1e-30)",
                    "created TuckerDecomposition");
}

TEST(ReplSuperlinearArguments, AnImageFilterIsBoundedOnItsKernelTimesTheImage) {
    // Every one of these visits each pixel once per kernel cell, so the cost is the
    // image times the kernel -- and for the morphology and median filters, times its
    // SQUARE. Measured on a 256x256: medfilt2 at ksize 21 takes 1.73 s, so the
    // medfilt2(ones(512,512), 999) the audit proposed is 2.6e11 pixel-cells, about four
    // hours. None of them was reachable by the oversized-argument sweep, whose values
    // the argument guard rejects before any of this.
    Interpreter interp;
    expect_ok(interp, "A = ones(512,512)");
    for (const auto* call : {"medfilt2(A, 999)", "boxfilter(A, 999999)",
                             "imdilate(A, 4999)", "imerode(A, 4999)", "imopen(A, 4999)",
                             "imclose(A, 4999)", "imtophat(A, 4999)", "imbothat(A, 4999)",
                             "imgradient_morph(A, 4999)"}) {
        expect_error_contains(interp, call, "is too large");
    }
    // sigma is not a tuning knob on the cost: the kernel half-width is a multiple of it,
    // so what sigma names is the kernel. The bound is stated on the kernel, because the
    // kernel is the thing that has to fit.
    for (const auto* call : {"imgaussfilt(A, 100000)", "laplacian_of_gaussian(A, 100000)",
                             "bilateral(A, 1000, 1)"}) {
        expect_error_contains(interp, call, "the kernel sigma implies");
    }
    // And every one of them still filters at a width anyone would use.
    expect_ok(interp, "S = ones(64,64)");
    expect_ok(interp, "medfilt2(S, 5)");
    expect_ok(interp, "boxfilter(S, 5)");
    expect_ok(interp, "imdilate(S, 3)");
    expect_ok(interp, "imopen(S, 3)");
    expect_ok(interp, "imgaussfilt(S, 2)");
    expect_ok(interp, "laplacian_of_gaussian(S, 2)");
    expect_ok(interp, "bilateral(S, 3, 1)");
}

TEST(ReplSuperlinearArguments, AStepCountCanBeARatioRatherThanAnArgument) {
    // The CFD advection family takes no step count at all. It takes `t_end` and `dt`,
    // and the number of sweeps is ceil(t_end/dt) -- so neither argument looks like a
    // size and their QUOTIENT is one. `cfd_advection1d(1000, 1.0, 1.0, 1e-9)` is a
    // billion sweeps of a thousand cells, with one whole grid retained per step.
    //
    // No per-argument guard can see this: 1.0 and 1e-9 are both unremarkable. The bound
    // is on the quotient, which is the only place the size actually appears.
    Interpreter interp;
    for (const auto* call : {"cfd_advection1d(1000, 1.0, 1.0, 1e-9)",
                             "cfd_advection2d(100, 100, 1.0, 0.0, 1.0, 1e-7)",
                             // The 3-D form is written with a target because at eight
                             // arguments it has no no-target spelling -- the same gap the
                             // "ninety-eight callees had no no-target form" entry covers,
                             // and not something this guard introduced.
                             "u = cfd_advection3d(30, 30, 30, 1.0, 0.0, 0.0, 1.0, 1e-6)"}) {
        expect_error_contains(interp, call, "the step count t_end/dt implies");
    }
    // The same three with a step size anyone would integrate at still run.
    expect_ok(interp, "cfd_advection1d(100, 1.0, 0.1, 0.001)");
    expect_ok(interp, "cfd_advection2d(20, 20, 1.0, 0.0, 0.1, 0.001)");
    expect_ok(interp, "u = cfd_advection3d(8, 8, 8, 1.0, 0.0, 0.0, 0.1, 0.005)");
    // And the grid-taking forms, whose cell count comes from the grid rather than from
    // an argument, are bounded by the same quotient.
    expect_ok(interp, "G = cfd_grid1d(0, 1, 100)");
    expect_ok(interp, "U = cfd_square_pulse(G, 0.35, 0.1)");
    expect_error_contains(interp, "cfd_run_advection(G, U, 1.0, 1.0, 1e-9)",
                          "the step count t_end/dt implies");
    expect_ok(interp, "cfd_run_advection(G, U, 1.0, 0.1, 0.001)");
}

TEST(ReplSuperlinearArguments, WorkForAnAnswerThatCouldNeverBeShown) {
    // The REPL's scalar is a double, and `bigint_to_scalar` requires the exact BigInt to
    // round-trip through one -- so 21! already fails, and so does fib(79). What the
    // bignum commands did with a large argument was compute the exact answer first and
    // refuse it afterwards: `bigint_fib(200000)` spent 15.3 s building a number it then
    // declined to print.
    //
    // The bounds here sit far ABOVE where that round trip stops succeeding, so they
    // refuse nothing that could have worked. All they do is stop the computing.
    Interpreter interp;
    for (const auto* call : {"bigint_factorial(2000000000)", "bigint_fib(2000000000)",
                             "bigint_pow(10, 1000000000)"}) {
        expect_error_contains(interp, call, "is too large");
    }
    // Everything inside the range a double can hold still answers, exactly.
    expect_contains(interp, "bigint_factorial(20)", "2.43290200817664e+18");
    expect_contains(interp, "bigint_fib(70)", "190392490709135");
    expect_contains(interp, "bigint_pow(2, 40)", "1099511627776");
}

TEST(ReplSuperlinearArguments, ASchmidtDecompositionIsCubicInTheSubsystemDimension) {
    // The Gram matrix is dim_a by dim_a and the Jacobi sweep over it is cubic: measured
    // 0.65 s at dim_a = 1024, so dim_a = 4096 is 6.9e10 units and 82 s. Ten qubits
    // (1024) is an ordinary subsystem to decompose and stays inside the bound; twelve
    // does not.
    Interpreter interp;
    expect_ok(interp, "P = ones(4096,1)");
    for (const auto* call : {"quantum_schmidt_rank(P, 4096, 1)",
                             "quantum_schmidt_number(P, 4096, 1)",
                             "quantum_entanglement_entropy(P, 4096, 1)",
                             "quantum_schmidt_decomposition(P, 4096, 1)",
                             "quantum_schmidt_bases(P, 4096, 1)"}) {
        expect_error_contains(interp, call, "dim_a 4096 is too large");
    }
    expect_ok(interp, "Q = ones(1024,1)");
    expect_ok(interp, "quantum_schmidt_rank(Q, 1024, 1)");
}

TEST(ReplSuperlinearArguments, AGuardCanBeWrongByASquareRoot) {
    // The guard that bounded `mathieu_a(0, 1e18)` was written with Mathieu's own sizing
    // rule baked into it -- the characteristic matrix is
    // `max(24, n + 16 + ceil(sqrt(|q|)))` -- and then applied to the spheroidal family
    // as well, whose rule is `(n - m)/2 + 22 + ceil(|c|)`. Linear, not a square root.
    //
    // So `c = 1e10` read as a dimension of 1e5, passed, and asked for a ten-billion
    // entry tridiagonal. All three spheroidal commands were measured ABORTING the
    // process on it, which is the failure the guard existed to prevent.
    Interpreter interp;
    for (const auto* call : {"spheroidal_lambda(0, 0, 10000000000)",
                             "spheroidal_s1(0, 0, 10000000000, 0.5)",
                             "spheroidal_s2(0, 0, 10000000000, 0.5)"}) {
        expect_error_contains(interp, call, "size an internal matrix");
    }
    // Neither family bounded the ORDER either, and the order is a dimension too:
    // `spheroidal_lambda(1e7, 0, 0)` was still running at 25 s, and `mathieu_a(1e7, 1)`
    // built a ten-million-entry tridiagonal in 1.9 s.
    expect_error_contains(interp, "spheroidal_lambda(10000000, 0, 0)",
                          "size an internal matrix");
    expect_error_contains(interp, "mathieu_a(10000000, 1)", "size an internal matrix");
    expect_error_contains(interp, "mathieu_mc(0, 1e30, 0.5)", "size an internal matrix");
    // The values these commands are actually for are unaffected, and still right:
    // a_1(5) = 1.858188 and lambda_{2,0}(c=1) = 6.533472.
    expect_contains(interp, "mathieu_a(1, 5)", "1.858188");
    expect_contains(interp, "spheroidal_lambda(2, 0, 1)", "6.533472");
}

TEST(ReplSuperlinearArguments, AFixedStepSolverIsBoundedByTheTrajectoryItKeeps) {
    // These keep every step and the REPL prints the lot as a steps+1 by 2 matrix, so a
    // step count past half the matrix cap is work for an answer that could never be
    // shown: 200000 steps integrated for 0.2 s and were then refused for being 400002
    // elements. At 1e8 -- which the linear cap admits -- that is two minutes to reach
    // the same refusal.
    Interpreter interp;
    for (const auto* call : {"ode_backward_euler(\"-50*y\", 0, 1, 1, 100000000)",
                             "ode_bdf2(\"-50*y\", 0, 1, 1, 100000000)",
                             "ode_trapezoidal(\"-50*y\", 0, 1, 1, 100000000)",
                             "ode_rosenbrock23(\"-50*y\", 0, 1, 1, 100000000)",
                             "ode_euler(\"-50*y\", 0, 1, 1, 100000000)"}) {
        expect_error_contains(interp, call, "steps 100000000 is too large");
        expect_error_contains(interp, call, "one row per step");
    }
    // `ode_exponential_euler` takes the decay rate as a second argument, so its step
    // count sits one place further along; the guard is the same one.
    expect_error_contains(interp,
                          "ode_exponential_euler(\"-50*y\", -50, 0, 1, 1, 100000000)",
                          "steps 100000000 is too large");
    // A step count whose trajectory fits still integrates.
    expect_ok(interp, "ode_backward_euler(\"-50*y\", 0, 1, 1, 100)");
    expect_ok(interp, "ode_rosenbrock23(\"-50*y\", 0, 1, 1, 100)");
}

TEST(ReplSuperlinearArguments, AnAccumulatorCapIsNotAWorkCap) {
    // `hough_lines` and `hough_circles` already bounded their accumulators, and both
    // probes below are INSIDE those bounds. What neither bounded is the voting.
    //
    //   - every edge pixel votes once per ANGLE, so 262144 angles over a 512x512 image
    //     is 6.9e10 votes and a 262144 x 1 accumulator, which fits exactly;
    //   - every edge pixel walks the CIRCUMFERENCE of each candidate radius, so a
    //     single plane at r = 3e8 is one cell per pixel and 4.9e17 votes.
    Interpreter interp;
    expect_ok(interp, "A = ones(512,512)");
    expect_error_contains(interp, "hough_lines(A, 0.5, 262144, 1, 1)",
                          "n_theta 262144 is too large");
    expect_error_contains(interp, "hough_circles(A, 300000000, 300000000)",
                          "the votes each edge pixel casts");
    // The resolutions anyone would use are unaffected: 180 angles is the textbook
    // Hough transform, and a radius range of a few tens is what a 64x64 image holds.
    expect_ok(interp, "B = ones(256,256)");
    expect_ok(interp, "hough_lines(B, 0.5, 180, 180, 1)");
    expect_ok(interp, "C = ones(64,64)");
    expect_ok(interp, "hough_circles(C, 5, 20)");
}

TEST(ReplSuperlinearArguments, ASimplicialComplexIsBoundedByItsOwnRowCount) {
    // `topo_vietoris_rips(zeros(200,200), 1, 2)` spent twenty seconds enumerating 1.3
    // million simplices and was then refused by the matrix cap -- the same shape as the
    // bignum commands, work for an answer that could never be shown. The rows ARE the
    // simplices, so the count that fits is the bound, and moving that verdict to the
    // front of the command costs the twenty seconds nothing.
    Interpreter interp;
    expect_ok(interp, "D = zeros(512,512)");
    expect_error_contains(interp, "topo_vietoris_rips(D, 1, 3)", "can enumerate");
    expect_error_contains(interp, "topo_cech_complex(D, 1, 2)", "can enumerate");
    // A complex whose simplices fit is built.
    expect_ok(interp, "E = zeros(40,40)");
    expect_ok(interp, "topo_vietoris_rips(E, 1, 2)");

    // `topo_betti_curve` is the one where bounding the result bounds nothing: its answer
    // is one row of Betti numbers per threshold. What it costs is a complex rebuilt AND
    // REDUCED at each of them, and the reduction is quadratic in the simplex count.
    expect_ok(interp, "F = zeros(60,60)");
    expect_ok(interp, "T = zeros(1000,1)");
    expect_error_contains(interp, "C = topo_betti_curve(F, T, 2)", "is too large");
    expect_ok(interp, "T4 = zeros(4,1)");
    expect_ok(interp, "C = topo_betti_curve(E, T4, 1)");
}

TEST(ReplSuperlinearArguments, ASignalCommandCostsTheSignalTimesItsWindow) {
    // Four shapes, one family. Measured on this machine:
    //
    //   signal_median_filter   one selection per sample per window cell   4 ns
    //   signal_lms             every tap touched twice per sample         9 ns
    //   signal_savgol          (window + polyorder) * polyorder^2         6 ns
    //   signal_cheby1          the pole product expanded, order^2        16 ns
    //
    // `signal_savgol` is the one worth naming: its CONVOLUTION is FFT-based and does not
    // grow with the window at all -- window 11 and window 1001 over 200000 samples both
    // take 0.3 s -- so bounding the window would have bounded the wrong thing. What grows
    // is the coefficient solve, and polyorder is what drives it.
    Interpreter interp;
    expect_ok(interp, "x = ones(262144,1)");
    expect_error_contains(interp, "signal_median_filter(x, 174763)",
                          "window_length 174763 is too large");
    expect_error_contains(interp, "signal_lms(x, x, 262144, 0.001)",
                          "filter_length 262144 is too large");
    expect_ok(interp, "s = ones(5001,1)");
    expect_error_contains(interp, "signal_savgol(s, 5001, 5000)",
                          "polyorder 5000 is too large");
    expect_error_contains(interp, "signal_cheby1(1000000, 1, 100, 1000)",
                          "order 1000000 is too large");
    // A Savitzky-Golay smoother is a quartic over a hundred-odd samples; an IIR filter is
    // sixth order; an LMS filter has a few dozen taps. None of that is refused.
    expect_ok(interp, "y = ones(20000,1)");
    expect_ok(interp, "signal_savgol(y, 101, 4)");
    expect_ok(interp, "signal_median_filter(y, 51)");
    expect_ok(interp, "signal_lms(y, y, 32, 0.001)");
    expect_ok(interp, "signal_cheby1(6, 1, 100, 1000)");
}

TEST(ReplSuperlinearArguments, AResultTooWideToShowIsStillPaidForFirst) {
    // Two more of the shape the bignum commands had. `signal_czt_zoom` returns one
    // (real, imaginary) row per output bin and `poly_pow` returns one row per
    // coefficient, so in both cases an argument past half the matrix cap buys a
    // computation whose answer the REPL will then refuse to print.
    Interpreter interp;
    expect_ok(interp, "x = ones(8,1)");
    expect_error_contains(interp, "signal_czt_zoom(x, 0, 100, 10000000, 1000)",
                          "m 10000000 is too large");
    expect_error_contains(interp, "poly_pow([0;1], 10000000)",
                          "coefficient count");
    expect_ok(interp, "signal_czt_zoom(x, 0, 100, 1024, 1000)");
    expect_ok(interp, "poly_pow([0;1;1], 100)");
}

TEST(ReplSuperlinearArguments, TheWorkIsBoundedEvenWhenTheAnswerIsSmall) {
    // Each of these returns almost nothing and spends a great deal to get there.
    //
    //   - `poly_fit` returns degree+1 coefficients and accumulates normal equations in
    //     nodes * degree^2 and solves them in degree^3: three points at degree 3000 is
    //     2.7e10 units.
    //   - `geo_bspline_eval` returns ONE point and evaluates a triangle of degree^2
    //     affine combinations to find it.
    //   - `combo_unrank_combination` returns k numbers and walks the candidates
    //     subtracting binomials to reach them; a rank near the top of the range walks
    //     nearly all n, and n = 2147483647 is about forty seconds of subtracting. The
    //     bound has to be that worst case, because which ranks are cheap is not
    //     something the caller can be expected to know.
    //   - `crypto_pbkdf2_sha256` returns 32 bytes. It is slow on purpose, which is why
    //     the iteration count needs a bound rather than an exemption from one.
    Interpreter interp;
    expect_error_contains(interp, "poly_fit([1;2;3], [1;4;9], 3000)",
                          "degree 3000 is too large");
    expect_ok(interp, "ctrl = ones(131072,2)");
    expect_ok(interp, "knots = ones(262144,1)");
    expect_error_contains(interp, "geo_bspline_eval(ctrl, knots, 131071, 0.5)",
                          "degree 131071 is too large");
    expect_error_contains(interp,
                          "combo_unrank_combination(2147483647, 2, 2305843005992468480)",
                          "n 2147483647 is too large");
    for (const auto* call : {"crypto_pbkdf2_sha256(00112233, 44556677, 4294967295, 32)",
                             "crypto_pbkdf2_hmac_sha512(00112233, 44556677, 4294967295, 64)"}) {
        expect_error_contains(interp, call, "iteration count 4294967295 is too large");
    }
    expect_ok(interp, "poly_fit([1;2;3], [1;4;9], 2)");
    expect_ok(interp, "c2 = ones(10,2)");
    expect_ok(interp, "k2 = ones(20,1)");
    expect_ok(interp, "geo_bspline_eval(c2, k2, 3, 0.5)");
    expect_ok(interp, "combo_unrank_combination(10, 3, 5)");
    expect_ok(interp, "crypto_pbkdf2_sha256(00112233, 44556677, 10000, 32)");
}

TEST(ReplSuperlinearArguments, APopulationTimesAGenerationCountIsAProduct) {
    // Neither factor looks wrong on its own and the linear cap bounds each at 1e7, which
    // says nothing about the 1e14 evaluations the two of them together ask for. The
    // formula is evaluated once per member per generation over a vector as wide as the
    // bounds list, so the budget charges the dimension first and the bound on the two
    // counts shrinks as the search space grows.
    //
    // Both are budgeted as a REQUEST rather than a slip, the way a Monte Carlo path count
    // is: a longer run is a better answer, and that is what the argument is for.
    Interpreter interp;
    expect_error_contains(interp,
                          "differential_evolution(\"x0*x0\", [[-5, 5]], 1000000, 0.8, 0.9, 1000000)",
                          "is too large");
    expect_error_contains(interp,
                          "particle_swarm(\"x0*x0\", [[-5, 5]], 1000000, 1000000)",
                          "is too large");
    expect_ok(interp, "differential_evolution(\"x0*x0\", [[-5, 5]], 20, 0.8, 0.9, 200)");
    expect_ok(interp, "particle_swarm(\"x0*x0\", [[-5, 5]], 20, 200)");
}

TEST(ReplSuperlinearArguments, LandmarkSelectionRescansForEveryLandmark) {
    // maxmin picks each landmark by rescanning every point against every landmark chosen
    // so far, so the landmark count enters the product TWICE: 2000 points and 300
    // landmarks took 2.7 s, and 600 took 9.7 s -- four times for double, which is the
    // signature of a square and not of the linear argument it looks like.
    Interpreter interp;
    expect_ok(interp, "P = zeros(4000,2)");
    expect_error_contains(interp, "topo_select_landmarks(P, 4000)", "n 4000 is too large");
    expect_ok(interp, "Q = zeros(500,2)");
    expect_ok(interp, "topo_select_landmarks(Q, 40)");
}

TEST(ReplSuperlinearArguments, ACapOnTheGridIsNotACapOnTheSystemItAssembles) {
    // Three commands assemble a DENSE square system and none of them takes its size as an
    // argument, so nothing in what the user typed looks like a size at all.
    //
    //   - `stats_pacf` builds the Yule-Walker table at (max_lag+1)^2, so max_lag 100000
    //     asks for 80 GB -- and the series it was asked about has five elements in it.
    //     Measured ABORTING the process at 2.3 s.
    //   - `stats_arfit` builds a p by p Toeplitz system the same way.
    //   - `pde_helmholtz_2d` assembles the five-point stencil densely, one row per
    //     interior point: an ordinary 100 by 100 grid is a 9604-unknown system, 738 MB of
    //     coefficients and 8.9e11 operations. It did not finish in 25 s.
    Interpreter interp;
    expect_ok(interp, "x = ones(2000,1)");
    expect_error_contains(interp, "stats_pacf(x, 100000)", "dense");
    expect_error_contains(interp, "stats_arfit(x, 20000)", "dense");
    expect_ok(interp, "G = ones(100,100)");
    expect_error_contains(interp, "pde_helmholtz_2d(G, 1.0, 0.01, 0.01)", "dense");
    // The sizes these are actually used at are unaffected.
    expect_ok(interp, "stats_pacf(x, 40)");
    expect_ok(interp, "stats_arfit(x, 20)");
    expect_ok(interp, "H = ones(20,20)");
    expect_ok(interp, "pde_helmholtz_2d(H, 1.0, 0.01, 0.01)");
}

TEST(ReplSuperlinearArguments, AModelCanBeBiggerThanTheDataItWasFittedTo) {
    // `ml_gmm_fit` returns 2K+2 rows by max(features, K, 3) columns, so the component
    // count enters the answer TWICE: 100000 components is a 200002 by 100000 model, 160
    // GB, and it ABORTED the process in 2.0 s. A component also needs a point to be a
    // mixture of, which is the other half of the bound.
    //
    // `ml_pca_fit` is the milder version of the same thing: its model is n_components+1
    // rows of one weight per feature, and a 512 by 512 matrix at the full 512 components
    // is 262656 -- just past the cap. It used to fit for 3.5 s and be refused afterwards.
    Interpreter interp;
    expect_ok(interp, "X = ones(1000,1)");
    expect_error_contains(interp, "ml_gmm_fit(X, 100000)", "expected 1 <= n_components");
    expect_ok(interp, "W = ones(512,512)");
    expect_error_contains(interp, "ml_pca_fit(W, 512)", "model, which is limited to");
    expect_ok(interp, "ml_gmm_fit(X, 5)");
    expect_ok(interp, "V = ones(200,50)");
    expect_ok(interp, "ml_pca_fit(V, 10)");
}

TEST(ReplSuperlinearArguments, AnMLFitCostsTheDataTimesTheThingBeingFitted) {
    // Six more products, each measured on a run that completes:
    //
    //   ml_kmeans_fit          rows x k^2 (more centres, more Lloyd passes)      15 ns
    //   ml_spectral_clustering rows^3 for the embedding, rows x k^2 after   8 / 350 ns
    //   ml_tsne_fit            rows x n_iter                                  25 us
    //   stats_bootstrap_mean   length x n_boot                                 23 ns
    //   stats_kde              samples x grid                                  14 ns
    //   lz77_encode_vec        bytes x window                                   6 ns
    //
    // `ml_kmeans_fit` and `ml_pca_fit` already bounded k by the data -- k clusters need k
    // points -- and that bound is not the cost: 3000 points at k = 3000 satisfies it and
    // did not finish in 25 s.
    Interpreter interp;
    expect_ok(interp, "X = ones(3000,2)");
    expect_error_contains(interp, "ml_kmeans_fit(X, 3000)", "k 3000 is too large");
    expect_ok(interp, "Y = ones(400,2)");
    expect_error_contains(interp, "ml_spectral_clustering(Y, 400)", "k 400 is too large");
    expect_ok(interp, "Z = ones(200,2)");
    expect_error_contains(interp, "ml_tsne_fit(Z, 30, 2000000000)",
                          "n_iter 2000000000 is too large");
    expect_ok(interp, "v = ones(1000,1)");
    expect_error_contains(interp, "m = stats_bootstrap_mean(v, 10000000)",
                          "n_boot 10000000 is too large");
    expect_ok(interp, "s = ones(100000,1)");
    expect_error_contains(interp, "stats_kde(s, s, 1)", "the grid length 100000 is too large");
    expect_ok(interp, "M = ones(512,512)");
    expect_error_contains(interp, "lz77_encode_vec(M, 262144, 1)", "window 262144 is too large");
    // And the sizes these are used at:
    expect_ok(interp, "ml_kmeans_fit(X, 5)");
    expect_ok(interp, "ml_spectral_clustering(Y, 4)");
    expect_ok(interp, "ml_tsne_fit(Z, 30, 250)");
    expect_ok(interp, "m = stats_bootstrap_mean(v, 10000)");
    expect_ok(interp, "g = ones(500,1)");
    expect_ok(interp, "stats_kde(v, g, 1)");
    expect_ok(interp, "N = ones(128,128)");
    expect_ok(interp, "lz77_encode_vec(N, 1024, 15)");
}

TEST(ReplSuperlinearArguments, APerArgumentCeilingIsNotAProductCeiling) {
    // The four ensemble commands already refuse more than 10000 learners and
    // `ml_isolation_forest_fit` also refuses a subsample past a million. Both are
    // per-argument, and what it costs is the two multiplied: 10000 trees over a 10000-row
    // subsample is 1e8 sampled rows and took 6.4 s, with each argument inside its own
    // maximum. `ml_random_forest_fit` was 4.3 s the same way.
    Interpreter interp;
    expect_ok(interp, "X = ones(10000,1)");
    expect_error_contains(interp, "ml_isolation_forest_fit(X, 10000, 10000)",
                          "n_trees 10000 is too large");
    expect_ok(interp, "S = ones(1000,1)");
    expect_ok(interp, "y = ones(1000,1)");
    expect_error_contains(interp, "ml_random_forest_fit(S, y, 10000, 5)",
                          "n_trees 10000 is too large");
    expect_ok(interp, "ml_isolation_forest_fit(X, 100, 256)");
    expect_ok(interp, "ml_random_forest_fit(S, y, 100, 5)");
}

TEST(ReplSuperlinearArguments, ASolverWhoseProbeConvergedWasMissedByTheLastPass) {
    // `pde_poisson_2d` has the same shape as the ten PDE solvers bounded in the previous
    // pass -- one relaxation sweep of the whole grid per iteration -- and it was missed
    // because its probe CONVERGED and came back in 1.66 s. Given a tolerance it cannot
    // reach, it runs the full count: 200x200 for 1e7 iterations is 4e11 cell-sweeps.
    //
    // `poly_cheb_expand` is the other one this pass found only by hand: the sweep wrote
    // its probe in a form the REPL could not parse, so the sweep's own record of it said
    // "parse_matrix: expected [ ... ]" rather than anything about the command. Every one
    // of the n+1 coefficients sums over all n+1 nodes, and at n = 2147483647 the samples
    // alone are 17 GB -- measured ABORTING the process before any summing began.
    Interpreter interp;
    expect_ok(interp, "G = ones(200,200)");
    expect_error_contains(interp, "pde_poisson_2d(G, 0.01, 0.01, 10000000, 1e-300)",
                          "max_iterations 10000000 is too large");
    expect_error_contains(interp, "poly_cheb_expand([1;2;3], 10000000)",
                          "n 10000000 is too large");
    expect_error_contains(interp, "c = poly_cheb_expand([1;2;3], 2147483647, -1, 1)",
                          "n 2147483647 is too large");
    expect_ok(interp, "P = ones(60,60)");
    expect_ok(interp, "pde_poisson_2d(P, 0.01, 0.01, 5000, 1e-8)");
    expect_ok(interp, "poly_cheb_expand([1;2;3], 40)");
}

TEST(ReplSuperlinearArguments, SomeCommandsHaveNoArgumentToBoundAtAll) {
    // The last shape in the sweep, and the awkward one: three commands whose cost is set
    // entirely by the size of the data handed to them, with nothing in the call that
    // could be called a size argument.
    //
    //   - `ml_svm_fit` evaluates the kernel between every pair on every SMO pass: 400
    //     rows took 3.5 s and 800 took 13.9 s.
    //   - `ml_agglomerative_fit` rescans the whole distance table at every merge: 300
    //     rows took 0.7 s, 600 took 5.9 s, 900 took 21.3 s -- 8x and 27x, a cube.
    //   - `stats_multiple_regression` forms X^T X, which is columns by columns and is
    //     then copied for the elimination. `ones(1, 262144)` is a design matrix of
    //     exactly the element cap -- inside every bound the REPL has -- and asks for 1.1
    //     TB. Measured ABORTING the process at 2.6 s.
    //
    // A matrix that fits is not a normal-equations system that fits.
    Interpreter interp;
    expect_ok(interp, "X = ones(2000,2)");
    expect_ok(interp, "y = ones(2000,1)");
    expect_error_contains(interp, "ml_svm_fit(X, y)", "the row count 2000 is too large");
    expect_error_contains(interp, "ml_agglomerative_fit(X, 3)",
                          "the row count 2000 is too large");
    expect_ok(interp, "W = ones(1,262144)");
    expect_ok(interp, "z = ones(1,1)");
    expect_error_contains(interp, "stats_multiple_regression(W, z)", "dense");
    // The sizes these are used at are unaffected, including a square design matrix at
    // the widest the normal equations allow.
    expect_ok(interp, "S = ones(300,2)");
    expect_ok(interp, "t = ones(300,1)");
    expect_ok(interp, "ml_svm_fit(S, t)");
    expect_ok(interp, "ml_agglomerative_fit(S, 3)");
    expect_ok(interp, "D = ones(512,512)");
    expect_ok(interp, "d = ones(512,1)");
    expect_ok(interp, "stats_multiple_regression(D, d)");
}

TEST(ReplSuperlinearArguments, AndOneOfThemWasFixedRatherThanBounded) {
    // `stats_kendall` was the fourth of that set and it did not get a bound. Its pair
    // loop is 5e9 comparisons at 100000 observations -- 3.5 s at n = 20000 and 13.9 s at
    // n = 40000 -- and Kendall's tau-b is an identity away from an inversion count, so
    // Knight's O(n log n) form reaches the same number in 0.2 s. Refusing an ordinary
    // statistic on an ordinary dataset would have been the wrong answer to it.
    //
    // `test_stats_timeseries` checks the new form against the old one exactly, over four
    // tie regimes. This asserts the thing the REPL user sees: it comes back, and it is
    // right. tau of a strictly increasing pair is 1.
    Interpreter interp;
    expect_ok(interp, "n = 100000");
    expect_ok(interp, "a = ones(100000,1)");
    expect_contains(interp, "stats_kendall([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])", "1");
    expect_ok(interp, "stats_kendall(a, a)");
}
