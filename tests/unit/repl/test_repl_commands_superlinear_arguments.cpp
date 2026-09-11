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
