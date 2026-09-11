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
