// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

// Reproducibility manifest: what this run actually did.
//
// A numerical result is only defensible if the conditions that produced it can be
// stated. Three things in this library change results without changing the source:
// the ISA path chosen at startup (a vectorised reduction sums in a different order
// from a scalar one), the number of worker threads (which changes how a parallel
// reduction is partitioned), and the seed behind any Monte Carlo routine. None of
// them was recorded anywhere, so "run it again on the other machine and see" was
// the only way to find out whether a difference was a bug or a dispatch decision.
//
//     ms::runtime::to_text(ms::runtime::capture())
//
// This is the minimum viable form of a full bit-reproducibility mode -- it reports
// the conditions rather than pinning them -- and it is the part that is nearly
// free, because every value it needs is already computed somewhere.

#include <cstdint>
#include <string>

namespace ms::runtime {

struct ReproManifest {
    std::string version;
    std::string commit;
    std::string build_date;

    /// The ISA the library will actually use, after the OS register-state check
    /// in ms::simd::detect_isa -- not what CPUID advertises.
    std::string isa;
    /// The value of MS_FORCE_ISA if a ceiling was applied, otherwise empty. A run
    /// under a forced ceiling is not comparable with one that was not.
    std::string forced_isa;
    std::string kernel;
    std::size_t batch_width = 1;

    unsigned hardware_threads = 0;
    std::size_t max_workers = 0;

    std::uint64_t seed = 0;
    /// False when `seed` is the built-in default rather than one the caller
    /// chose. The distinction matters: an unset seed is reproducible on this
    /// build and says nothing about any other.
    bool seed_explicit = false;
};

/// The global seed reported by the manifest.
///
/// This does not reach into the individual routines -- they still take their own
/// seed arguments, and that is the documented contract in docs/API.md. What it
/// provides is one number to record and one number to quote when asking for a run
/// to be repeated.
void set_global_seed(std::uint64_t seed);
std::uint64_t global_seed();
bool global_seed_is_explicit();
/// Return to the built-in default, as if set_global_seed had never been called.
void clear_global_seed();

ReproManifest capture();

/// Human-readable, one field per line.
std::string to_text(const ReproManifest& m);

/// Machine-readable, for embedding in a result file or an audit record.
std::string to_json(const ReproManifest& m);

} // namespace ms::runtime
