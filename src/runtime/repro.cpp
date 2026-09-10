// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/runtime/repro.hpp"

#include "ms/runtime/thread_pool.hpp"
#include "ms/simd/simd.hpp"
#include "ms/version.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>

namespace ms::runtime {

namespace {

// The default is a constant rather than something drawn from the clock. A seed
// that varies per run would make every run unreproducible by default, which is the
// opposite of what this file is for; a caller who wants variation says so.
constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ULL;

std::atomic<std::uint64_t>& seed_slot() {
    static std::atomic<std::uint64_t> value{kDefaultSeed};
    return value;
}

std::atomic<bool>& seed_explicit_slot() {
    static std::atomic<bool> value{false};
    return value;
}

const char* kernel_name(ms::simd::Kernel k) {
    switch (k) {
    case ms::simd::Kernel::Avx512:
        return "avx512";
    case ms::simd::Kernel::Avx2:
        return "avx2";
    case ms::simd::Kernel::Scalar:
        break;
    }
    return "scalar";
}

std::string trimmed(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

/// Minimal JSON string escaping. The values here are version strings, ISA names
/// and a commit hash, so the character set is narrow -- but a build date format or
/// a future field could carry a quote, and a manifest that produces invalid JSON
/// is worse than no manifest.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            } else {
                out += c;
            }
            break;
        }
    }
    return out;
}

} // namespace

void set_global_seed(std::uint64_t seed) {
    seed_slot().store(seed, std::memory_order_relaxed);
    seed_explicit_slot().store(true, std::memory_order_relaxed);
}

std::uint64_t global_seed() {
    return seed_slot().load(std::memory_order_relaxed);
}

bool global_seed_is_explicit() {
    return seed_explicit_slot().load(std::memory_order_relaxed);
}

void clear_global_seed() {
    seed_slot().store(kDefaultSeed, std::memory_order_relaxed);
    seed_explicit_slot().store(false, std::memory_order_relaxed);
}

ReproManifest capture() {
    ReproManifest m;
    m.version = std::string(ms::VERSION_STRING);
    m.commit = std::string(ms::BUILD_COMMIT);
    m.build_date = std::string(ms::BUILD_DATE);

    const auto info = ms::simd::dispatch_info();
    m.isa = trimmed(ms::simd::isa_summary(info.isa));
    m.kernel = kernel_name(info.active);
    m.batch_width = info.batch_width;

    if (const char* const forced = std::getenv("MS_FORCE_ISA")) {
        m.forced_isa = forced;
    }

    m.hardware_threads = std::thread::hardware_concurrency();
    m.max_workers = ms::ThreadPoolConfig{}.max_workers;

    m.seed = global_seed();
    m.seed_explicit = global_seed_is_explicit();
    return m;
}

std::string to_text(const ReproManifest& m) {
    std::ostringstream out;
    out << "version:           " << m.version << "\n"
        << "commit:            " << m.commit << "\n"
        << "built:             " << m.build_date << "\n"
        << "isa:               " << m.isa << "\n"
        << "forced_isa:        " << (m.forced_isa.empty() ? "(unset)" : m.forced_isa) << "\n"
        << "kernel:            " << m.kernel << "\n"
        << "batch_width:       " << m.batch_width << "\n"
        << "hardware_threads:  " << m.hardware_threads << "\n"
        << "max_workers:       " << m.max_workers << "\n"
        << "seed:              " << m.seed
        << (m.seed_explicit ? " (set)" : " (default)") << "\n";
    return out.str();
}

std::string to_json(const ReproManifest& m) {
    std::ostringstream out;
    out << "{\"version\":\"" << json_escape(m.version)
        << "\",\"commit\":\"" << json_escape(m.commit)
        << "\",\"built\":\"" << json_escape(m.build_date)
        << "\",\"isa\":\"" << json_escape(m.isa)
        << "\",\"forced_isa\":\"" << json_escape(m.forced_isa)
        << "\",\"kernel\":\"" << json_escape(m.kernel)
        << "\",\"batch_width\":" << m.batch_width
        << ",\"hardware_threads\":" << m.hardware_threads
        << ",\"max_workers\":" << m.max_workers
        << ",\"seed\":" << m.seed
        << ",\"seed_explicit\":" << (m.seed_explicit ? "true" : "false")
        << "}";
    return out.str();
}

} // namespace ms::runtime
