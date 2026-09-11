// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Replays a fuzz target's checked-in corpus, plus a bounded, deterministic set of
// mutations of it, through LLVMFuzzerTestOneInput.
//
// The seven fuzz targets in this directory only run under the 24-hour libFuzzer job, and
// libFuzzer's runtime is not installed on every developer machine. Their corpora are
// checked in, though, and a corpus entry is exactly a previously interesting input -- so
// replaying it costs milliseconds and turns the corpus into an ordinary regression net.
// The mutation pass on top is deterministic (a fixed seed), so a failure here is
// reproducible rather than a flake.
//
// A crash inside LLVMFuzzerTestOneInput takes the process down, which CTest reports as a
// failed test with the sanitizer's output attached -- exactly what is wanted.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

namespace {

std::vector<std::vector<uint8_t>> load_corpus(const std::string& dir) {
    std::vector<std::vector<uint8_t>> out;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().filename() == "README.md") continue;
        std::ifstream in(entry.path(), std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
        if (!bytes.empty()) out.push_back(std::move(bytes));
    }
    return out;
}

void mutate(std::vector<uint8_t>& buf, std::mt19937& rng,
            const std::vector<std::vector<uint8_t>>& corpus) {
    const int rounds = 1 + static_cast<int>(rng() % 6);
    for (int i = 0; i < rounds; ++i) {
        switch (rng() % 5) {
            case 0:
                if (!buf.empty()) buf[rng() % buf.size()] = static_cast<uint8_t>(rng() & 0xFF);
                break;
            case 1:
                buf.insert(buf.begin() + static_cast<long>(rng() % (buf.size() + 1)),
                           static_cast<uint8_t>(rng() & 0xFF));
                break;
            case 2:
                if (!buf.empty()) buf.erase(buf.begin() + static_cast<long>(rng() % buf.size()));
                break;
            case 3: {
                const auto& other = corpus[rng() % corpus.size()];
                if (!other.empty()) {
                    const std::size_t cut = rng() % (buf.size() + 1);
                    const std::size_t take = 1 + rng() % other.size();
                    buf.insert(buf.begin() + static_cast<long>(cut), other.begin(),
                               other.begin() + static_cast<long>(take));
                }
                break;
            }
            default:
                if (!buf.empty()) buf[rng() % buf.size()] ^= static_cast<uint8_t>(1u << (rng() % 8));
                break;
        }
        if (buf.size() > 4096) buf.resize(4096);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <corpus-dir> [mutations] [seed]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const long mutations = argc > 2 ? std::atol(argv[2]) : 20000;
    // The CTest suites leave the seed alone so a failure there is reproducible.
    // A longer session passes one, which is how the same net covers more shapes
    // without making the regression suite non-deterministic.
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10))
                                   : 0x5EEDu;

    auto corpus = load_corpus(dir);
    if (corpus.empty()) {
        // A target with no corpus still gets a few shapes so the entry point is exercised.
        corpus.push_back({'0', '1', '2', '3'});
        corpus.push_back(std::vector<uint8_t>(64, 0));
        corpus.push_back(std::vector<uint8_t>(64, 0xFF));
    }

    for (const auto& seed : corpus) {
        LLVMFuzzerTestOneInput(seed.data(), seed.size());
    }

    std::mt19937 rng(seed);
    for (long i = 0; i < mutations; ++i) {
        std::vector<uint8_t> buf = corpus[rng() % corpus.size()];
        mutate(buf, rng, corpus);
        LLVMFuzzerTestOneInput(buf.data(), buf.size());
    }

    std::printf("replayed %zu corpus entries and %ld mutations\n", corpus.size(), mutations);
    return 0;
}
