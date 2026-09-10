// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.3: golden transcripts for the REPL, run through `mathscriptc`.
//
// `repl_engine.cpp` and `repl_engine_internal.cpp` are about 40,000 lines in two
// files, and the test that covered them was one 21,000-line source. Adding a command
// meant editing that file, which is why coverage of the dispatch chain grew by
// accretion and why nobody could tell from a diff what a change to it was for.
//
// A transcript is a different shape of test. `tests/repl_corpus/x.ms` is a script; the
// committed `x.out` beside it is exactly what running it prints; `x.err` is exactly
// what it writes to standard error, and its absence means "nothing". Adding a command
// means adding a file. A behaviour change shows up as a diff in the expected output,
// which is a thing a reviewer can read, rather than as an edit to an assertion, which
// is a thing a reviewer has to reconstruct.
//
// The corpus is discovered at run time, so a new transcript needs no build-system
// edit. That also means an empty or unreadable corpus directory would let this file
// pass while testing nothing, so the first assertion is that the corpus is not empty.
//
// To refresh the expected files after an intentional change:
//
//     MS_REPL_CORPUS_UPDATE=1 ./build/bin/test_repl_corpus
//
// and read the resulting diff before committing it. That is the point of the mode: the
// expected files are the record of what the program does, and a record that is
// regenerated without being read is not a record.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#ifndef MATHSCRIPTC_PATH
#error "MATHSCRIPTC_PATH must be defined by CMake"
#endif
#ifndef MS_REPL_CORPUS_DIR
#error "MS_REPL_CORPUS_DIR must be defined by CMake"
#endif

namespace {

namespace fs = std::filesystem;

int system_exit_code(const std::string& command) {
    const int rc = std::system(command.c_str());
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) {
        return -1;
    }
    if (WIFEXITED(rc)) {
        return WEXITSTATUS(rc);
    }
    if (WIFSIGNALED(rc)) {
        return 128 + WTERMSIG(rc);
    }
    return rc;
#endif
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void write_file(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

/// Windows writes "\r\n" where the expected files hold "\n". The transcripts are about
/// what the REPL says, not about which line ending the C runtime chose, so the
/// comparison is made on normalised text and the committed files stay in one form.
std::string normalise(std::string text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') {
            continue;
        }
        out.push_back(text[i]);
    }
    return out;
}

bool updating() {
    const char* flag = std::getenv("MS_REPL_CORPUS_UPDATE");
    return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

std::vector<fs::path> corpus_scripts() {
    std::vector<fs::path> scripts;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(fs::path(MS_REPL_CORPUS_DIR), ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ms") {
            scripts.push_back(entry.path());
        }
    }
    std::sort(scripts.begin(), scripts.end());
    return scripts;
}

/// The first line of the difference, with its line number. A whole-transcript diff in a
/// GoogleTest failure message is unreadable; the first line that differs is the thing
/// anyone actually looks for.
std::string first_difference(const std::string& expected, const std::string& actual) {
    std::istringstream want(expected);
    std::istringstream got(actual);
    std::string want_line;
    std::string got_line;
    int number = 0;
    while (true) {
        const bool have_want = static_cast<bool>(std::getline(want, want_line));
        const bool have_got = static_cast<bool>(std::getline(got, got_line));
        ++number;
        if (!have_want && !have_got) {
            return "the two are identical line by line but differ in trailing bytes";
        }
        if (!have_want) {
            return "line " + std::to_string(number) + ": expected end of output, got \"" +
                   got_line + "\"";
        }
        if (!have_got) {
            return "line " + std::to_string(number) + ": expected \"" + want_line +
                   "\", got end of output";
        }
        if (want_line != got_line) {
            return "line " + std::to_string(number) + ":\n  expected \"" + want_line +
                   "\"\n  actual   \"" + got_line + "\"";
        }
    }
}

} // namespace

TEST(ReplCorpus, TheCorpusIsNotEmpty) {
    const auto scripts = corpus_scripts();
    ASSERT_FALSE(scripts.empty())
        << "no .ms files under " << MS_REPL_CORPUS_DIR
        << " -- every other assertion in this file would pass without running anything";
}

TEST(ReplCorpus, EveryTranscriptMatches) {
    const auto scripts = corpus_scripts();
    ASSERT_FALSE(scripts.empty());
    const fs::path work = fs::path(MATHSCRIPTC_PATH).parent_path();

    for (const fs::path& script : scripts) {
        const std::string stem = script.stem().string();
        const fs::path out_path = script.parent_path() / (stem + ".out");
        const fs::path err_path = script.parent_path() / (stem + ".err");
        const fs::path actual_out = work / (stem + ".actual.out");
        const fs::path actual_err = work / (stem + ".actual.err");

        const std::string command = "\"" + std::string(MATHSCRIPTC_PATH) + "\" \"" +
                                    script.generic_string() + "\" > \"" +
                                    actual_out.generic_string() + "\" 2> \"" +
                                    actual_err.generic_string() + "\"";
        const int status = system_exit_code(command);
        const std::string got_out = normalise(read_file(actual_out));
        const std::string got_err = normalise(read_file(actual_err));

        if (updating()) {
            write_file(out_path, got_out);
            if (got_err.empty()) {
                std::error_code ec;
                fs::remove(err_path, ec);
            } else {
                write_file(err_path, got_err);
            }
            continue;
        }

        ASSERT_TRUE(fs::exists(out_path))
            << stem << ".ms has no committed " << stem
            << ".out -- run with MS_REPL_CORPUS_UPDATE=1 and review the result";
        const std::string want_out = normalise(read_file(out_path));
        const std::string want_err = normalise(read_file(err_path)); // absent reads empty

        // A transcript that produced NOTHING on either stream is not a content
        // difference and should not be reported as one. `mathscriptc` cannot reach a
        // non-zero exit without writing to standard error, so empty-empty-nonzero means
        // the process died before its buffered stdout was flushed -- which is what
        // Windows did with the combined symbolic transcript, and the message said only
        // that line 1 was missing.
        if (got_out.empty() && got_err.empty() && status != 0) {
            // Say which of the two possible stories this is, rather than asserting one.
            //
            // `mathscriptc` reaches a non-zero exit only after writing to stderr -- the
            // four argument errors and the "cannot open file" all print first, and a
            // failing line prints "error: ..." -- and it flushes stdout after every
            // line. So it cannot itself produce this. Two things outside it can:
            //
            //   1. the process died, losing whatever was in flight; the redirect files
            //      exist and are empty (or hold a partial transcript, in which case we
            //      would not be here);
            //   2. the shell never ran it -- it could not create a redirect file, or
            //      could not launch the program -- and returned its own status; the
            //      redirect files may not exist at all, and the shell's own complaint
            //      went to a stderr nobody captured.
            //
            // Whether the files exist separates the two, so report it. Then run the
            // same command again with nothing redirected: its output lands on this
            // process's own streams, which the test runner captures, so if there is a
            // message it becomes visible instead of being swallowed by a redirect that
            // may be the problem.
            const bool out_exists = fs::exists(actual_out);
            const bool err_exists = fs::exists(actual_err);
            const std::string bare =
                "\"" + std::string(MATHSCRIPTC_PATH) + "\" \"" + script.generic_string() + "\"";
            std::cout << "--- re-running " << stem << ".ms unredirected ---" << std::endl;
            const int bare_status = system_exit_code(bare);
            std::cout << "--- unredirected exit status " << bare_status << " ---" << std::endl;

            ADD_FAILURE() << stem << ".ms produced no output on either stream and exited "
                          << status << ". mathscriptc cannot do that through any normal "
                          << "path -- it writes to stderr before returning non-zero, and "
                          << "flushes stdout after every line -- so either it died or the "
                          << "shell never ran it.\n"
                          << "  redirect files: " << actual_out.generic_string() << " "
                          << (out_exists ? "exists" : "MISSING") << ", "
                          << actual_err.generic_string() << " "
                          << (err_exists ? "exists" : "MISSING") << "\n"
                          << "  a MISSING file means the shell could not create the "
                          << "redirect, so the status is the shell's and not the "
                          << "program's\n"
                          << "  re-run with no redirection exited " << bare_status
                          << "; anything it printed is above this failure";
            continue;
        }
        EXPECT_EQ(got_out, want_out)
            << stem << ".ms stdout differs -- " << first_difference(want_out, got_out);
        EXPECT_EQ(got_err, want_err)
            << stem << ".ms stderr differs -- " << first_difference(want_err, got_err);
        // mathscriptc exits non-zero when any line failed, so the exit code is a
        // one-bit summary of whether the transcript expected an error at all. Asserting
        // it separately catches a script whose error moved between lines in a way that
        // happens to leave the same text on the two streams.
        EXPECT_EQ(status, want_err.empty() ? 0 : 1)
            << stem << ".ms exit status";

        std::error_code ec;
        fs::remove(actual_out, ec);
        fs::remove(actual_err, ec);
    }
}
