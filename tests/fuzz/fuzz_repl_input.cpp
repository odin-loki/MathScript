#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

#include "ms/interp/repl_engine.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static const bool isolated_cwd = [] {
        std::error_code ec;
        const auto dir = std::filesystem::temp_directory_path() / "mathscript_fuzz_repl";
        std::filesystem::create_directories(dir, ec);
        if (!ec) {
            std::filesystem::current_path(dir, ec);
        }
        return true;
    }();
    (void)isolated_cwd;

    if (size < 4 || size > 512) {
        return 0;
    }

    std::string line(reinterpret_cast<const char*>(data), size);
    for (char& ch : line) {
        if (ch == '\0' || ch == '\r' || ch == '\n') {
            ch = ' ';
        }
    }
    line = ms::interp::Interpreter::trim(line);
    if (line.empty()) {
        return 0;
    }

    ms::interp::Interpreter interp;
    (void)interp.execute(line);
    (void)interp.execute("help");
    (void)interp.execute("vars");
    return 0;
}
