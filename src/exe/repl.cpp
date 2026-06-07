#include <iostream>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/ms.hpp"

int main() {
    std::cout << "MathScript REPL v0.2 (console mode)\n";
    std::cout << "Type 'help' for commands, 'exit' to quit.\n";

    ms::interp::Interpreter interp;
    std::string line;
    while (true) {
        std::cout << "ms> ";
        if (!std::getline(std::cin, line)) {
            break;
        }
        const std::string trimmed = ms::interp::Interpreter::trim(line);
        if (trimmed == "exit" || trimmed == "quit") {
            break;
        }
        if (trimmed.empty()) {
            continue;
        }

        auto result = interp.execute(trimmed);
        if (result) {
            if (!result->empty()) {
                std::cout << *result;
            }
        } else {
            std::cout << "error: " << ms::format_error(result.error()) << "\n";
        }
    }

    return 0;
}
