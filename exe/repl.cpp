// MathScript REPL Implementation

#include <iostream>
#include <string>
#include <memory>

int main(int argc, char* argv[]) {
    std::cout << "MathScript REPL - Type expressions and press Enter" << std::endl;
    std::cout << "Type 'quit' or Ctrl+D to exit" << std::endl;
    
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit") break;
        std::cout << "> " << line << std::endl;
    }
    
    return 0;
}