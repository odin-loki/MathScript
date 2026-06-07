// MathScript Compiler Implementation

#include <iostream>
#include <fstream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: mathscriptc input.msc [output.m]" << std::endl;
        return 1;
    }
    
    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "Error: Cannot open " << argv[1] << std::endl;
        return 1;
    }
    
    std::string content((std::istreambuf_iterator<char>(input)),
                        std::istreambuf_iterator<char>());
    
    std::ofstream output;
    if (argc >= 3) {
        output.open(argv[2]);
        if (!output) {
            std::cerr << "Error: Cannot open " << argv[2] << std::endl;
            return 1;
        }
    } else {
        output = std::cout;
    }
    
    output << "// Compiled output\n" << content;
    
    return 0;
}