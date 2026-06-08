// MUST NOT compile when the ms-profile Clang plugin is active.

#include <expected>

std::expected<int, int> compute() {
    return 1;
}

int main() {
    compute();
    return 0;
}
