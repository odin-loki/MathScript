// MUST compile cleanly with the ms-profile Clang plugin active.

#include <expected>

std::expected<int, int> compute() {
    return 1;
}

int main() {
    const auto result = compute();
    if (!result) {
        return 1;
    }
    return *result;
}
