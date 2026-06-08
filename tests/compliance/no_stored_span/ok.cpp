// MUST compile cleanly with the ms-profile Clang plugin active.

#include <span>

void consume(std::span<const int> view) {
    (void)view.size();
}

int main() {
    int data[2] = {1, 2};
    consume(data);
    return 0;
}
