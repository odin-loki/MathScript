// MUST NOT compile when the ms-profile Clang plugin is active.

#include <span>

struct Holder {
    std::span<int> view;
};

int main() {
    int data[2] = {1, 2};
    Holder h{data};
    return static_cast<int>(h.view.size());
}
