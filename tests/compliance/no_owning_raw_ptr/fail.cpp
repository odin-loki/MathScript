// MUST NOT compile when the ms-profile Clang plugin is active.

struct Owner {
    int* data;
};

int main() {
    Owner o{};
    return o.data != nullptr ? 1 : 0;
}
