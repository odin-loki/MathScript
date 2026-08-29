// MUST NOT compile when the ms-profile Clang plugin is active.
// Local stand-in: the plugin matches any record named `expected`.

template <typename T, typename E>
struct expected {
    T value{};
    expected() = default;
    expected(T v) : value(v) {}
};

expected<int, int> compute() {
    return 1;
}

int main() {
    compute();
    return 0;
}
