// MUST compile cleanly with the ms-profile Clang plugin active.
// Local stand-in: the plugin matches any record named `expected` (libstdc++
// may not expose std::expected to Clang + -fno-exceptions on CI).

template <typename T, typename E>
struct expected {
    T value{};
    expected() = default;
    expected(T v) : value(v) {}
    explicit operator bool() const { return true; }
    const T& operator*() const { return value; }
};

expected<int, int> compute() {
    return 1;
}

int main() {
    const auto result = compute();
    if (!result) {
        return 1;
    }
    return *result;
}
