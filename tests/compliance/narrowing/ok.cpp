// MUST compile cleanly with the ms-profile Clang plugin active.

int main() {
    long long wide = 42LL;
    const int narrow = static_cast<int>(wide);
    return narrow;
}
