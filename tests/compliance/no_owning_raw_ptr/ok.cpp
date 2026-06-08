// MUST compile cleanly with the ms-profile Clang plugin active.

#include <memory>
#include <vector>

struct Owner {
    std::unique_ptr<int> data;
    std::vector<int> backup;
};

int main() {
    Owner o;
    return o.backup.size();
}
