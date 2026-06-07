// MathScript Unsafe Registry - Tracks [[ms::unsafe]] annotation sites
// Every unsafe site must be registered with a justification

#pragma once

#include <string>
#include <map>
#include <vector>

namespace ms::plugin {

struct UnsafeSite {
    std::string file;
    int line;
    std::string justification;
    std::string category;  // "CUDA interop", "SIMD intrinsics", etc.
};

class UnsafeRegistry {
public:
    static UnsafeRegistry& instance();
    
    void record_unsafe(const std::string& file, int line,
                       const std::string& justification,
                       const std::string& category);
    
    std::vector<UnsafeSite> get_sites() const;
    
    int count() const;
    
private:
    UnsafeRegistry() = default;
    std::map<std::string, std::vector<UnsafeSite>> sites_;
};

} // namespace ms::plugin

// Custom attribute for marking unsafe blocks
#define UNSAFE_SITE(reason) ms::plugin::record_unsafe_annotation(__FILE__, __LINE__, reason)