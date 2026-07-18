// MathScript Unsafe Registry - collects [[ms::unsafe]] / MS_UNSAFE audit sites

#include "unsafe_registry.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace ms::plugin {

namespace {

std::mutex g_registry_mutex;

std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char ch : text) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += ch;
            break;
        }
    }
    return out;
}

std::string infer_category(const std::string& justification) {
    const auto colon = justification.find(':');
    if (colon != std::string::npos && colon > 0 && colon < 64) {
        return justification.substr(0, colon);
    }
    return "general";
}

std::string site_to_json_line(const UnsafeSite& site) {
    std::ostringstream oss;
    oss << "{\"file\":\"" << json_escape(site.file) << "\","
        << "\"line\":" << site.line << ","
        << "\"justification\":\"" << json_escape(site.justification) << "\","
        << "\"category\":\"" << json_escape(site.category) << "\"}";
    return oss.str();
}

bool site_key_equal(const UnsafeSite& a, const UnsafeSite& b) {
    return a.file == b.file && a.line == b.line &&
           a.justification == b.justification;
}

std::vector<UnsafeSite> parse_jsonl_sites(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }

    std::vector<UnsafeSite> sites;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        auto read_string = [&](const char* key) -> std::string {
            const std::string needle = std::string("\"") + key + "\":\"";
            const auto key_pos = line.find(needle);
            if (key_pos == std::string::npos) {
                return {};
            }
            std::string value;
            bool escaped = false;
            for (std::size_t i = key_pos + needle.size(); i < line.size(); ++i) {
                const char ch = line[i];
                if (escaped) {
                    if (ch == 'n') {
                        value.push_back('\n');
                    } else if (ch == 'r') {
                        value.push_back('\r');
                    } else if (ch == 't') {
                        value.push_back('\t');
                    } else {
                        value.push_back(ch);
                    }
                    escaped = false;
                    continue;
                }
                if (ch == '\\') {
                    escaped = true;
                    continue;
                }
                if (ch == '"') {
                    break;
                }
                value.push_back(ch);
            }
            return value;
        };

        auto read_int = [&](const char* key) -> int {
            const std::string needle = std::string("\"") + key + "\":";
            const auto key_pos = line.find(needle);
            if (key_pos == std::string::npos) {
                return 0;
            }
            return std::stoi(line.substr(key_pos + needle.size()));
        };

        UnsafeSite site;
        site.file = read_string("file");
        site.line = read_int("line");
        site.justification = read_string("justification");
        site.category = read_string("category");
        if (!site.file.empty() && site.line > 0) {
            sites.push_back(std::move(site));
        }
    }
    return sites;
}

void merge_unique(std::vector<UnsafeSite>& into, const std::vector<UnsafeSite>& add) {
    for (const auto& site : add) {
        const auto found = std::find_if(into.begin(), into.end(),
                                        [&](const UnsafeSite& existing) {
                                            return site_key_equal(existing, site);
                                        });
        if (found == into.end()) {
            into.push_back(site);
        }
    }
}

bool write_json_report(const std::string& path, const std::vector<UnsafeSite>& sites) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return false;
    }

    out << "{\n  \"version\": 1,\n  \"total\": " << sites.size() << ",\n  \"sites\": [\n";
    for (std::size_t i = 0; i < sites.size(); ++i) {
        const auto& site = sites[i];
        out << "    {\"file\":\"" << json_escape(site.file) << "\","
            << "\"line\":" << site.line << ","
            << "\"justification\":\"" << json_escape(site.justification) << "\","
            << "\"category\":\"" << json_escape(site.category) << "\"}";
        if (i + 1 < sites.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n}\n";
    return static_cast<bool>(out);
}

bool append_jsonl_sites(const std::string& path, const std::vector<UnsafeSite>& sites) {
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return false;
    }
    for (const auto& site : sites) {
        out << site_to_json_line(site) << '\n';
    }
    return static_cast<bool>(out);
}

} // namespace

UnsafeRegistry& UnsafeRegistry::instance() {
    static UnsafeRegistry registry;
    return registry;
}

void UnsafeRegistry::record_unsafe(const std::string& file, int line,
                                   const std::string& justification,
                                   const std::string& category) {
    if (file.empty() || line <= 0 || justification.empty()) {
        return;
    }

    std::lock_guard lock(g_registry_mutex);
    UnsafeSite site{file, line, justification,
                    category.empty() ? infer_category(justification) : category};

    auto& bucket = sites_[file];
    const auto found = std::find_if(bucket.begin(), bucket.end(),
                                    [&](const UnsafeSite& existing) {
                                        return site_key_equal(existing, site);
                                    });
    if (found == bucket.end()) {
        bucket.push_back(std::move(site));
    }
}

std::vector<UnsafeSite> UnsafeRegistry::get_sites() const {
    std::lock_guard lock(g_registry_mutex);
    std::vector<UnsafeSite> all;
    for (const auto& [file, bucket] : sites_) {
        (void)file;
        all.insert(all.end(), bucket.begin(), bucket.end());
    }
    std::sort(all.begin(), all.end(), [](const UnsafeSite& a, const UnsafeSite& b) {
        if (a.file != b.file) {
            return a.file < b.file;
        }
        return a.line < b.line;
    });
    return all;
}

int UnsafeRegistry::count() const {
    std::lock_guard lock(g_registry_mutex);
    int total = 0;
    for (const auto& [file, bucket] : sites_) {
        (void)file;
        total += static_cast<int>(bucket.size());
    }
    return total;
}

bool UnsafeRegistry::emit_audit_report(const std::string& output_dir,
                                         const std::vector<UnsafeSite>& sites) {
    if (output_dir.empty() || sites.empty()) {
        return false;
    }

    const std::string jsonl_path = output_dir + "/ms-unsafe-audit.jsonl";
    const std::string json_path = output_dir + "/ms-unsafe-audit.json";
    if (!append_jsonl_sites(jsonl_path, sites)) {
        return false;
    }

    auto merged = parse_jsonl_sites(jsonl_path);
    std::sort(merged.begin(), merged.end(), [](const UnsafeSite& a, const UnsafeSite& b) {
        if (a.file != b.file) {
            return a.file < b.file;
        }
        return a.line < b.line;
    });
    merged.erase(std::unique(merged.begin(), merged.end(),
                             [](const UnsafeSite& a, const UnsafeSite& b) {
                                 return site_key_equal(a, b);
                             }),
                 merged.end());
    return write_json_report(json_path, merged);
}

void record_unsafe_annotation(const char* file, int line, const char* reason) {
    if (!file || !reason) {
        return;
    }
    UnsafeRegistry::instance().record_unsafe(file, line, reason, "");
}

} // namespace ms::plugin
