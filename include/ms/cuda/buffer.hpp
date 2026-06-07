#pragma once

#include <cstddef>
#include <string>

namespace ms::cuda {

enum class MemoryKind { Host, Pinned, Device };

struct DeviceBuffer {
    void* data = nullptr;
    size_t bytes = 0;
    MemoryKind kind = MemoryKind::Host;
    int device = -1;

    DeviceBuffer() = default;
    DeviceBuffer(size_t count, MemoryKind mem_kind, int device_id = 0);
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer();

    template<typename T>
    T* as() {
        return static_cast<T*>(data);
    }

    template<typename T>
    const T* as() const {
        return static_cast<const T*>(data);
    }
};

struct PinnedBuffer {
    void* data = nullptr;
    size_t bytes = 0;

    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t bytes);
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
    ~PinnedBuffer();
};

class StreamPool {
public:
    StreamPool();
    ~StreamPool();
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    void* acquire();
    void release(void* stream);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

bool available();
int device_count();
std::string device_name(int device = 0);

DeviceBuffer make_device_buffer(size_t bytes, int device = 0);
void copy_host_to_device(const void* host, DeviceBuffer& device, size_t bytes);
void copy_device_to_host(const DeviceBuffer& device, void* host, size_t bytes);

} // namespace ms::cuda
