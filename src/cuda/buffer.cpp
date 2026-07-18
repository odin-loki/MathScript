#include "ms/cuda/buffer.hpp"
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace ms::cuda {

namespace {

void free_buffer(void* data, MemoryKind kind) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (data == nullptr) {
        return;
    }
    if (kind == MemoryKind::Device) {
        cudaFree(data);
        return;
    }
    if (kind == MemoryKind::Pinned) {
        cudaFreeHost(data);
        return;
    }
#endif
    ::operator delete(data);
}

void* alloc_buffer(size_t bytes, MemoryKind kind, int device) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (kind == MemoryKind::Device) {
        cudaSetDevice(device);
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }
    if (kind == MemoryKind::Pinned) {
        void* ptr = nullptr;
        if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }
#endif
    (void)device;
    return ::operator new(bytes);
}

} // namespace

bool available() {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    return device_count() > 0;
#else
    return false;
#endif
}

int device_count() {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

std::string device_name(int device) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return "cuda-device";
    }
    return prop.name;
#else
    (void)device;
    return "cpu-stub";
#endif
}

DeviceBuffer::DeviceBuffer(size_t count, MemoryKind mem_kind, int device_id)
    : bytes(count * sizeof(double)), kind(mem_kind), device(device_id) {
    data = alloc_buffer(bytes, kind, device_id);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data(other.data),
      bytes(other.bytes),
      kind(other.kind),
      device(other.device) {
    other.data = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free_buffer(data, kind);
        data = other.data;
        bytes = other.bytes;
        kind = other.kind;
        device = other.device;
        other.data = nullptr;
        other.bytes = 0;
    }
    return *this;
}

DeviceBuffer::~DeviceBuffer() {
    free_buffer(data, kind);
}

PinnedBuffer::PinnedBuffer(size_t byte_count) : bytes(byte_count) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    cudaHostAlloc(&data, bytes, cudaHostAllocDefault);
#else
    data = ::operator new(bytes);
#endif
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data(other.data), bytes(other.bytes) {
    other.data = nullptr;
    other.bytes = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
        if (data) {
            cudaFreeHost(data);
        }
#else
        ::operator delete(data);
#endif
        data = other.data;
        bytes = other.bytes;
        other.data = nullptr;
        other.bytes = 0;
    }
    return *this;
}

PinnedBuffer::~PinnedBuffer() {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (data) {
        cudaFreeHost(data);
    }
#else
    ::operator delete(data);
#endif
}

struct StreamPool::Impl {
    static constexpr size_t kDefaultStreams = 8;
    std::vector<void*> owned;
    std::vector<void*> available;

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    ~Impl() {
        for (void* stream : owned) {
            if (stream != nullptr) {
                cudaStreamDestroy(static_cast<cudaStream_t>(stream));
            }
        }
    }
#endif
};

StreamPool::StreamPool() : impl_(new Impl) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (!available()) {
        return;
    }
    impl_->owned.reserve(Impl::kDefaultStreams);
    impl_->available.reserve(Impl::kDefaultStreams);
    for (size_t i = 0; i < Impl::kDefaultStreams; ++i) {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            continue;
        }
        impl_->owned.push_back(stream);
        impl_->available.push_back(stream);
    }
#endif
}

StreamPool::~StreamPool() {
    delete impl_;
}

void* StreamPool::acquire() {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (impl_ != nullptr && !impl_->available.empty()) {
        void* stream = impl_->available.back();
        impl_->available.pop_back();
        return stream;
    }
    if (available() && impl_ != nullptr) {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) == cudaSuccess) {
            impl_->owned.push_back(stream);
            return stream;
        }
    }
#else
    (void)impl_;
#endif
    return nullptr;
}

void StreamPool::release(void* stream) {
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (stream != nullptr && impl_ != nullptr) {
        impl_->available.push_back(stream);
    }
#else
    (void)stream;
    (void)impl_;
#endif
}

DeviceBuffer make_device_buffer(size_t bytes, int device) {
    const MemoryKind kind =
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
        available() ? MemoryKind::Device : MemoryKind::Host;
#else
        MemoryKind::Host;
#endif
    const size_t count = (bytes + sizeof(double) - 1) / sizeof(double);
    return DeviceBuffer(count, kind, device);
}

void copy_host_to_device(const void* host, DeviceBuffer& device, size_t bytes) {
    if (device.data == nullptr || host == nullptr) {
        return;
    }
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (device.kind == MemoryKind::Device) {
        cudaMemcpy(device.data, host, bytes, cudaMemcpyHostToDevice);
        return;
    }
#endif
    std::memcpy(device.data, host, bytes);
}

void copy_device_to_host(const DeviceBuffer& device, void* host, size_t bytes) {
    if (device.data == nullptr || host == nullptr) {
        return;
    }
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (device.kind == MemoryKind::Device) {
        cudaMemcpy(host, device.data, bytes, cudaMemcpyDeviceToHost);
        return;
    }
#endif
    std::memcpy(host, device.data, bytes);
}

} // namespace ms::cuda
