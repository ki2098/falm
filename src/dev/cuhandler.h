#ifndef FALM_CUHANDLER_H
#define FALM_CUHANDLER_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../flag.h"

namespace Falm {

static inline void *falmMalloc(size_t size) {
    return malloc(size);
}

static inline void *falmMallocPinned(size_t size) {
    void *ptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

static inline void *falmMallocDevice(size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

static inline void falmMallocDeviceAsync(void **ptr, size_t size, STREAM stream) {
    cudaMallocAsync(ptr, size, stream);
}

static inline void falmMemset(void *ptr, int value, size_t size) {
    memset(ptr, value, size);
}

static inline void falmMemsetDevice(void *ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

static inline void falmMemsetDeviceAsync(void *ptr, int value, size_t size, STREAM stream) {
    cudaMemsetAsync(ptr, value, size, stream);
}

static inline void falmFree(void *ptr) {
    free(ptr);
}

static inline void falmFreePinned(void *ptr) {
    cudaFreeHost(ptr);
}

static inline void falmFreeDevice(void *ptr) {
    cudaFree(ptr);
}

static inline void falmFreeDeviceAsync(void *ptr, STREAM stream) {
    cudaFreeAsync(ptr, stream);
}

static void falmMemcpy(void *dst, void *src, size_t size, FLAG mcptype) {
    if (mcptype == MCpType::Hst2Hst) {
        memcpy(dst, src, size);
    } else if (mcptype == MCpType::Hst2Dev) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else if (mcptype == MCpType::Dev2Hst) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    } else if (mcptype == MCpType::Dev2Dev) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
}

static void falmMemcpyAsync(void *dst, void *src, size_t size, FLAG mcptype, STREAM stream) {
    if (mcptype == MCpType::Hst2Hst) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, stream);
    } else if (mcptype == MCpType::Hst2Dev) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    } else if (mcptype == MCpType::Dev2Hst) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    } else if (mcptype == MCpType::Dev2Dev) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
}

}

#endif
