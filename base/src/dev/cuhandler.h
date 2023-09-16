#ifndef FALM_CUHANDLER_H
#define FALM_CUHANDLER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include "../flag.h"

namespace Falm {

static inline void *falmHostMalloc(size_t size) {
    return malloc(size);
}

static inline void *falmDevMalloc(size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

static inline void falmHostMemset(void *ptr, int value, size_t size) {
    memset(ptr, value, size);
}

static inline void falmDevMemset(void *ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

static inline void falmHostFreePtr(void *ptr) {
    free(ptr);
    ptr = nullptr;
}

static inline void falmDevFreePtr(void *ptr) {
    cudaFree(ptr);
    ptr = nullptr;
}

static void falmMemcpy(void *dst, void *src, size_t size, unsigned int mcptype) {
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

}

#endif