#ifndef FALM_CUHANDLER_H
#define FALM_CUHANDLER_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../flag.h"

namespace Falm {

static inline void *falmHostMalloc(SIZE_T size) {
    return malloc(size);
}

static inline void *falmDevMalloc(SIZE_T size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

static inline void falmHostMemset(void *ptr, INT_T value, SIZE_T size) {
    memset(ptr, value, size);
}

static inline void falmDevMemset(void *ptr, INT_T value, SIZE_T size) {
    cudaMemset(ptr, value, size);
}

static inline void falmHostFreePtr(void *ptr) {
    free(ptr);
}

static inline void falmDevFreePtr(void *ptr) {
    cudaFree(ptr);
}

static void falmMemcpy(void *dst, void *src, SIZE_T size, FLAG mcptype) {
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
