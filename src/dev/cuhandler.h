#ifndef FALM_CUHANDLER_H
#define FALM_CUHANDLER_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../flag.h"
#include "../error.h"

namespace Falm {

static inline int falmMalloc(void **ptr, size_t size) {
    *ptr =  malloc(size);
    if (*ptr == nullptr) {
        return FalmErr::mallocErr;
    }
    // printf("falm malloc called\n");
    return FalmErr::success;
}

// static inline void *falmMallocPinned(size_t size) {
//     void *ptr;
//     cudaMallocHost(&ptr, size);
//     return ptr;
// }

static inline int falmMallocPinned(void **ptr, size_t size) {
    // return FalmErr::cuErrMask * (int)cudaMallocHost(ptr, size);
    *ptr =  malloc(size);
    if (*ptr == nullptr) {
        return FalmErr::mallocErr;
    }
    // printf("falm malloc pinned called\n");
    return FalmErr::success;
}

static inline int falmMallocDevice(void **ptr, size_t size) {
    // printf("falm malloc device called\n");
    return FalmErr::cuErrMask * (int)cudaMalloc(ptr, size);
}

// static inline void falmMallocDeviceAsync(void **ptr, size_t size, STREAM stream = (STREAM)0) {
//     cudaMallocAsync(ptr, size, stream);
// }

static inline int falmMemset(void *ptr, int value, size_t size) {
    memset(ptr, value, size);
    return FalmErr::success;
}

static inline int falmMemsetDevice(void *ptr, int value, size_t size) {
    return FalmErr::cuErrMask * (int)cudaMemset(ptr, value, size);
}

static inline int falmMemsetDeviceAsync(void *ptr, int value, size_t size, STREAM stream = (STREAM)0) {
    return FalmErr::cuErrMask * (int)cudaMemsetAsync(ptr, value, size, stream);
}

static inline int falmFree(void *ptr) {
    // printf("falm free called\n");
    free(ptr);
    return FalmErr::success;
}

// static inline void falmFreePinned(void *ptr) {
//     cudaFreeHost(ptr);
// }

static inline int falmFreePinned(void *ptr) {
    // printf("falm free pinned called\n");
    // return FalmErr::cuErrMask * (int)cudaFreeHost(ptr);
    free(ptr);
    return FalmErr::success;
}

static inline int falmFreeDevice(void *ptr) {
    // printf("falm free device called\n");
    /* cudaError_t err =  */ 
    return FalmErr::cuErrMask * (int)cudaFree(ptr);
    // if (err != cudaSuccess) {
    //     printf("cuda free device failed with error %d\n", (int)err);
    // }
}

// static inline void falmFreeDeviceAsync(void *ptr, STREAM stream = (STREAM)0) {
//     cudaFreeAsync(ptr, stream);
// }

static int falmMemcpy(void *dst, void *src, size_t size, FLAG mcptype) {
    if (mcptype == MCP::Hst2Hst) {
        memcpy(dst, src, size);
        return FalmErr::success;
    } else if (mcptype == MCP::Hst2Dev) {
        return FalmErr::cuErrMask * (int)cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else if (mcptype == MCP::Dev2Hst) {
        return FalmErr::cuErrMask * (int)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    } else if (mcptype == MCP::Dev2Dev) {
        return FalmErr::cuErrMask * (int)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    return FalmErr::success;
}

static int falmMemcpyAsync(void *dst, void *src, size_t size, FLAG mcptype, STREAM stream = (STREAM)0) {
    if (mcptype == MCP::Hst2Hst) {
        return FalmErr::cuErrMask * (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, stream);
    } else if (mcptype == MCP::Hst2Dev) {
        return FalmErr::cuErrMask * (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    } else if (mcptype == MCP::Dev2Hst) {
        return FalmErr::cuErrMask * (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    } else if (mcptype == MCP::Dev2Dev) {
        return FalmErr::cuErrMask * (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
    return FalmErr::success;
}

static inline int falmWaitStream(STREAM stream = (STREAM)0) {
    return FalmErr::cuErrMask * (int)cudaStreamSynchronize(stream);
}

}

#endif
