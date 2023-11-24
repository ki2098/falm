#ifndef FALM_CUHANDLER_H
#define FALM_CUHANDLER_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../flag.h"
#include "../error.h"

namespace Falm {

static inline INT falmMalloc(void **ptr, size_t size) {
    *ptr =  malloc(size);
    if (*ptr == nullptr) {
        return FalmErr::mallocErr;
    }
    return FalmErr::success;
}

// static inline void *falmMallocPinned(size_t size) {
//     void *ptr;
//     cudaMallocHost(&ptr, size);
//     return ptr;
// }

static inline INT falmMallocPinned(void **ptr, size_t size) {
    return FalmErr::cuErrMask * (INT)cudaMallocHost(ptr, size);
}

static inline INT falmMallocDevice(void **ptr, size_t size) {
    return FalmErr::cuErrMask * (INT)cudaMalloc(ptr, size);
}

static inline INT falmMallocDeviceAsync(void **ptr, size_t size, STREAM stream = 0) {
    return FalmErr::cuErrMask * (INT)cudaMallocAsync(ptr, size, stream);
}

// static inline void falmMallocDeviceAsync(void **ptr, size_t size, STREAM stream = (STREAM)0) {
//     cudaMallocAsync(ptr, size, stream);
// }

static inline INT falmMemset(void *ptr, int value, size_t size) {
    memset(ptr, value, size);
    return FalmErr::success;
}

static inline INT falmMemsetDevice(void *ptr, int value, size_t size) {
    return FalmErr::cuErrMask * (INT)cudaMemset(ptr, value, size);
}

static inline INT falmMemsetDeviceAsync(void *ptr, int value, size_t size, STREAM stream = (STREAM)0) {
    return FalmErr::cuErrMask * (INT)cudaMemsetAsync(ptr, value, size, stream);
}

static inline INT falmFree(void *ptr) {
    free(ptr);
    return FalmErr::success;
}

// static inline void falmFreePinned(void *ptr) {
//     cudaFreeHost(ptr);
// }

static inline INT falmFreePinned(void *ptr) {
    return FalmErr::cuErrMask * (INT)cudaFreeHost(ptr);
}

static inline INT falmFreeDevice(void *ptr) {
    /* cudaError_t err =  */ 
    return FalmErr::cuErrMask * (INT)cudaFree(ptr);
    // if (err != cudaSuccess) {
    //     printf("cuda free device failed with error %d\n", (int)err);
    // }
}

// static inline void falmFreeDeviceAsync(void *ptr, STREAM stream = (STREAM)0) {
//     cudaFreeAsync(ptr, stream);
// }

static inline INT falmFreeDeviceAsync(void *ptr, STREAM stream = 0) {
    return FalmErr::cuErrMask * (INT)cudaFreeAsync(ptr, stream);
}

static INT falmMemcpy(void *dst, void *src, size_t size, FLAG mcptype) {
    if (mcptype == MCpType::Hst2Hst) {
        memcpy(dst, src, size);
        return FalmErr::success;
    } else if (mcptype == MCpType::Hst2Dev) {
        return FalmErr::cuErrMask * (INT)cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else if (mcptype == MCpType::Dev2Hst) {
        return FalmErr::cuErrMask * (INT)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    } else if (mcptype == MCpType::Dev2Dev) {
        return FalmErr::cuErrMask * (INT)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    return FalmErr::success;
}

static INT falmMemcpyAsync(void *dst, void *src, size_t size, FLAG mcptype, STREAM stream = (STREAM)0) {
    if (mcptype == MCpType::Hst2Hst) {
        return FalmErr::cuErrMask * (INT)cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, stream);
    } else if (mcptype == MCpType::Hst2Dev) {
        return FalmErr::cuErrMask * (INT)cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    } else if (mcptype == MCpType::Dev2Hst) {
        return FalmErr::cuErrMask * (INT)cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    } else if (mcptype == MCpType::Dev2Dev) {
        return FalmErr::cuErrMask * (INT)cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
    return FalmErr::success;
}

static inline INT falmWaitStream(STREAM stream = (STREAM)0) {
    return FalmErr::cuErrMask * (INT)cudaStreamSynchronize(stream);
}

}

#endif
