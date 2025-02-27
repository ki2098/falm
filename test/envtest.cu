#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void foo(double *array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        array[tid] += 1;
        printf("%lf\n", array[tid]);
    }
}

const int N = 100;

int main() {
    double array[N];
    for (int i = 0; i < N; i ++) {
        array[i] = i;
    }
    double *array_device;
    cudaMalloc((void**)&array_device, sizeof(double)*N);
    cudaMemcpy(array_device, array, sizeof(double)*N, cudaMemcpyHostToDevice);
    const int block = 32;
    const int grid = (N + block - 1) / block;
    foo<<<grid, block>>>(array_device, N);
    cudaMemcpy(array, array_device, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaFree(array_device);
    for (int i = 0; i < N; i ++) {
        printf("%lf\n", array[i]);
    }

    return 0;
}