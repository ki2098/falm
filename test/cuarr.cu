#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add(double *a, double *b, int3 sz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i < sz.x && j < sz.y && k < sz.x) {
        int idx = i + j * sz.x + k * sz.y * sz.z;
        b[idx] += a[idx];
    }
}

int main() {
    double a[27];
    double b[27];
    int3 sz = {3, 3, 3};

    for (int i = 0; i < sz.x; i ++) {
    for (int j = 0; j < sz.y; j ++) {
    for (int k = 0; k < sz.z; k ++) {
        int idx = i + j * sz.x + k * sz.y * sz.z;
        b[idx] = 1;
        a[idx] = 2;
    }}}

    double *adev, *bdev;
    cudaMalloc(&adev, sizeof(double) * 27);
    cudaMalloc(&bdev, sizeof(double) * 27);
    cudaMemcpy(adev, a, sizeof(double) * 27, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, sizeof(double) * 27, cudaMemcpyHostToDevice);
    add<<<dim3(2, 2, 2), dim3(2, 2, 2)>>>(adev, bdev, sz);
    cudaMemcpy(a, adev, sizeof(double) * 27, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, bdev, sizeof(double) * 27, cudaMemcpyDeviceToHost);

    for (int i = 0; i < sz.x; i ++) {
    for (int j = 0; j < sz.y; j ++) {
    for (int k = 0; k < sz.z; k ++) {
        int idx = i + j * sz.x + k * sz.y * sz.z;
        printf("%lf %lf\n", a[idx], b[idx]);
    }}}
}