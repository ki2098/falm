#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

struct Size {
    int sz[3];

    __device__ __host__ int &operator[](int i) {
        return sz[i];
    }
};

__global__ void add(double *a, double *b, Size sz) {
    Size __sz{{sz[0], sz[1], sz[2]}};
    int isz = sz[0];
    int jsz = sz[1];
    int ksz = sz[2];
    int i = threadIdx[0] + blockIdx[0] * blockDim[0];
    int j = threadIdx[1] + blockIdx[1] * blockDim[1];
    int k = threadIdx[2] + blockIdx[2] * blockDim[2];
    if (i < isz && j < jsz && k < ksz) {
        int idx = i + j * isz + k * isz * jsz;
        b[idx] += a[idx];
    }
}

int main() {
    double a[27];
    double b[27];
    int sz[] = {3, 3, 3};

    for (int i = 0; i < sz[0]; i ++) {
    for (int j = 0; j < sz[1]; j ++) {
    for (int k = 0; k < sz[2]; k ++) {
        int idx = i + j * sz[0] + k * sz[0] * sz[1];
        b[idx] = 1;
        a[idx] = 2;
    }}}

    Size vsz{sz[0], sz[1], sz[2]};

    double *adev, *bdev;
    cudaMalloc(&adev, sizeof(double) * 27);
    cudaMalloc(&bdev, sizeof(double) * 27);
    cudaMemcpy(adev, a, sizeof(double) * 27, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, sizeof(double) * 27, cudaMemcpyHostToDevice);
    add<<<dim3(2, 2, 2), dim3(2, 2, 2)>>>(adev, bdev, vsz);
    cudaMemcpy(a, adev, sizeof(double) * 27, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, bdev, sizeof(double) * 27, cudaMemcpyDeviceToHost);

    for (int i = 0; i < sz[0]; i ++) {
    for (int j = 0; j < sz[1]; j ++) {
    for (int k = 0; k < sz[2]; k ++) {
        int idx = i + j * sz[0] + k * sz[0] * sz[1];
        printf("%lf %lf\n", a[idx], b[idx]);
    }}}

    Size sz0{1, 2, 3};
    Size sz1 = sz0;
    sz1[1] = 5;
    printf("%d %d %d %d %d %d\n", sz0[0], sz0[1], sz0[2], sz1[0], sz1[1], sz1[2]);
}