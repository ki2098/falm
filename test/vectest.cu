#include <stdlib.h>
#include <vector>
#include <stdio.h>

using namespace std;

__global__ void vecaddone(double *vec, size_t size) {
    size_t id = blockDim.x*blockIdx.x + threadIdx.x;
    if (id < size) {
        vec[id] += 1;
    }
}

int main() {
    vector<double> x(100);
    for (size_t i = 0; i < x.size(); i ++) {
        x[i] = i;
    }
    double *ptr;
    cudaMalloc(&ptr, sizeof(double)*x.size());
    cudaMemcpy(ptr, x.data(), sizeof(double)*x.size(), cudaMemcpyHostToDevice);
    vecaddone<<<8,8>>>(ptr, x.size());
    cudaMemcpy(x.data(), ptr, sizeof(double)*x.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < x.size(); i ++) {
        printf("%lf\n", x[i]);
    }
}