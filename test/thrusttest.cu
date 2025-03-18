#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

int main() {
    double host_data[] = {1,2,3,4,5};
    double *device_data;
    cudaMalloc((void**)&device_data, 5*sizeof(double));
    cudaMemcpy(device_data, host_data, 5*sizeof(double), cudaMemcpyHostToDevice);
    thrust::device_ptr<double> device_ptr = thrust::device_pointer_cast(device_data);
    double sum = thrust::reduce(device_ptr, device_ptr + 5);
    printf("%lf\n", sum);
}