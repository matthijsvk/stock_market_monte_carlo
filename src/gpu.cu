#include "stock_market_monte_carlo/gpu.h"
#include "cuda.h"
#include <iostream>

using namespace std;

// actual GPU kernel
__global__ void impl_vector_add_gpu(float *out, float *a, float *b, int n) {
//    for (int i = 0; i < n; i++) {
//        out[i] = a[i] + b[i];
//    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}

// host function that allocates memory and calls the GPU
void vector_add_gpu(float *out, float *a, float *b, int n) {
    printf("Allocating device memory on host..\n");
    float *a_d, *b_d, *out_d;
    cudaMalloc((void **) &a_d, n * sizeof(float));
    cudaMalloc((void **) &b_d, n * sizeof(float));
    cudaMalloc((void **) &out_d, n * sizeof(float));

    printf("Copying to device..\n");
    cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 1024; // should always be multiple of 32, max 1024
    int block_no = ceil(float(n) / block_size);
    printf("block_no: %d, block_size: %d | \n", block_no, block_size);

    dim3 grid(block_no, 1, 1);
    dim3 block(block_size, 1, 1); // max block dimensions: [1024,1024,64]

    clock_t start_d = clock();
    impl_vector_add_gpu<<<grid, block>>>(out_d, a_d, b_d, n);
    cudaDeviceSynchronize();
    clock_t end_d = clock();
    double time_d = (double) (end_d - start_d) / CLOCKS_PER_SEC;

    cudaMemcpy(out, out_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU time: %f\n", time_d);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);
}
