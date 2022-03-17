#include "cuda.h"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

//#define DEBUG

__device__ void update_fund(float fund_value, float period_return, float &next_value) {
  next_value = fund_value * (float(100.0) + period_return) / 100;
}

__global__ void __many_updates_gpu_kernel(float *returns, float *totals, int n_periods) {

  for (int i = 0; i < n_periods; i++) {
    update_fund(totals[i], returns[i], totals[i + 1]);
  }
}

// host function that allocates memory and calls the GPU
void __many_updates_gpu(float *returns, float *totals, int n) {
#ifdef DEBUG
  printf("Allocating device memory on host..\n");
#endif
  float *returns_d, *totals_d;
  cudaMalloc((void **) &returns_d, n * sizeof(float));
  cudaMalloc((void **) &totals_d, (n + 1) * sizeof(float));

#ifdef DEBUG
  printf("Copying to device..\n");
#endif
  cudaMemcpy(returns_d, returns, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(totals_d, totals, (n + 1) * sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 32; // should always be multiple of 32, max 1024
  int block_no = ceil(float(n) / block_size);
  dim3 grid(block_no, 1, 1);
  dim3 block(block_size, 1, 1); // max block dimensions: [1024,1024,64]

#ifdef DEBUG
  printf("block_no: %d, block_size: %d | \n", block_no, block_size);
  clock_t start_d = clock();
#endif
  __many_updates_gpu_kernel<<<grid, block>>>(returns_d, totals_d, n);
  cudaDeviceSynchronize();

#ifdef DEBUG
  clock_t end_d = clock();
  double time_d = (double) (end_d - start_d) / CLOCKS_PER_SEC;
  printf("GPU time: %f\n", time_d);
#endif

  cudaMemcpy(totals, totals_d, (n + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(returns_d);
  cudaFree(totals_d);
}

////////////////////////////////
__global__ void mc_simulations_gpu_kernel(float *historical_returns,
                                          const int n_historical_returns,
                                          float *totals,
                                          const int max_n_simulations,
                                          const int n_periods){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_n_simulations)
    return;

  // http://ianfinlayson.net/class/cpsc425/notes/cuda-random
  curandState_t state;
  curand_init(id, /* the seed can be the same for each core, here we pass the time in from the CPU */
              id, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  // every step, sample random return and update total
  for (int i = 0; i < n_periods; i++) {
    int return_idx = int(n_historical_returns * curand_uniform(&state));
    int total_idx = id * (n_periods + 1) + i;
    update_fund(totals[total_idx], historical_returns[return_idx], totals[total_idx+1]);
//    printf("%f -> %f\n", this_return, totals[id+i+1]);
  }
}

void _mc_simulations_gpu(float *historical_returns,
                         const int n_historical_returns,
                         float *totals,
                         const int max_n_simulations,
                         const int n_periods) {

  int block_size = 1024;
  int n_blocks = std::ceil(max_n_simulations / float(block_size));
  dim3 grid(n_blocks, 1, 1);
  dim3 block(block_size, 1, 1); // max block dimensions: [1024,1024,64]
  printf("block_no: %d, block_size: %d | \n", n_blocks, block_size);
  // TODO store historical_returns and/or totals in shared_memory?

  //allocations
  float *historical_returns_d, *totals_d;
  int memsize_hist_returns = n_historical_returns * sizeof(float);
  int memsize_totals = max_n_simulations * (n_periods+1) * sizeof(float);

  cudaMalloc((void **) &historical_returns_d, memsize_hist_returns);
  cudaMalloc((void **) &totals_d, memsize_totals);

  cudaMemcpy(historical_returns_d, historical_returns, memsize_hist_returns, cudaMemcpyHostToDevice);
  cudaMemcpy(totals_d, totals, memsize_totals, cudaMemcpyHostToDevice);

  mc_simulations_gpu_kernel<<<grid, block>>>(historical_returns_d,
                                             n_historical_returns,
                                             totals_d,
                                             max_n_simulations,
                                             n_periods);
  cudaDeviceSynchronize();
  cudaMemcpy(totals, totals_d, memsize_totals, cudaMemcpyDeviceToHost);
  cudaFree(historical_returns_d);
  cudaFree(totals_d);
}
