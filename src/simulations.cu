#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <vector>

#include "cuda.h"

//#define DEBUG

// RNG init kernel
__global__ void initRNG(curandState *const rngStates, const unsigned int seed) {
  // Determine thread ID
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialise the RNG
  curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ inline void update_fund(float fund_value,
                            float period_return,
                            float &next_value) {
  next_value = fund_value * (float(100.0) + period_return) / 100;
}


////////////////////////////////
__global__ void mc_simulations_gpu_kernel(float *historical_returns,
                                          const long n_historical_returns,
                                          float *totals,
                                          const long max_n_simulations,
                                          const long n_periods
//                                          curandState *const rngStates
                                          ) {
  // threads in a block are on the same SM
  unsigned int bid = blockIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = gridDim.x * blockDim.x;
  unsigned int wid = threadIdx.x / warpSize;

  if (tid >= max_n_simulations) return;

  if (tid % (max_n_simulations / 10) == 0)
    printf("Simulation %d/%ld\n", tid, max_n_simulations);

  // shmem buffer for historical returns, 1 warp in block loads it
  __shared__ float buffer[1200];
  if (threadIdx.x < n_historical_returns) {
    for(int i = threadIdx.x; i  < n_historical_returns; i += warpSize) {
      buffer[i] = historical_returns[i];
    }
  }
  __syncthreads();

  // Initialise the RNG
//  curandState localState = rngStates[tid];
    curandState localState;
    curand_init(1234, tid, 0, &localState);

  // every step, sample random return and update total
  for (unsigned int i = 0; i < n_periods; i++) {
    unsigned int return_idx = int(n_historical_returns * curand_uniform(&localState));
//    update_fund(totals[tid], historical_returns[return_idx], totals[tid]);
    update_fund(totals[tid], buffer[return_idx], totals[tid]);
    //    printf("%f -> %f\n", this_return, totals[id+i+1]);
  }
}

void _mc_simulations_gpu(float *historical_returns,
                         const long n_historical_returns,
                         float *totals,
                         const long max_n_simulations,
                         const long n_periods) {
  struct cudaDeviceProp deviceProperties;
  struct cudaFuncAttributes funcAttributes;

  cudaGetDeviceProperties(&deviceProperties, 0);

  int block_size = 256; // TODO how to set?
  dim3 block, grid;
  block.x = block_size;
  grid.x = (max_n_simulations + block_size - 1 ) / block_size;

  printf("block_no: %d, block_size: %d | warps/block: %d \n", grid.x, block.x, block.x / 32);
  //-----------------------------------------------------------

  cudaFuncGetAttributes(&funcAttributes, initRNG);
  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for initRNG kernel");
  }

  cudaFuncGetAttributes(&funcAttributes, mc_simulations_gpu_kernel);
  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for mc_simulations_gpu_kernel");
  }

  // Check the dimensions are valid
  if (block.x > (unsigned int)deviceProperties.maxThreadsDim[0]) {
    throw std::runtime_error("Block X dimension is too large for device");
  }

  if (grid.x > (unsigned int)deviceProperties.maxGridSize[0]) {
    throw std::runtime_error("Grid X dimension is too large for device");
  }

  //-----------------------------------------------------------

//  curandState *d_rngStates = 0;
//  cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));

  // allocations
  float *historical_returns_d, *totals_d;
  long memsize_hist_returns = n_historical_returns * sizeof(float);
  long memsize_totals = max_n_simulations * sizeof(float);

  cudaMalloc(&historical_returns_d, memsize_hist_returns);
  cudaMalloc(&totals_d, memsize_totals);
  cudaMemcpy(historical_returns_d, historical_returns, memsize_hist_returns, cudaMemcpyHostToDevice);
  cudaMemcpy(totals_d, totals, memsize_totals, cudaMemcpyHostToDevice);

//  // Initialise RNG
//  initRNG<<<grid, block>>>(d_rngStates, 12345); // TODO seed?
  // launch kernel!
  mc_simulations_gpu_kernel<<<grid, block>>>(historical_returns_d,
                                             n_historical_returns,
                                             totals_d,
                                             max_n_simulations,
                                             n_periods);
  cudaDeviceSynchronize();
  cudaMemcpy(totals, totals_d, memsize_totals, cudaMemcpyDeviceToHost);
//  cudaFree(d_rngStates);
  cudaFree(historical_returns_d);
  cudaFree(totals_d);
}

void mc_simulations_gpu(std::atomic<long> &n_simulations,
                        const long max_n_simulations,
                        const long n_periods,
                        const float initial_capital,
                        std::vector<float> &historical_returns,
                        std::vector<float> &final_values) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  // initialize output array
  std::fill(final_values.begin(), final_values.end(), initial_capital);
  // get pointers b/c GPU can't use std::vectors
  float *totals_arr = &final_values[0];
  float *historical_returns_arr = &historical_returns[0];

  _mc_simulations_gpu(historical_returns_arr,
                      historical_returns.size(),
                      totals_arr,
                      max_n_simulations,
                      n_periods);

  n_simulations = max_n_simulations;  // TODO increment inside GPU kernel?

  // assert(n_simulations = max_n_simulations); // must be true here

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  fmt::print(
      "All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);
}
