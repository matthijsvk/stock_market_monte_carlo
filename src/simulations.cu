#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda.h"

//#define DEBUG

// efficient random numbers on GPU
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
// S1, S2, S3, and M are all constants, and z is part of the    // private
// per-thread generator state.
__device__ __inline__ unsigned TausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}
__device__ __inline__ unsigned LCGStep(unsigned int &z, unsigned int A, unsigned int C) {
  return z = (A * z + C);
}
__device__ __inline__ float HybridTaus(unsigned int &z1,
                            unsigned int &z2,
                            unsigned int &z3,
                            unsigned int &z4) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^
      TausStep(z2, 2, 25, 4, 4294967288UL) ^
      TausStep(z3, 3, 11, 17, 4294967280UL) ^
      LCGStep(z4, 1664525, 1013904223UL));
}

__device__ __inline__ float HybridTausSimple(unsigned int &z1,
                                  unsigned int &z2) {
  // Combined period is lcm(p1,p2)~ 2^60
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^ TausStep(z2, 2, 25, 4, 4294967288UL));
}

__device__ __inline__ float HybridTausSimplest(unsigned int &z1) {
  // Combined period is lcm(p1,p2)~ 2^30
  return float(2.3283064365387e-10) * TausStep(z1, 13, 19, 12, 4294967294UL);
}

__global__ void testRNG(int n) {
  unsigned int rstate[4];
  for (int i = 0; i < 4; i++) rstate[i] = i * 12371;

  for (int i = 0; i < n; i++) {
    printf("%f\t", HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]));
  }
}

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

//__global__ void mc_simulations_gpu_kernel_v1(
//    float *historical_returns,
//    const unsigned int n_returns,
//    float *totals,
//    const unsigned long max_n_simulations,
//    const unsigned int n_periods) {
//  // https://cvw.cac.cornell.edu/gpu/memory_arch
//  // threads in a block are on the same SM
//  // 32 threads are a warp
//  unsigned int bid = blockIdx.x;
//  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//  unsigned int step = gridDim.x * blockDim.x;
//  unsigned int wid = threadIdx.x / warpSize;
//
//  if (tid >= max_n_simulations) return;
//
//  // Initialise the RNG. From
//  // https://docs.nvidia.com/cuda/curand/device-api-overview.html#pseudorandom-sequences:
//  // "Sequences generated with the same seed and different sequence numbers will
//  // not have statistically correlated values."
//
//  // generate starting numbers for HybridTaus with curand
//  curandState localState;
//  curand_init(1234567, tid, 0, &localState);  // seed, sequence number, offset, state
//
//  for (unsigned int i = 0; i < n_periods; i++) {
//    unsigned int return_idx = n_returns * curand_uniform(&localState);
//    update_fund(totals[tid], historical_returns[return_idx], totals[tid]);
//    //    printf("%f -> %f\n", this_return, totals[id+i+1]);
//  }
//
//  if (tid % (max_n_simulations / 10) == 0)
//    printf("Simulation %d/%ld\n", tid, max_n_simulations);
//}

__global__ void mc_simulations_gpu_kernel(
    float *historical_returns,
    const unsigned int n_returns,
    float *totals,
    const unsigned long max_n_simulations,
    const unsigned int n_periods) {
  // https://cvw.cac.cornell.edu/gpu/memory_arch
  // threads in a block are on the same SM
  // 32 threads are a warp
  //  unsigned int bid = blockIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //  unsigned int step = gridDim.x * blockDim.x;
//  unsigned int wid = threadIdx.x / warpSize;

  if (tid >= max_n_simulations) {
    return;
  }

  // Initialise the RNG. From
  // https://docs.nvidia.com/cuda/curand/device-api-overview.html#pseudorandom-sequences:
  // "Sequences generated with the same seed and different sequence numbers will
  // not have statistically correlated values."

//  curandState localState;
//  curand_init(1234567, tid, 0, &localState);  // seed, sequence number, offset, state

//  // don't use curand b/c global memory
//  // use local LCG and Tausworthe generator within registers. Use dynamic init numbers to ensure diversity
//  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
  unsigned int rstate[] = {tid + 21701};//, blockIdx.x + 21701};//, 1297, threadIdx.x + 65537};

  __shared__ float buffer[1129];
  // TODO multiple threads to load, using coalesced memory accesses?
  // global memory reads: 8 floats (32 bytes)
  const unsigned int n_threads_loading = 1; //warpSize;
  if (threadIdx.x < n_threads_loading) {
//    // slip n_returns in N warpSize chunks, every thread loads N values
//    unsigned int start_idx = threadIdx.x * n_per_thread;
//    unsigned int end_idx = min(n_returns, (threadIdx.x + 1) * n_per_thread);
//    for (unsigned int i = start_idx; i < end_idx; i += 1) {
//      buffer[i] = historical_returns[i];
//    }
    for (unsigned int i = threadIdx.x; i < n_returns; i += n_threads_loading) {
      buffer[i] = historical_returns[i]; // * float(0.01);
    }
    //printf("\nLoaded to shmem!\n");
  }
  __syncthreads();

  //todo Sobol PRNG to avoid shmem bank conflicts? -> every thread different dimension to have variety
  // in this way equal load on all banks

  // we do all this locally, reading return from shmem
  for (unsigned int i = 0; i < n_periods; i++) {
//    unsigned int return_idx = n_returns * curand_uniform(&localState);
//    unsigned int return_idx = n_returns * HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]);
//    unsigned int return_idx = n_returns * HybridTausSimple(rstate[0], rstate[1]);
    unsigned int return_idx = n_returns * HybridTausSimplest(rstate[0]);
    totals[tid] = totals[tid] + totals[tid] * buffer[return_idx];
  }

//  if (tid % (max_n_simulations / 10) == 0)
//    printf("Simulation %d/%ld\n", tid, max_n_simulations);
}

void _mc_simulations_gpu(float *historical_returns,
                         const unsigned int n_historical_returns,
                         float *totals,
                         const unsigned long max_n_simulations,
                         const unsigned int n_periods) {
  struct cudaDeviceProp deviceProperties;
  struct cudaFuncAttributes funcAttributes;

  cudaGetDeviceProperties(&deviceProperties, 0);

  int block_size = 256;  // TODO how to set?
  dim3 block, grid;
  block.x = block_size;
  grid.x = (max_n_simulations + block_size - 1) / block_size;

  printf("block_no: %d, block_size: %d | warps/block: %d \n",
         grid.x,
         block.x,
         block.x / 32);
  //-----------------------------------------------------------

  //  cudaFuncGetAttributes(&funcAttributes, initRNG);
  //  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
  //    throw std::runtime_error(
  //        "Block X dimension is too large for initRNG kernel");
  //  }

  cudaFuncGetAttributes(&funcAttributes, mc_simulations_gpu_kernel);
  if (block.x > (unsigned int) funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for mc_simulations_gpu_kernel");
  }

  // Check the dimensions are valid
  if (block.x > (unsigned int) deviceProperties.maxThreadsDim[0]) {
    throw std::runtime_error("Block X dimension is too large for device");
  }

  if (grid.x > (unsigned int) deviceProperties.maxGridSize[0]) {
    throw std::runtime_error("Grid X dimension is too large for device");
  }

  // allocations
  float *historical_returns_d, *totals_d;
  long memsize_hist_returns = n_historical_returns * sizeof(float);
  long memsize_totals = max_n_simulations * sizeof(float);

  cudaMalloc(&historical_returns_d, memsize_hist_returns);
  cudaMalloc(&totals_d, memsize_totals);
  cudaMemcpy(historical_returns_d,
             historical_returns,
             memsize_hist_returns,
             cudaMemcpyHostToDevice);
  cudaMemcpy(totals_d, totals, memsize_totals, cudaMemcpyHostToDevice);

  // launch kernel!
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

void mc_simulations_gpu(std::atomic<unsigned long> &n_simulations,
                        const unsigned long max_n_simulations,
                        const unsigned int n_periods,
                        const float initial_capital,
                        std::vector<float> &historical_returns,
                        std::vector<float> &final_values) {
//  printf("%f", float(2^-32));

  // initialize output array
  std::fill(final_values.begin(), final_values.end(), initial_capital);
  // get pointers b/c GPU can't use std::vectors
  float *totals_arr = &final_values[0];

  // create array copy instead of float *historical_returns_arr = &historical_returns[0];
  // b/c we want to modify it (mult by 0.01)
  float historical_returns_arr[historical_returns.size()];
  for (unsigned int i = 0; i < historical_returns.size(); i += 1) {
    historical_returns_arr[i] = historical_returns[i] * float(0.01);
  }

  _mc_simulations_gpu(historical_returns_arr,
                      historical_returns.size(),
                      totals_arr,
                      max_n_simulations,
                      n_periods);

  n_simulations = max_n_simulations;  // TODO increment inside GPU kernel?
  // assert(n_simulations = max_n_simulations); // must be true here
}
