#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#include "cuda.h"

//#define DEBUG
#define THREADS_PER_BLOCK 256

//==================================================================================
// CUDA helper functions
//==================================================================================

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//==================================================================================
// fast GPU PRNG functions
//==================================================================================
// efficient random numbers on GPU
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

__device__ __inline__ unsigned TausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}
__device__ __inline__ unsigned LCGStep(unsigned int &z, unsigned int A, unsigned int C) { return z = (A * z + C); }
__device__ __inline__ float HybridTaus(unsigned int &z1, unsigned int &z2, unsigned int &z3, unsigned int &z4) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^ TausStep(z2, 2, 25, 4, 4294967288UL) ^
                                       TausStep(z3, 3, 11, 17, 4294967280UL) ^ LCGStep(z4, 1664525, 1013904223UL));
}

__device__ __inline__ float HybridTausSimple(unsigned int &z1, unsigned int &z2) {
  // Combined period is lcm(p1,p2)~ 2^60
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^ TausStep(z2, 2, 25, 4, 4294967288UL));
}

__device__ __inline__ float HybridTausSimplest(unsigned int &z1) {
  // Combined period is lcm(p1,p2)~ 2^30
  return float(2.3283064365387e-10) * TausStep(z1, 13, 19, 12, 4294967294UL);
}

//-------------------------------------------------------------------------

__device__ __inline__ unsigned int pcg_hash(unsigned int input) {
  unsigned int state = input * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ __inline__ unsigned int rand_pcg(unsigned int rng_state) {
  unsigned int state = rng_state;
  rng_state = rng_state * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ __inline__ unsigned int xorshift(unsigned int y) {
  // Liao et al 2020 SAGC "A 23.8Tbps Random Number Generator on a Single GPU"
  // https://github.com/L4Xin/quadruples-xorshift/
  y = y ^ (y << 11);
  y = y ^ (y >> 7);
  return y ^ (y >> 12);
}

__device__ __inline__ unsigned int xorshiftGM(unsigned int rng_state) {
  // Xorshift algorithm from George Marsaglia's paper
  rng_state ^= (rng_state << 13);
  rng_state ^= (rng_state >> 17);
  rng_state ^= (rng_state << 5);
  return rng_state;
}

__device__ __inline__ unsigned int xorshf96(unsigned int x) {
  unsigned int y = 362436069, z = 521288629;  // period 2^96-1
  unsigned int t;
  x ^= x << 16;
  x ^= x >> 5;
  x ^= x << 1;

  t = x;
  x = y;
  y = z;
  z = t ^ x ^ y;

  return z;
}

__device__ void testRNG(int n) {
  unsigned int rstate[4];
  for (int i = 0; i < 4; i++) rstate[i] = i * 12371;

  for (int i = 0; i < n; i++) {
    printf("%f\t", HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]));
  }
}

//==================================================================================
// actual MC gpu kernel
//==================================================================================

__global__ void mc_simulations_gpu_kernel(
    float *returns, const unsigned int n_returns, float *totals, const unsigned long N, const unsigned int n_periods) {
  // https://cvw.cac.cornell.edu/gpu/memory_arch
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N) return;
//  printf("Hello from thread %d!\n", tid);

  //  // don't use curand b/c global memory
  //  https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
  unsigned int rstate[] = {rand_pcg(tid), 21701, 1297, 65537};

  // first warp loads returns to shmem
  __shared__ float bufferReturns[1129];
  if (threadIdx.x < warpSize) {
    for (unsigned int i = threadIdx.x; i < n_returns; i += warpSize) {
      bufferReturns[i] = returns[i] * float(0.01);
    }
  }
  //  // second warp loads totals to shmem
  //  __shared__ float bufferTotals[THREADS_PER_BLOCK];  // should be blockDim.x
  //  if (threadIdx.x >= warpSize && threadIdx.x < 2 * warpSize) {
  //    for (unsigned int i = threadIdx.x - warpSize; i < THREADS_PER_BLOCK; i += warpSize) {
  //      //  if (threadIdx.x < warpSize) {
  //      //    for (unsigned int i = threadIdx.x; i < THREADS_PER_BLOCK; i +=
  //      //    warpSize) {
  //      bufferTotals[i] = totals[blockIdx.x * blockDim.x + i];
  //    }
  //  }
  __syncthreads();
  float total = totals[tid];
  //  float total = bufferTotals[threadIdx.x];

  // todo Sobol PRNG to avoid shmem bank conflicts?
  // https://github.com/NVIDIA/cuda-samples/tree/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/5_Domain_Specific/SobolQRNG

  // https://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
  unsigned int prng_state = rand_pcg(tid);

  unsigned int return_idx;
  for (unsigned int i = 0; i < n_periods; i++) {
    //    return_idx = n_returns * HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]);
    //        return_idx = n_returns_f * HybridTausSimple(rstate[0], rstate[1]);
    prng_state = xorshift(prng_state);
    return_idx = n_returns * (prng_state * powf(2, -32));
    total += total * bufferReturns[return_idx];
  }
  totals[tid] = total;
  //  bufferTotals[threadIdx.x] = total;
  //  __syncthreads();
  //
  //  // write results from this block to global memory, coalesced
  //  if (threadIdx.x < warpSize) {
  //    for (unsigned int i = threadIdx.x; i < blockDim.x; i += warpSize) {
  //      totals[blockIdx.x * blockDim.x + i] = bufferTotals[i];
  //    }
  //  }
}

struct Plan {
  unsigned long n;
  float *returns_d;
  float *totals_d;
};

void create_plan(Plan &plan, int gpu_id, unsigned long N, unsigned int n_returns) {
  cudaSetDevice(gpu_id);
  plan.n = N;
  gpuErrchk(cudaMalloc(&(plan.returns_d), n_returns * sizeof(float)));
  gpuErrchk(cudaMalloc(&(plan.totals_d), N * sizeof(float)));
}


void _mc_simulations_multi_gpu(float *returns,
                               const unsigned int n_returns,
                               float *totals,
                               const unsigned long N,
                               const unsigned int n_periods,
                               unsigned int n_gpus) {
  // TODO see https://stackoverflow.com/questions/11673154/concurrency-in-cuda-multi-gpu-executions/35010019#35010019

  //-----------------------
  printf("Allocating memory...");
  unsigned long n_todo = N;
  Plan plan[n_gpus];
  for (int dev = 0; dev < n_gpus; dev++) {
    printf("\tgpu %d", dev);
    unsigned long n_this_gpu = std::min(n_todo, N / n_gpus);
    n_todo = N - n_this_gpu;
    // allocate memory on the correct GPU device
    printf("-> will run %ld simulations", n_this_gpu);
    create_plan(plan[dev], dev, n_this_gpu, n_returns);
  }
  printf("\n");
  //----------------------
  printf("Transferring data ...");
  unsigned long n_done = 0;
  for (int dev = 0; dev < n_gpus; dev++) {
    printf("\tgpu %d", dev);
    cudaSetDevice(dev);
    cudaMemcpy(plan[dev].returns_d, returns, n_returns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(plan[dev].totals_d, totals + n_done, plan[dev].n * sizeof(float), cudaMemcpyHostToDevice);
    n_done += dev * plan[dev].n;
  }
  printf("\n");
  //----------------------
  printf("Launching kernels...\n");
  for (int dev = 0; dev < n_gpus; dev++) {
    cudaSetDevice(dev);
    dim3 block, grid;
    int block_size = THREADS_PER_BLOCK;
    block.x = block_size;
    grid.x = (plan[dev].n + block_size - 1) / block_size;
    printf("\t GPU %d -> block_no: %d, block_size: %d | warps/block: %d \n", dev, grid.x, block.x, block.x / 32);
    mc_simulations_gpu_kernel<<<grid, block>>>(
        plan[dev].returns_d, n_returns, plan[dev].totals_d, plan[dev].n, n_periods);
  }
  printf("\n");
  cudaDeviceSynchronize();
  //----------------------
  printf("Gathering results...");
  n_done = 0;
  for (int dev = 0; dev < n_gpus; dev++) {
    printf("\tgpu %d", dev);
    cudaSetDevice(dev);
    cudaMemcpy(totals + n_done, plan[dev].totals_d, plan[dev].n * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(plan[dev].returns_d);
//    cudaFree(plan[dev].totals_d);
    n_done += plan[dev].n;
  }
  printf("\n");
  cudaDeviceReset();
}

//void _mc_simulations_multi_gpu(float *returns,
//                               const unsigned int n_returns,
//                               float *totals,
//                               const unsigned long N,
//                               const unsigned int n_periods,
//                               int n_gpus) {
//  // TODO see https://stackoverflow.com/questions/11673154/concurrency-in-cuda-multi-gpu-executions/35010019#35010019
//
//  int dev = 0;
//  //-----------------------
//  printf("Allocating memory...");
//  unsigned long n_todo = N;
//  Plan plan[n_gpus];
//  create_plan(plan[dev], dev, N, n_returns);
//  printf("\n");
//
//  //----------------------
//  printf("Transferring data ...");
//  unsigned long n_done = 0;
//  gpuErrchk(cudaSetDevice(dev));
//  gpuErrchk(cudaMemcpy(plan[dev].returns_d, returns, n_returns * sizeof(float), cudaMemcpyHostToDevice));
//  gpuErrchk(cudaMemcpy(plan[dev].totals_d, totals, plan[dev].n * sizeof(float), cudaMemcpyHostToDevice));
//  n_done += plan[dev].n;
//  printf("\n");
//
//  //----------------------
//  printf("Launching kernels...\n");
//  gpuErrchk(cudaSetDevice(dev));
//  dim3 block, grid;
//  int block_size = THREADS_PER_BLOCK;
//  block.x = block_size;
//  grid.x = (plan[dev].n + block_size - 1) / block_size;
//  printf("\t GPU %d -> block_no: %u, block_size: %d | warps/block: %d \n", dev, grid.x, block.x, block.x / 32);
//  mc_simulations_gpu_kernel<<<grid, block>>>(
//      plan[dev].returns_d, n_returns, plan[dev].totals_d, plan[dev].n, n_periods);
//  printf("\n");
//  gpuErrchk(cudaDeviceSynchronize());
//
//  //----------------------
//  printf("Gathering results...");
//  n_done = 0;
//  printf("\tgpu %d", dev);
//  gpuErrchk(cudaSetDevice(dev));
//  gpuErrchk(cudaMemcpy(totals + n_done, plan[dev].totals_d, plan[dev].n * sizeof(float), cudaMemcpyDeviceToHost));
////  cudaFree(plan[dev].returns_d);
////  cudaFree(plan[dev].totals_d);
////  n_done += plan[dev].n;
//  printf("Done!\n");
////  std::this_thread::sleep_for(std::chrono::milliseconds(5 * 1000));;
//  gpuErrchk(cudaDeviceReset());
//}

void _mc_simulations_gpu(
    float *returns, const unsigned int n_returns, float *totals, const unsigned long N, const unsigned int n_periods) {
  int block_size = THREADS_PER_BLOCK;
  dim3 block, grid;
  block.x = block_size;
  grid.x = (N + block_size - 1) / block_size;

  printf("block_no: %d, block_size: %d | warps/block: %d \n", grid.x, block.x, block.x / 32);

  cudaSetDevice(0);
  //-----------------------
  // Memory allocations
  //----------------------
  float *returns_d, *totals_d;
  long memsize_hist_returns = n_returns * sizeof(float);
  long memsize_totals = N * sizeof(float);


  cudaMalloc(&returns_d, memsize_hist_returns);
  cudaMalloc(&totals_d, memsize_totals);

  cudaMemcpy(returns_d, returns, memsize_hist_returns, cudaMemcpyHostToDevice);
  cudaMemcpy(totals_d, totals, memsize_totals, cudaMemcpyHostToDevice);

  // launch kernel!
  mc_simulations_gpu_kernel<<<grid, block>>>(returns_d, n_returns, totals_d, N, n_periods);

  cudaDeviceSynchronize();
  cudaMemcpy(totals, totals_d, memsize_totals, cudaMemcpyDeviceToHost);
  cudaFree(returns_d);
  cudaFree(totals_d);
}

void mc_simulations_gpu(std::atomic<unsigned long> &n_simulations,
                        const unsigned long N,
                        const unsigned int n_periods,
                        const float initial_capital,
                        std::vector<float> &returns,
                        std::vector<float> &final_values) {
  // initialize output array
  std::fill(final_values.begin(), final_values.end(), initial_capital);
  // get pointers b/c GPU can't use std::vectors
  float *totals_arr = &final_values[0];
  float *returns_arr = &returns[0];

//    _mc_simulations_gpu(returns_arr, returns.size(), totals_arr, N, n_periods);
  _mc_simulations_multi_gpu(returns_arr, returns.size(), totals_arr, N, n_periods, 1);

  n_simulations = N;  // TODO increment inside GPU kernel?
  // assert(n_simulations = N); // must be true here
}
