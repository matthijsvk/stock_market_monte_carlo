#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cuda.h"

//#define DEBUG
#define THREADS_PER_BLOCK 256

//==================================================================================
// CUDA helper functions
//==================================================================================

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

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
// https://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
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

__global__ void mc_simulations_gpu_kernel(const float *__restrict__ returns,
                                          const int n_returns,
                                          float *__restrict__ totals,
                                          const long N,
                                          float initial_capital,
                                          const int n_periods) {
  // https://cvw.cac.cornell.edu/gpu/memory_arch
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  // first warp loads returns to shmem
  __shared__ float bufferReturns[1127];
  if (threadIdx.x < warpSize) {
    for (int i = threadIdx.x; i < n_returns; i += warpSize) {
      bufferReturns[i] = returns[i] * float(0.01);
    }
  }
  __syncthreads();

  // todo Sobol PRNG to avoid shmem bank conflicts?
  // https://github.com/NVIDIA/cuda-samples/tree/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/5_Domain_Specific/SobolQRNG

  // hase tid to get good starting seed
  unsigned int prng_state = rand_pcg(tid + 1);

  float total = initial_capital;
  int return_idx;
  for (int i = 0; i < n_periods; i++) {
    prng_state = xorshift(prng_state);
    return_idx = n_returns * (prng_state * float(2.3283064e-10));  // powf(2, -32));
    total += total * bufferReturns[return_idx];
  }
  totals[tid] = total;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <int blockSize>
__global__ void mc_simulations_gpu_kernel_reduceBlock(const float *__restrict__ returns,
                                                      const int n_returns,
                                                      float *__restrict__ totals,
                                                      float *__restrict__ variances,
                                                      const long N,
                                                      float initial_capital,
                                                      const int n_periods) {
  __shared__ float bufferReturns[1127];  // todo n_returns??
  __shared__ double s_totals[blockSize];
  __shared__ double s_totals_var[blockSize];  // keep copy b/c we're reducing

  // TODO debug: build/benchmark_mc_gpu_reduceBlock 1 360 425220

  // https://cvw.cac.cornell.edu/gpu/memory_arch
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid >= N) return;

  // first warp loads returns to shmem
  if (threadIdx.x < warpSize) {
    for (int i = threadIdx.x; i < n_returns; i += warpSize) {
      bufferReturns[i] = returns[i] * float(0.01);
    }
  }
  __syncthreads();

  // todo Sobol PRNG to avoid shmem bank conflicts?
  // https://github.com/NVIDIA/cuda-samples/tree/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/5_Domain_Specific/SobolQRNG

  // https://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
  unsigned int prng_state = rand_pcg(global_tid + 1);

  float total = initial_capital;
  int return_idx;
  for (int i = 0; i < n_periods; i++) {
    prng_state = xorshift(prng_state);
    return_idx = n_returns * (prng_state * float(2.3283064e-10));  // powf(2, -32));
    total += total * bufferReturns[return_idx];
  }
  s_totals[threadIdx.x] = float(total);
  s_totals_var[threadIdx.x] = float(total);
  __syncthreads();

  //  // compute mean and variance in a single iteration
  //  // https://www.johndcook.com/blog/standard_deviation/
  //  if (threadIdx.x == 0) {
  //    double m = s_totals[0], prev_m = s_totals[0];
  //    double s = 0, prev_s = 0;
  //    for (int i = 1; i < blockSize; i++) {
  //      double x = s_totals[i];
  //      m = prev_m + (x - prev_m) / i;
  //      s = prev_s + (x - prev_m) * (x - m);
  //      // set up for next iteration
  //      prev_m = m;
  //      prev_s = s;
  //    }
  //    float mean = float(m);
  //    float var = float(s) / (blockSize - 1);
  //    totals[blockIdx.x] = mean;
  //    variances[blockIdx.x] = var;
  //  }

  // calc mean using the shmem buffer: hierarchical reduction
  // https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
  // reduction is destructive, so need to do everything twice to have copy we can use for variance computation!
  unsigned int tid = threadIdx.x;
  if (blockSize >= 512) {
    if (tid < 256) s_totals[tid] += s_totals[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) s_totals[tid] += s_totals[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) s_totals[tid] += s_totals[tid + 64];
    __syncthreads();
  }
  if (tid < 32) warpReduce<blockSize>(s_totals, tid);
  __syncthreads();

  // now compute variance
  float mean = s_totals[0] / blockSize;
  s_totals_var[tid] -= mean;
  s_totals_var[tid] = s_totals_var[tid] * s_totals_var[tid];
  // and reduce
  if (blockSize >= 512) {
    if (tid < 256) s_totals_var[tid] += s_totals_var[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) s_totals_var[tid] += s_totals_var[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) s_totals_var[tid] += s_totals_var[tid + 64];
    __syncthreads();
  }
  if (tid < 32) warpReduce<blockSize>(s_totals_var, tid);
  __syncthreads();

  if (tid == 0) {
    float var = s_totals_var[0] / blockSize;
    totals[blockIdx.x] = mean;
    variances[blockIdx.x] = var;
//    printf("block %d mean= %f | var=%f\n", blockIdx.x, mean, var);
  }
}

//==================================================================================
// single GPU launcher
//==================================================================================
void mc_simulations_gpu_launcher(std::vector<float> &returns_vec,
                                 const int n_returns,
                                 std::vector<float> &totals_vec,
                                 const long N,
                                 float initial_value,
                                 const int n_periods) {
  // initialize data structures | Reserve is cheaper than resize! esp for large arrays
  totals_vec.resize(N, initial_value);
  // get pointers b/c GPU can't use std::vectors
  float *totals = &totals_vec[0];
  float *returns = &returns_vec[0];

  int block_size = THREADS_PER_BLOCK;
  dim3 block, grid;
  block.x = block_size;
  grid.x = (N + block_size - 1) / block_size;

  printf("block_no: %d, block_size: %d | warps/block: %d \n", grid.x, block.x, block.x / 32);

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
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  mc_simulations_gpu_kernel<<<grid, block>>>(returns_d, n_returns, totals_d, N, initial_value, n_periods);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("GPU kernel itself took {} s!\n", timediff / 1000.0);

  cudaMemcpy(totals, totals_d, memsize_totals, cudaMemcpyDeviceToHost);
  cudaFree(returns_d);
  cudaFree(totals_d);

  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
  auto timediff2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin).count();
  fmt::print("GPU kernel + transfer to host took {} s!\n", timediff2 / 1000.0);
}

void mc_simulations_gpu_reduceBlock_launcher(std::vector<float> &returns_vec,
                                             const int n_returns,
                                             std::vector<float> &means_vec,
                                             std::vector<float> &variances_vec,
                                             const long N,
                                             float initial_value,
                                             const int n_periods) {
  const int block_size = THREADS_PER_BLOCK;
  dim3 block, grid;
  block.x = block_size;
  grid.x = (N + block_size - 1) / block_size;
  unsigned int n_blocks = grid.x;
  printf("block_no: %d, block_size: %d | warps/block: %d \n", grid.x, block.x, block.x / 32);

  // initialize data structures
  means_vec.resize(n_blocks);
  std::fill(means_vec.begin(), means_vec.end(), 0.0);
  variances_vec.resize(n_blocks);
  std::fill(variances_vec.begin(), variances_vec.end(), 0.0);

  // get pointers b/c GPU can't use std::vectors
  float *returns = &returns_vec[0];
  float *means = &means_vec[0];
  float *variances = &variances_vec[0];

  //-----------------------
  // Memory allocations
  //----------------------
  float *returns_d, *means_d, *variances_d;
  long memsize_hist_returns = n_returns * sizeof(float);
  long memsize_means = n_blocks * sizeof(float);
  long memsize_variances = n_blocks * sizeof(float);

  gpuErrchk(cudaMalloc(&returns_d, memsize_hist_returns));
  gpuErrchk(cudaMalloc(&means_d, memsize_means));
  gpuErrchk(cudaMalloc(&variances_d, memsize_variances));

  gpuErrchk(cudaMemcpy(returns_d, returns, memsize_hist_returns, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(means_d, means, memsize_means, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(variances_d, variances, memsize_variances, cudaMemcpyHostToDevice));

  // launch kernel!
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  mc_simulations_gpu_kernel_reduceBlock<block_size>
      <<<grid, block>>>(returns_d, n_returns, means_d, variances_d, N, initial_value, n_periods);
  gpuErrchk(cudaDeviceSynchronize());

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("GPU kernel itself took {} s!\n", timediff / 1000.0);

  gpuErrchk(cudaMemcpy(means, means_d, memsize_means, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(variances, variances_d, memsize_variances, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(returns_d));
  gpuErrchk(cudaFree(means_d));
  gpuErrchk(cudaFree(variances_d));
  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
  auto timediff2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin).count();
  fmt::print("GPU kernel + transfer to host took {} s!\n", timediff2 / 1000.0);
}

//==================================================================================
// multi-GPU
//==================================================================================

struct Plan {
  long n;
  float *returns_d;
  float *totals_d;
};

void create_plan(Plan &plan, int gpu_id, long N, int n_returns) {
  cudaSetDevice(gpu_id);
  plan.n = N;
  gpuErrchk(cudaMalloc(&(plan.returns_d), n_returns * sizeof(float)));
  gpuErrchk(cudaMalloc(&(plan.totals_d), N * sizeof(float)));
}

void mc_simulations_multi_gpu_launcher_v1(std::vector<float> &returns_vec,
                                          const int n_returns,
                                          std::vector<float> &totals_vec,
                                          const long N,
                                          float initial_value,
                                          const int n_periods,
                                          int n_gpus) {
  // TODO see https://stackoverflow.com/questions/11673154/concurrency-in-cuda-multi-gpu-executions/35010019#35010019
  // initialize data structures
  totals_vec.resize(N, initial_value);
  // get pointers b/c GPU can't use std::vectors
  float *totals = &totals_vec[0];
  float *returns = &returns_vec[0];

  //-----------------------
  printf("Allocating memory...");
  long n_todo = N;
  Plan plan[n_gpus];
  for (int dev = 0; dev < n_gpus; dev++) {
    printf("\tgpu %d", dev);
    long n_this_gpu = std::min(n_todo, N / n_gpus);
    n_todo = N - n_this_gpu;
    // allocate memory on the correct GPU device
    printf("-> will run %ld simulations", n_this_gpu);
    create_plan(plan[dev], dev, n_this_gpu, n_returns);
  }
  printf("\n");
  //----------------------
  printf("Transferring data ...");
  long n_done = 0;
  for (int dev = 0; dev < n_gpus; dev++) {
    printf("\tgpu %d", dev);
    gpuErrchk(cudaSetDevice(dev));
    gpuErrchk(cudaMemcpy(plan[dev].returns_d, returns, n_returns * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(plan[dev].totals_d, totals + n_done, plan[dev].n * sizeof(float), cudaMemcpyHostToDevice));
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
        plan[dev].returns_d, n_returns, plan[dev].totals_d, plan[dev].n, initial_value, n_periods);
  }
  printf("\n");
  gpuErrchk(cudaDeviceSynchronize());
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
  gpuErrchk(cudaDeviceReset());
}

struct Plan_v2 {
  long n;
  float *returns_d;
  //  float *returns_h;
  float *totals_d;
  //  float *totals_h;
  //  cudaStream_t    stream;
};

void create_plan_v2(Plan_v2 &plan, int gpu_id, long N, int n_returns) {
  cudaSetDevice(gpu_id);
  plan.n = N;
  gpuErrchk(cudaMalloc(&(plan.returns_d), n_returns * sizeof(float)));
  gpuErrchk(cudaMalloc(&(plan.totals_d), N * sizeof(float)));
  //  gpuErrchk(cudaStreamCreate(&plan.stream));
}

void mc_simulations_multi_gpu_launcher_async(std::vector<float> &returns_vec,
                                             const int n_returns,
                                             std::vector<float> &totals_vec,
                                             const long N,
                                             float initial_value,
                                             const int n_periods,
                                             int n_gpus) {
  // TODO see https://stackoverflow.com/questions/11673154/concurrency-in-cuda-multi-gpu-executions/35010019#35010019
  // initialize data structures
  totals_vec.resize(N, initial_value);
  // get pointers b/c GPU can't use std::vectors
  float *totals = &totals_vec[0];
  float *returns = &returns_vec[0];

  //-----------------------
  printf("Allocating memory...");
  long n_todo = N;
  Plan_v2 plan[n_gpus];
  for (int dev = 0; dev < n_gpus; dev++) {
    printf("\tgpu %d", dev);
    long n_this_gpu = std::min(n_todo, N / n_gpus);
    n_todo = N - n_this_gpu;
    // allocate memory on the correct GPU device
    printf("-> will run %ld simulations", n_this_gpu);
    create_plan_v2(plan[dev], dev, n_this_gpu, n_returns);
  }
  printf("\n");

  // allocate on host for pinned memory so we can write back asynchronously
  gpuErrchk(cudaMallocHost(&returns, n_returns * sizeof(float)));
  gpuErrchk(cudaMallocHost(&totals, N * sizeof(float)));  // todo how to reuse the already allocated 'totals'??

  dim3 block, grid;
  int block_size = THREADS_PER_BLOCK;
  long n_done = 0;
  // we're using default CUDA stream, example 5 in
  //  https://stackoverflow.com/questions/11673154/concurrency-in-cuda-multi-gpu-executions/35010019#35010019
  for (int k = 0; k < n_gpus; k++) {
    gpuErrchk(cudaSetDevice(k));
    gpuErrchk(cudaMemcpyAsync(plan[k].returns_d, returns, n_returns * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyAsync(plan[k].totals_d, totals + n_done, plan[k].n * sizeof(float), cudaMemcpyHostToDevice));
    block.x = block_size;
    grid.x = iDivUp(plan[k].n, block_size);
    mc_simulations_gpu_kernel<<<grid, block>>>(
        plan[k].returns_d, n_returns, plan[k].totals_d, plan[k].n, initial_value, n_periods);
    gpuErrchk(cudaMemcpyAsync(totals + n_done, plan[k].totals_d, plan[k].n * sizeof(float), cudaMemcpyDeviceToHost));
    n_done += plan[k].n;
  }
  gpuErrchk(cudaDeviceReset());  // TODO does this reset all gpus? what's the duration?
}

//==================================================================================
// CPP callable function
//==================================================================================

void mc_simulations_gpu(std::atomic<long> &n_simulations,
                        const long N,
                        const int n_periods,
                        const float initial_capital,
                        std::vector<float> &returns,
                        std::vector<float> &totals,
                        const int n_gpus) {
  //  mc_simulations_multi_gpu_launcher_async(returns, returns.size(), totals, N, initial_capital, n_periods, n_gpus);
  if (n_gpus == 1) {
    mc_simulations_gpu_launcher(returns, returns.size(), totals, N, initial_capital, n_periods);
  } else if (n_gpus > 1) {
    mc_simulations_multi_gpu_launcher_async(returns, returns.size(), totals, N, initial_capital, n_periods, n_gpus);
  } else {
    throw std::invalid_argument("INVALID NUMBER OF GPUS SPECIFIED!!");
  }

  n_simulations = N;  // TODO increment inside GPU kernel?
  // assert(n_simulations = N); // must be true here
}

void mc_simulations_gpu_reduceBlock(std::atomic<long> &n_simulations,
                                    const long N,
                                    const int n_periods,
                                    const float initial_capital,
                                    std::vector<float> &returns,
                                    std::vector<float> &means,
                                    std::vector<float> &variances,
                                    const int n_gpus) {
  if (n_gpus == 1) {
    mc_simulations_gpu_reduceBlock_launcher(returns, returns.size(), means, variances, N, initial_capital, n_periods);
  } else {
    throw std::invalid_argument(">1 GPU not yet supported for reduceBlock version!!");
  }
  n_simulations = N;  // TODO increment inside GPU kernel?
  // assert(n_simulations = N); // must be true here
}