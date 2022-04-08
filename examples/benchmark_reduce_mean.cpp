#include <fmt/core.h>

#include <atomic>
#include <locale>
#include "stock_market_monte_carlo/simulations.h"


int main(int argc, char *argv[]) {
  std::locale::global(std::locale("en_US.UTF-8"));
  fmt::print("argc: {}\n", argc);
  long n;
  if (argc == 2) {
    char *end;
    n = long(std::strtol(argv[1], &end, 10));
    fmt::print("n: {}\n", n);
  } else {
    fmt::print("usage: compute_avg <n>");
    exit(0);
  }
  auto begin_setup = std::chrono::steady_clock::now();
  std::vector<float> vec(n);
  for (int i=0; i<n; i++)
    vec[i] = float(i);
  auto end_setup = std::chrono::steady_clock::now();
  auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end_setup - begin_setup).count();
  fmt::print("setup of vector took {} s!\n", timediff / 1000.0);

  // verify with CPU computation
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  float mean_cpu = float(sum) / vec.size();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("CPU took {} s!\n", timediff / 1000.0);

  // TODO why does is GPU time reported here so much less than inside reduce_mean_gpu function??? it's just a function call..
  begin = std::chrono::steady_clock::now();
  float mean_gpu = reduce_mean_gpu(vec, vec.size());
  end = std::chrono::steady_clock::now();
  timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("GPU took {} s!\n", timediff / 1000.0);

  fmt::print("mean_cpu: {:.2f} | mean_gpu: {:.2f} \n", mean_cpu, mean_gpu);
//  for (int i=0; i<n; i++)
//    fmt::print("idx: {:3d} | {:5.2f}\n", i, vec[i]);
  return 0;
}