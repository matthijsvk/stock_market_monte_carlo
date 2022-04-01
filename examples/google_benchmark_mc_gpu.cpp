#include <benchmark/benchmark.h>
#include <fmt/core.h>

#include <cstdlib>
#include <thread>
#include <vector>

#include "stock_market_monte_carlo/simulations.h"

static void BM_MCGPU(benchmark::State& state) {
  // Perform setup here
  unsigned long max_n_simulations = 100000000; //state.range(0);
  unsigned int n_periods = 1000;  // state.range(1);
  fmt::print("n_sim: {:d} | n_periods: {:d}\n", max_n_simulations, n_periods);

  float initial_capital = 1000;
  std::vector<float> historical_returns =
      read_historical_returns("data/SP500_monthly_returns.csv");
  std::vector<float> final_values(max_n_simulations, initial_capital);
  std::atomic<unsigned long> n_simulations = 0;

  for (auto _ : state) {
    // This code gets timed
    mc_simulations_gpu(n_simulations,
                       max_n_simulations,
                       n_periods,
                       initial_capital,
                       historical_returns,
                       final_values);
  }
}

// BENCHMARK(BM_MCGPU)->Ranges({{2 << 20, 2 << 27}, {1000}})->Complexity();
//BENCHMARK(BM_MCGPU)->RangeMultiplier(10)->Range(1e6, 1e8);  //->Complexity();
BENCHMARK(BM_MCGPU);
// BENCHMARK(BM_MCGPU);
//  Run the benchmark
BENCHMARK_MAIN();
