#include "stock_market_monte_carlo/simulations.h"

#include <fmt/core.h>

#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "csv.h"
#include "stock_market_monte_carlo/helpers.h"

float update_fund(float fund_value, float period_return) {
  return fund_value * (float(100.0) + period_return) / 100;
}

void __many_updates(float *returns, float *totals, unsigned int n_periods) {
  for (unsigned int i = 0; i < n_periods; i++) {
    totals[i + 1] = update_fund(totals[i], returns[i]);
  }
}

std::vector<float> many_updates(float fund_value,
                                std::vector<float> &returns,
                                unsigned int n_periods) {
  // initialize arrays
  float totals[n_periods + 1];
  totals[0] = fund_value;

  float *returns_arr = &returns[0];

  // do the computations
  __many_updates(returns_arr, totals, n_periods);

  // convert to vector b/c it's expected TODO this is a copy?
  std::vector<float> v(totals, totals + n_periods + 1);
  return v;
}

std::vector<float> sample_returns_gaussian(unsigned int n,
                                           float return_mean,
                                           float return_std) {
  /* Create gaussian random engine with the help of seed */
  unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
  std::default_random_engine e(seed);
  std::normal_distribution<float> distN(return_mean, return_std);

  std::vector<float> returns;
  returns.reserve(n);
  for (auto period = 0; period < n; period++) {
    returns[period] = distN(e);  //.push_back(distN(e));
  }
  return returns;
}

void one_simulation_gaussian(const std::string output_fname,
                             float initial_capital,
                             unsigned int n_periods,
                             float return_mean,
                             float return_std) {
  std::vector<float> returns =
      sample_returns_gaussian(n_periods, return_mean, return_std);
  std::vector<float> values = many_updates(initial_capital, returns, n_periods);
  write_data_file(output_fname, returns, values);
}

void monte_carlo_gaussian(unsigned long n,
                          float initial_capital,
                          unsigned int n_periods,
                          float return_mean,
                          float return_std) {
  // TODO: write all to 1 data structure, then to 1 csv file so save on I/O for
  // Monte Carlo?
  for (unsigned long i = 0; i < n; i++) {
    std::string output_fname = fmt::format("gaussian_{:05d}.csv", i);
    one_simulation_gaussian(
        output_fname, initial_capital, n_periods, return_mean, return_std);
  }
}

//// Historical data
std::vector<float> read_historical_returns(std::string csv_fpath) {
  io::CSVReader<1> in(csv_fpath);
  in.read_header(io::ignore_extra_column, "returns");
  float this_return;

  std::vector<float> historical_returns;
  while (in.read_row(this_return)) {
    historical_returns.push_back(this_return);
  }
  return historical_returns;
}

std::vector<float> sample_returns_historical(
    unsigned int n, std::vector<float> &historical_returns) {
  std::vector<float> returns;
  returns.reserve(n);

  std::random_device rd;   // only used once to initialise (seed) engine
  std::mt19937 rng(rd());  // random-number engine: Mersenne-Twister

  // alternatively, roll our own sampler
  std::uniform_int_distribution<int> uni(
      0, historical_returns.size() - 1);  // guaranteed unbiased
  for (unsigned int i = 0; i < n; i++) {
    unsigned int random_idx = uni(rng);
    returns.push_back(historical_returns.at(random_idx));
  }

  return returns;
}

void one_simulation_historical(const std::string output_fname,
                               float initial_capital,
                               unsigned int n_periods,
                               std::vector<float> &historical_returns) {
  std::vector<float> returns =
      sample_returns_historical(n_periods, historical_returns);
  std::vector<float> values = many_updates(initial_capital, returns, n_periods);
  write_data_file(output_fname, returns, values);
}

void monte_carlo_historical(unsigned long n,
                            float initial_capital,
                            unsigned int n_periods,
                            const std::string csv_fpath) {
  // TODO: write all to 1 data structure, then to 1 csv file so save on I/O for
  // Monte Carlo?
  std::vector<float> historical_returns = read_historical_returns(csv_fpath);

  for (unsigned int i = 0; i < n; i++) {
    std::string output_fname = fmt::format("historical_{:05d}.csv", i);
    one_simulation_historical(
        output_fname, initial_capital, n_periods, historical_returns);
  }
}

// TODO Can't overload mc_cimulations for some reason??
void mc_simulations_keepdata(std::atomic<unsigned long> &n_simulations,
                             const unsigned long max_n_simulations,
                             const unsigned int n_periods,
                             const float initial_capital,
                             std::vector<float> &historical_returns,
                             std::vector<std::vector<float>> &mc_data,
                             std::vector<float> &final_values) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  unsigned long block_size = 1000;
  unsigned long n_blocks = std::ceil(max_n_simulations / float(block_size));

  // leave 1 core for visualization
  const int num_cpu_cores =
      std::max(1, int(std::thread::hardware_concurrency() - 1));
  fmt::print("number of cpu cores: {}\n", num_cpu_cores);

#pragma omp parallel for schedule(dynamic)                              \
    num_threads(num_cpu_cores) default(none) shared(n_blocks,           \
                                                    block_size,         \
                                                    n_simulations,      \
                                                    max_n_simulations,  \
                                                    n_periods,          \
                                                    initial_capital,    \
                                                    historical_returns, \
                                                    mc_data,            \
                                                    final_values)
  for (unsigned long n_sims_blocks = 0; n_sims_blocks < n_blocks;
       n_sims_blocks++) {
    // last block may not contain full 'block_size' elements!
    unsigned long block_id = n_sims_blocks * block_size;
    unsigned int this_block_size =
        std::min(block_size, max_n_simulations - block_id);
    //        fmt::print("Block id {}, block_id: {}, this_block_size:
    //        {}\n", n_sims_blocks, block_id, this_block_size);

    for (long n_sims = 0; n_sims < this_block_size; n_sims++) {
      unsigned long id = block_id + n_sims;

      // do calculations
      std::vector<float> returns =
          sample_returns_historical(n_periods, historical_returns);
      std::vector<float> values =
          many_updates(initial_capital, returns, n_periods);
      // assert (returns.size() == values.size());

      final_values[id] = values.back();
      mc_data[id] = values;

      //            fmt::print("sim id: {:d} \n", id);
    }
    n_simulations += this_block_size;
    fmt::print(
        "{:d}/{:d} simulations done\n", n_simulations, max_n_simulations);
  }
  assert(n_simulations = max_n_simulations);  // must be true here

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  fmt::print(
      "All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);
}

void mc_simulations(std::atomic<unsigned long> &n_simulations,
                    const unsigned long max_n_simulations,
                    const unsigned int n_periods,
                    const float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<float> &final_values) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  const unsigned long block_size = 1000;
  const unsigned long n_blocks =
      std::ceil(max_n_simulations / float(block_size));

  // leave 1 core for visualization
  const unsigned int num_cpu_cores =
      std::max(1, int(std::thread::hardware_concurrency() - 1));
  fmt::print("number of cpu cores: {}\n", num_cpu_cores);

#pragma omp parallel for schedule(dynamic)                              \
    num_threads(num_cpu_cores) default(none) shared(n_blocks,           \
                                                    block_size,         \
                                                    n_simulations,      \
                                                    max_n_simulations,  \
                                                    n_periods,          \
                                                    initial_capital,    \
                                                    historical_returns, \
                                                    final_values)
  for (unsigned long n_sims_blocks = 0; n_sims_blocks < n_blocks;
       n_sims_blocks++) {
    // last block may not contain full 'block_size' elements!
    unsigned long block_id = n_sims_blocks * block_size;
    unsigned int this_block_size =
        std::min(block_size, max_n_simulations - block_id);
    //        fmt::print("Block id {}, block_id: {}, this_block_size:
    //        {}\n", n_sims_blocks, block_id, this_block_size);

    for (unsigned long n_sims = 0; n_sims < this_block_size; n_sims++) {
      unsigned long id = block_id + n_sims;
      float total = initial_capital;

      // set up random generator // todo 1 per block??
      std::random_device rd;   // only used once to initialise (seed) engine
      std::mt19937 rng(rd());  // random-number engine: Mersenne-Twister
      std::uniform_int_distribution<int> uni(0, historical_returns.size() - 1);

      for (unsigned int i = 0; i < n_periods; i++) {
        total = update_fund(total, historical_returns[uni(rng)]);
      }
      final_values[id] = total;
    }
    n_simulations += this_block_size;
    if (n_sims_blocks % (n_blocks / 10) == 0)
      fmt::print("{:d}/{:d} simulations done\n", n_simulations, max_n_simulations);
  }
  assert(n_simulations = max_n_simulations);  // must be true here

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  fmt::print(
      "All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);
}

