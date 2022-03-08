#include <helpers.h>

#include <chrono>
#include <random>
#include <vector>

#include <fmt/core.h>
#include "csv.h"

float update_fund(float fund_value, float period_return) {
  return fund_value * (float(100.0) + period_return) / 100;
}

std::vector<float> many_updates(float fund_value, std::vector<float> &returns) {
  std::vector<float> fund_values = {fund_value};

  for (auto this_return: returns) {
    float new_fund_value = update_fund(fund_values.back(), this_return);
    fund_values.push_back(new_fund_value);
  }
  return fund_values;
}

std::vector<float> sample_returns_gaussian(int n, float return_mean, float return_std) {
  /* Create gaussian random engine with the help of seed */
  unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
  std::default_random_engine e(seed);
  std::normal_distribution<float> distN(return_mean, return_std);

  std::vector<float> returns;
  returns.reserve(n);
  for (auto period = 0; period < n; period++) {
    returns.push_back(distN(e));
  }
  return returns;
}

void one_simulation_gaussian(const std::string output_fname,
                             float initial_capital,
                             int n_periods,
                             float return_mean,
                             float return_std) {
  std::vector<float> returns = sample_returns_gaussian(n_periods, return_mean, return_std);
  std::vector<float> values = many_updates(initial_capital, returns);
  write_data_file(output_fname, returns, values);
}

void monte_carlo_gaussian(int n, float initial_capital, int n_periods, float return_mean, float return_std) {
  // TODO: write all to 1 data structure, then to 1 csv file so save on I/O for Monte Carlo?
  for (int i = 0; i < n; i++) {
    std::string output_fname = fmt::format("gaussian_{:05d}.csv", i);
    one_simulation_gaussian(output_fname, initial_capital, n_periods, return_mean, return_std);
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

std::vector<float> sample_returns_historical(int n, std::vector<float> &historical_returns) {
  std::vector<float> returns;
  returns.reserve(n);

  std::sample(historical_returns.begin(),
              historical_returns.end(),
              std::back_inserter(returns),
              n,
              std::mt19937{std::random_device{}()});

  return returns;
}

void one_simulation_historical(const std::string output_fname,
                               float initial_capital,
                               int n_periods,
                               std::vector<float> &historical_returns) {
  std::vector<float> returns = sample_returns_historical(n_periods, historical_returns);
  std::vector<float> values = many_updates(initial_capital, returns);
  write_data_file(output_fname, returns, values);
}

void monte_carlo_historical(int n, float initial_capital, int n_periods, const std::string csv_fpath) {
  // TODO: write all to 1 data structure, then to 1 csv file so save on I/O for Monte Carlo?
  std::vector<float> historical_returns = read_historical_returns(csv_fpath);

  for (int i = 0; i < n; i++) {
    std::string output_fname = fmt::format("historical_{:05d}.csv", i);
    one_simulation_historical(output_fname, initial_capital, n_periods, historical_returns);
  }
}