#include <fmt/core.h>
#include <atomic>
#include "stock_market_monte_carlo/simulations.h"

int main(int argc, char *argv[]) {
  fmt::print("argc: {}\n", argc);
  unsigned long max_n_simulations;
  unsigned int n_periods;
  if (argc == 3) {
    char *end;
    n_periods = long(std::strtol(argv[1], &end, 10));
    max_n_simulations = long(std::strtol(argv[2], &end, 10));
    fmt::print("n_periods: {} | max_n_simulations: {}\n",
               n_periods,
               max_n_simulations);
  } else {
    fmt::print(
        "usage: visualize_returns <n_months> <n_simulations>, eg "
        "visualize_returns 360 100000");
    exit(0);
  }

  float initial_capital = 1000;
  std::vector<float> historical_returns =
      read_historical_returns("data/SP500_monthly_returns.csv");
  std::vector<float> final_values(max_n_simulations, initial_capital);
  std::atomic<long> n_simulations = 0;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  mc_simulations(n_simulations,
                 max_n_simulations,
                 n_periods,
                 initial_capital,
                 historical_returns,
                 final_values);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  fmt::print(
      "All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);

  return 0;
}