#include <fmt/core.h>

#include <atomic>
#include <locale>
#include "stock_market_monte_carlo/simulations.h"

void update_mean_std(float &mean, float &std, std::vector<float> &v, long n_el) {
//  double sum = std::accumulate(v.begin(), v.begin() + n_el, 0.0);
//  mean = sum / n_el;
//
//  double sqsum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
//  std = std::sqrt(sqsum / v.size() - mean * mean);

  double sum = 0;
  for (int i = 0; i < v.size(); i++) {
    sum += v[i];
  }
  mean = sum / v.size();

  sum = 0;
  for (int i = 0; i < v.size(); i++) {
    float tmp = (v[i] - mean);
    sum += tmp * tmp;
  }
  float var = sum / v.size();

  std = std::sqrt(var);
}

long update_count_below_min(float &min_final_amount, const std::vector<float> &final_values, long n_simulations) {
  long count_below_min = 0;
  for (long i = 0; i < n_simulations; i++) {
    float val = final_values[i];
    if (val == -1) {
      // TODO this should never happen, but it does with openMP....
      fmt::print("final value[{}] == -1!\n", i);
    }
    if (val < min_final_amount) count_below_min++;
  }
  return count_below_min;
}


int main(int argc, char *argv[]) {
  std::locale::global(std::locale("en_US.UTF-8"));
  fmt::print("argc: {}\n", argc);
  long max_n_simulations;
  int n_periods, n_gpus;
  if (argc == 4) {
    char *end;
    n_gpus = atoi(argv[1]);
    n_periods = atoi(argv[2]);
    max_n_simulations = long(std::strtol(argv[3], &end, 10));
    fmt::print("n_periods: {} | max_n_simulations: {}\n", n_periods, max_n_simulations);
  } else {
    fmt::print(
        "usage: example_gui_simulated <n_gpus> <n_months> <n_simulations>, eg "
        "example_gui_simulated 1 360 100000");
    exit(0);
  }

  float initial_capital = 1000;
  std::vector<float> historical_returns = read_historical_returns("data/SP500_monthly_returns.csv");
  std::vector<float> final_values;//(max_n_simulations, initial_capital);
  std::atomic<long> n_simulations = 0;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  mc_simulations_gpu(n_simulations, max_n_simulations, n_periods, initial_capital, historical_returns, final_values, n_gpus);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);

  // get and print statistics
  float mean, std;
  update_mean_std(mean, std, final_values, final_values.size());
  fmt::print("mean: {:.2f} | std: {:.2f} \n", mean, std);

  long count = update_count_below_min(initial_capital, final_values, max_n_simulations);
  fmt::print("count_below_min: {:L} ({:4f}%)",count, 100*float(count)/max_n_simulations);

  return 0;
}