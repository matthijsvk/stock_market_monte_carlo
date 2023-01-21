#include <fmt/core.h>

#include <atomic>
#include <locale>
#include "stock_market_monte_carlo/simulations.h"

void update_mean_std(float &mean, float &std, std::vector<float> &means, std::vector<float> &variances) {
  // population mean: just mean of means
  double sum = std::accumulate(means.begin(), means.end(), 0.0);
  mean = float(sum) / means.size();

  // population variance: https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
  // -> average the variances; then you can take square root to get the average standard deviation
//  double sum_var = std::accumulate(variances.begin(), variances.end(), 0.0);
//  double mean_var = sum_var / variances.size();
//  std = float(std::sqrt(mean_var));

  sum = 0;
  for (int i = 0; i < variances.size(); i++) {
//    float tmp = v[i] - mean);
    sum += variances[i];
  }
  printf("Sum var: %f\n", sum);
  float var = sum / variances.size();
  std = std::sqrt(var);
}

double normalCDF(double value)
{
   return 0.5 * erfc(-value * M_SQRT1_2);
}

static double cumulative_normal_standard(double d)
{
  const double       A1 = 0.31938153;
  const double       A2 = -0.356563782;
  const double       A3 = 1.781477937;
  const double       A4 = -1.821255978;
  const double       A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double
      K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(- 0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
    cnd = 1.0 - cnd;

  return cnd;
}

float cumulative_normal(float x, float m, float s){
  // simply move and scale so we can use standard cumulative normal distribution
  return normalCDF((x-m)/s); //cumulative_normal_standard((x-m)/s);
}

float normal(float x, float m, float s) {
  static const float inv_sqrt_2pi = 0.3989422804014327;
  float a = (x - m) / s;
  return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

long update_count_below_min(float &min_final_amount, float mean, float std, long n_simulations) {
  // central limit theorem: totals are normally distributed (for large N)
  // -> estimate count from mean/var
  float prob = cumulative_normal(min_final_amount, mean, std);
  return n_simulations * prob;
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
        "usage: benchmark_mc_gpu_reduceBlock <n_gpus> <n_months> <n_simulations>, eg "
        "benchmark_mc_gpu_reduceBlock 1 360 100000");
    exit(0);
  }

  float initial_capital = 1000;
  std::vector<float> historical_returns = read_historical_returns("data/SP500_monthly_returns.csv");
  std::vector<float> means, variances;
  std::atomic<long> n_simulations = 0;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  mc_simulations_gpu_reduceBlock(
      n_simulations, max_n_simulations, n_periods, initial_capital, historical_returns, means, variances, n_gpus);
  fmt::print("n_simulations: {:d}\n", n_simulations);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);

  // get and print statistics
  float mean, std;
  update_mean_std(mean, std, means, variances);
  fmt::print("mean: {:.2f} | std: {:.2f} \n", mean, std);

  // TODO the distribution is NOT a symmetrical gaussian, so normalCDF to estimate counts from mean/std isn't correct!
  // it may be an upper bound on the number of failures
  long count = update_count_below_min(initial_capital, mean, std, max_n_simulations);
  fmt::print("count_below {:.1f}: {:L} ({:4f}%) \n", initial_capital, count, 100*float(count)/max_n_simulations);
  fmt::print("prob below min: {:.3f}% \n", 100 * cumulative_normal(initial_capital, mean, std));

  return 0;
}