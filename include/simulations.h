#include <helpers.h>

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <fmt/core.h>

float update_fund(float fund_value, float period_return);
std::vector<float> many_updates(float fund_value, std::vector<float> &returns);

// simulated with Gaussian monthly returns
std::vector<float> sample_returns_gaussian(int n, float return_mean, float return_std);
void one_simulation_gaussian(const std::string output_fname,
                             float initial_capital,
                             int n_periods,
                             float return_mean,
                             float return_std);
void monte_carlo_gaussian(int n, float initial_capital, int n_periods, float return_mean, float return_std);

// using historical data
std::vector<float> read_historical_returns(std::string csv_fpath);
std::vector<float> sample_returns_historical(int n, std::vector<float> &historical_returns);
void one_simulation_historical(const std::string output_fname,
                               float initial_capital,
                               int n_periods,
                               std::vector<float> &historical_returns);
void monte_carlo_historical(int n, float initial_capital, int n_periods, const std::string csv_fpath);

