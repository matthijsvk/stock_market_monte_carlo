#include "helpers.h"

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include "fmt/core.h"

float update_fund(float fund_value, float period_return);
void __many_updates(float *returns, float *totals, long n_periods);
std::vector<float> many_updates(float fund_value, std::vector<float> &returns, long n_updates);

// simulated with Gaussian monthly returns
std::vector<float> sample_returns_gaussian(long n, float return_mean, float return_std);
void one_simulation_gaussian(std::string output_fname,
                             float initial_capital,
                             long n_periods,
                             float return_mean,
                             float return_std);
void monte_carlo_gaussian(long n, float initial_capital, long n_periods, float return_mean, float return_std);

// using historical data
std::vector<float> read_historical_returns(std::string csv_fpath);
std::vector<float> sample_returns_historical(long n, std::vector<float> &historical_returns);
void one_simulation_historical(std::string output_fname,
                               float initial_capital,
                               long n_periods,
                               std::vector<float> &historical_returns);
void monte_carlo_historical(long n, float initial_capital, long n_periods, std::string csv_fpath);

void mc_simulations(std::atomic<long> &n_simulations,
                    long max_n_simulations,
                    long n_periods,
                    float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<std::vector<float>> &mc_data,
                    std::vector<float> &final_values);

// GPU
void __many_updates_gpu(float *returns, float *totals, long n);
std::vector<float> many_updates_gpu(float fund_value, std::vector<float> &returns, long n_updates);

void _mc_simulations_gpu(float *historical_returns,
                         long n_historical_returns,
                         float *totals,
                         long max_n_simulations,
                         long n_periods);

void mc_simulations_gpu(std::atomic<long> &n_simulations,
                        long max_n_simulations,
                        long n_periods,
                        float initial_capital,
                        std::vector<float> &historical_returns,
                        std::vector<float> &final_values);
