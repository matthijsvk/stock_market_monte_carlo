#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "fmt/core.h"
#include "helpers.h"

float update_fund(float fund_value, float period_return);
void __many_updates(float *returns, float *totals, unsigned int n_periods);
std::vector<float> many_updates(float fund_value,
                                std::vector<float> &returns,
                                unsigned long n_updates);

// simulated with Gaussian monthly returns
std::vector<float> sample_returns_gaussian(unsigned int n,
                                           float return_mean,
                                           float return_std);
void one_simulation_gaussian(std::string output_fname,
                             float initial_capital,
                             unsigned int n_periods,
                             float return_mean,
                             float return_std);
void monte_carlo_gaussian(unsigned long n,
                          float initial_capital,
                          unsigned int n_periods,
                          float return_mean,
                          float return_std);

// using historical data
std::vector<float> read_historical_returns(std::string csv_fpath);
std::vector<float> sample_returns_historical(
    unsigned int n, std::vector<float> &historical_returns);
void one_simulation_historical(std::string output_fname,
                               float initial_capital,
                               unsigned int n_periods,
                               std::vector<float> &historical_returns);
void monte_carlo_historical(unsigned long n,
                            float initial_capital,
                            unsigned int n_periods,
                            std::string csv_fpath);

// stores only final values
void mc_simulations(std::atomic<unsigned long> &n_simulations,
                    unsigned long max_n_simulations,
                    unsigned int n_periods,
                    float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<float> &final_values);
// keeps everything stored in mc_data
void mc_simulations_keepdata(std::atomic<unsigned long> &n_simulations,
                    unsigned long max_n_simulations,
                    unsigned int n_periods,
                    float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<std::vector<float>> &mc_data,
                    std::vector<float> &final_values);

// GPU
void _mc_simulations_gpu(float *historical_returns,
                         unsigned int n_historical_returns,
                         float *totals,
                         unsigned long max_n_simulations,
                         unsigned int n_periods);

void mc_simulations_gpu(std::atomic<unsigned long> &n_simulations,
                        unsigned long max_n_simulations,
                        unsigned int n_periods,
                        float initial_capital,
                        std::vector<float> &historical_returns,
                        std::vector<float> &final_values,
                        int n_gpus);
void testRNG(int n);
