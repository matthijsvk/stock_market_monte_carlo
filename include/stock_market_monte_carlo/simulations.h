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
                                long n_updates);

// simulated with Gaussian monthly returns
std::vector<float> sample_returns_gaussian(unsigned int n,
                                           float return_mean,
                                           float return_std);
void one_simulation_gaussian(std::string output_fname,
                             float initial_capital,
                             unsigned int n_periods,
                             float return_mean,
                             float return_std);
void monte_carlo_gaussian(long n,
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
void monte_carlo_historical(long n,
                            float initial_capital,
                            unsigned int n_periods,
                            std::string csv_fpath);


//////////////////////////////////////////////////////
//////////////////////// CPU /////////////////////////
//////////////////////////////////////////////////////

// stores only final values
void mc_simulations(std::atomic<long> &n_simulations,
                    long max_n_simulations,
                    unsigned int n_periods,
                    float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<float> &final_values);
                    
// keeps everything stored in mc_data
void mc_simulations_keepdata(std::atomic<long> &n_simulations,
                    long max_n_simulations,
                    unsigned int n_periods,
                    float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<std::vector<float>> &mc_data,
                    std::vector<float> &final_values);

//////////////////////////////////////////////////////
//////////////////////// GPU /////////////////////////
//////////////////////////////////////////////////////

float reduce_mean_gpu(std::vector<float> &vec, long n);

void mc_simulations_gpu(std::atomic<long> &n_simulations,
                        long max_n_simulations,
                        int n_periods,
                        float initial_capital,
                        std::vector<float> &returns,
                        std::vector<float> &totals,
                        int n_gpus);

void mc_simulations_gpu_reduceBlock(std::atomic<long> &n_simulations,
                        long max_n_simulations,
                        int n_periods,
                        float initial_capital,
                        std::vector<float> &returns,
                        std::vector<float> &means,
                        std::vector<float> &variances,
                        int n_gpus);

