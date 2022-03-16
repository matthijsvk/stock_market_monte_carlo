#include <iostream>
#include "stock_market_monte_carlo/simulations.h"

int main() {
  ///////////////////////////////////////////
  // Fund and Market configuration
  ///////////////////////////////////////////
  float initial_capital = 1000;

  float monthly_return_mean = 6.0 / 12;
  float monthly_return_std = 10.0 / 12;  // 68% is within 1 std from mean, 95% within 2 std, 99.7% within 3 std

  int n_years = 30;
  int n_months = 12 * n_years;

  // Monte Carlo
  int n_simulations = 1000;
  ///////////////////////////////////////////

  // many simulations
  monte_carlo_gaussian(n_simulations, initial_capital, n_months, monthly_return_mean, monthly_return_std);

  /* single simulation
  one_simulation_gaussian("output1.csv",
                          initial_capital,
                          n_months,
                          monthly_return_mean,
                          monthly_return_std);
  */

  // what goes on under the hood
  /*
  std::vector<float> returns = get_returns_gaussian(n_months,
                                                    monthly_return_mean,
                                                    monthly_return_std);
  fmt::print("Monthly Returns: \n");
  print_vector(returns);

  std::vector<float> values = many_updates(initial_capital, returns);
  fmt::print("Monthly Evolution of Fund Value: \n");
  print_vector(values);

  write_data_file("output2.csv", returns, values);
  */

  std::cout << "Done!" << std::endl;

  return 0;
}

