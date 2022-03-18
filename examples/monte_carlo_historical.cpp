#include <iostream>

#include "stock_market_monte_carlo/simulations.h"

int main() {
  ///////////////////////////////////////////
  // Fund and Market configuration
  ///////////////////////////////////////////
  float initial_capital = 1000;

  int n_years = 30;
  int n_months = 12 * n_years;

  // Monte Carlo
  int n_simulations = 1000;
  ///////////////////////////////////////////

  //  // using historical monthly return data
  //  std::string output_fname = fmt::format("historical_{:05d}.csv", 1);
  //  std::vector<float> historical_returns =
  //  read_historical_returns("data/SP500_monthly_returns.csv");
  //  one_simulation_historical(output_fname, initial_capital, n_months,
  //  historical_returns);

  monte_carlo_historical(n_simulations,
                         initial_capital,
                         n_months,
                         "data/SP500_monthly_returns.csv");

  std::cout << "Done!" << std::endl;

  return 0;
}
