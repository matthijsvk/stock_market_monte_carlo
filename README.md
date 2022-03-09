# Monte Carlo simulations of long-term stock market evolution

## Quickstart

C++ side: simulations
- requires g++ >= 9.1 (for `std::filesystem` support)
```
sudo apt install libfmt-dev libglfw3-dev
mkdir build && cd build
cmake .. -GNinja && ninja && cd ..
./build/example_simulated
./build/example_gui_simulated
```

Python side: visualization

```
conda create --name stock_market_monte_carlo python=3.9
conda activate stock_market_monte_carl
pip install -r python/requirements.txt
python python/plot_returns.py plot_many_returns --dir=output
```

Using historical data

```
rm -rf ./output
python python/get_data.py get_data_SP500
./build/example_historical
python python/plot_returns.py plot_many_returns --dir=output
```

## Overview

The goal of this project is to estimate the likelihood distribution of future development of a stock portfolio.
This could be used to decide how much to invest when saving for retirement.

Perform Monte Carlo simulations where we sample monthly returns either from:
    1. Gaussian with mean/std
    2. historical data (S&P500)

C++ code runs the MC simulations and writes each run to a CSV file.
Python is used to get the S&P500 data, and for visualization.

Future goals: 
    - GUI to visualize from C++ while simulations are running
    - run simulations on GPU
    - add withdrawal plans (verify '4% rule')

### Implementation Steps

- [x] gaussian monthly returns, calculate fund value after N periods
- [x] monte carlo simulation of (1)
- [x] use historical return data
- [x] visualization of fund value and average returns
- [ ] c++ executable with arguments to avoid need to recompile
- [x] GUI with [imgui](https://github.com/ocornut/imgui) to see progress while simulating, see [implot](https://github.com/epezent/implot) and also [mahi-gui](https://github.com/mahilab/mahi-gui)
  - [ ] fancy graphs with shaded Q1-Q3
  - [ ] separate threads for simulation and visualization
  - [ ] slider for final amount & probability of reaching that
- [ ] likelihood of going broke + reaching some target amount
- [ ] withdrawal strategies
  - [ ] taking out fixed amount every period
  - [ ] taking out some percentage every period
  - [ ] taking out varying percentage every period
- [ ] Monte Carlo simulation on GPU

## Visualization

1. write to CSV file, then plot with python & matplotlib
    1. plot single simulation: `python python/plot_returns.py plot_returns --csv_file=output/gaussian_00001.csv`
    2. plot many simulations: `python python/plot_returns.py plot_many_returns --dir=output/`
    3. single CSV for all simulations (faster) ?
