# Monte Carlo simulations of long-term stock market evolution

## Overview

The main goal of this project is to learn some things :).

The other practical goal is to estimate the likelihood distribution of future development of a stock portfolio. This
could be used to decide how much to invest when saving for retirement for example.

Python is used to get the S&P500 data, C++ code runs the MonteCarlo simulations in one thread and visualizes with
DearImGUI in another. For each simulation, we update the portfoli value per month, using monthly returns that are
sampled either from:

1. Gaussian with mean/std
2. historical data (S&P500)

## Quickstart

C++ side: simulations + visualization with [DearImGui](https://github.com/ocornut/imgui)

- requires g++-10 and CMake 3.20!
    - `sudo pip install cmake --upgrade`
    - `sudo apt install -y g++-11 && update-alternatives --config g++`

```
sudo apt install libfmt-dev libglfw3-dev
mkdir build && cd build
cmake .. -GNinja && ninja && cd ..

# get historical stock data with Python Yahoo-finances
conda create --name stock_market_monte_carlo python=3.9
conda activate stock_market_monte_carl
pip install -r python/requirements.txt
python python/get_data.py get_data_SP500

# now run the simulations and visualize
./build/example_gui_simulated
```

### Using C++ only for simulation and write to disk, Python for visualization

- simulation assuming stock market is Gaussian: `example_simulated`
- simulation based on historical data: `example_historical`

```
rm -rf ./output/
./build/example_<simulated,historical>
python python/plot_returns.py plot_many_returns --dir=output
```

### Offline Visualization with Python

1. run C++ program that writes simulation results to CSV files: `build/example_<simulated,historical>`
2. plot with python & matplotlib
    1. plot single simulation: `python python/plot_returns.py plot_returns --csv_file=output/gaussian_00001.csv`
    2. plot many simulations: `python python/plot_returns.py plot_many_returns --dir=output/`
    3. 
## Implementation Checklist

- [x] gaussian monthly returns, calculate fund value after N periods
- [x] monte carlo simulation of (1)
- [x] use historical return data
- [x] visualization of fund value and average returns
- [x] GUI with [imgui](https://github.com/ocornut/imgui) to see progress while simulating,
  see [implot](https://github.com/epezent/implot) and also [mahi-gui](https://github.com/mahilab/mahi-gui)
    - [x] separate threads for simulation and visualization
    - [x] slider for final amount & probability of reaching that
- [x] c++ executable with arguments to avoid need to recompile
- [ ] withdrawal strategies
    - [ ] taking out fixed amount every period
    - [ ] taking out some percentage every period
    - [ ] taking out varying percentage every period
- [x] Monte Carlo simulation with openMP
- [ ] Monte Carlo simulation on GPU


