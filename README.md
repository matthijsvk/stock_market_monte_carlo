# Monte Carlo simulations of taking money out of a fund for some time, considering varying market conditions

## Quickstart

C++ side: simulations
`mkdir build && cd build && cmake .. && make && cd ..`
`./build/example_simulated`

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


## Markets

- monthly returns
    1. Gaussian with mean/std
    2. Sample from historical data
        1. python python/get_data.py get_data_SP500

## Implementation

1. guassian monthly returns, calculate fund value after N periods
2. monte carlo simulation of (1)
3. use historical return data
4. c++ executable with arguments
5. withdrawal strategies
    1. taking out fixed amount every period
    2. taking out some percentage every period
    3. taking out varying percentage every period
6. Monte Carlo simulation on GPU

## Visualization

1. write to CSV file, then plot with python & matplotlib
    1. plot single simulation: `python python/plot_returns.py plot_returns --csv_file=output/gaussian_00001.csv`
    2. plot many simulations: `python python/plot_returns.py plot_many_returns --dir=output/`
2. write GUI using https://github.com/ocornut/imgui to display from C++
3. dynamically update the GUI while simulations are running

## Installation

```
sudo apt install libfmt-dev

```