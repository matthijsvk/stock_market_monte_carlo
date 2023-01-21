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

- requires g++-10 and CMake 3.19!
    - `sudo pip install cmake --upgrade`
    - `sudo apt install -y g++-11 && update-alternatives --config g++`

```bash
sudo apt install libfmt-dev libglfw3-dev
mkdir build && cd build
cmake .. -GNinja && ninja && cd ..

# get historical stock data with Python Yahoo-finances
conda create --name smmc -y python=3.10
conda activate smmc
pip install -r python/requirements.txt
python python/get_data.py get_data_SP500

# now run the simulations and visualize
# following runs for 30 years (360 months), 1000000 simulations
./build/visualize_returns_cpu 360 1000000

# for lots of simulations (>10 million), 'v2' stores only the final result, not all intermediate values.
# this speeds things up and mostly saves lots of memory
./build/visualize_returns_cpu_v2 360 10000000
```

If you have an NVIDIA GPU with CUDA, there's also a GPU version which can do hundreds of millions of simulations in a few seconds. 

```bash
build/visualize_returns_gpu 1 360 1000000000
```

## Using C++ for simulation and write to disk, Python for visualization of return plots

1. run C++ program that writes simulation results to CSV files:
   - simulation assuming stock market is Gaussian: `build/monte_carlo_simulated`
   - simulation based on historical data: `build/monte_carlo_historical`
2. plot with python & matplotlib
    1. plot single simulation: `python python/plot_returns.py plot_returns --csv_file=outputs/gaussian_00001.csv`
    2. plot many simulations: `python python/plot_returns.py plot_many_returns --max_n=100 --dir=outputs/`


## Benchmarks of different implementations

CPU: Ryzen 7 5800X 8-Core, 3.6GHz
GPU: NVIDIA RTX 3070

-  360 periods, 100.000.000 simulations

|       Program       | Time (s) | Memory (GB) |
|:-------------------:|:--------:|:-----------:|
| CPU v1, single core |  479.52  |     ~28     | store all points
|   CPU v1, openMP    |  85.26   |     ~28     | use all CPU cores
|   CPU v2, openMP    |  41.81   |    ~0.5     | only keep final value
|         GPU         |   1.76   |    ~0.5     | 
|   GPU, optimized    |   0.26   |    ~0.5     | 
|   GPU, reduceBlock  |   0.13   |             | compute means on GPU to further reduce data transfer


// TODO: un-optimized GPU variants? -> see executables.... what do the versions mean??

```
build/benchmark_mc_cpu 360 100000000
build/benchmark_mc_cpu_v2 360 100000000
build/benchmark_mc_gpu 1 360 100000000
build/benchmark_mc_gpu_reduceBlock 1 360 100000000
```

You can also use [Google Benchmark](https://github.com/google/benchmark), which allows easy comparison between versions.
Edit benchmark_mc_gpu_google.cpp with the function/arguments, recompile, and then:
```
build/benchmark_mc_gpu_google --benchmark_out=benchmark_output/bench.json --benchmark_repetitions=10
cat profiling/bench.json
```
Compare 2 run:
```
python benchmark/tools/compare.py benchmarks benchmark_output/baseline.json benchmark_output/contender.json
```

## GPU Development

#### Profiling
see `profiling/howto.sh`
```
# with GPU 2
CUDA_VISIBLE_DEVICES=1 sudo -E /usr/local/cuda/bin/nv-nsight-cu-cli -f --devices 0 --target-processes all --set full  --call-stack --nvtx -o `pwd`/profiling/ncu_1k_1M_bs256_v3 "build/benchmark_mc_gpu_bs256_v3" 1000 5000000
```

## Debugging
checking races in CUDA kernel:
`compute-sanitizer --tool racecheck build/benchmark_mc_gpu_reduceBlock 1 360 1000000`


## Implementation Checklist

- [x] gaussian monthly returns, calculate fund value after N periods
- [x] monte carlo simulation of (1)
- [x] use historical return data
- [x] visualization of fund value and average returns
- [x] GUI with [imgui](https://github.com/ocornut/imgui) to see progress while simulating,
  see [implot](https://github.com/epezent/implot)
    - [x] separate threads for simulation and visualization
    - [x] slider for final amount & probability of reaching that
- [x] c++ executable with arguments to avoid need to recompile
- [x] Monte Carlo simulation with openMP
- [x] Monte Carlo simulation on GPU
  - [x] optimize kernel w/ Nsight Compute (fancy local PRNG, shmem, etc)
  - [x] summarize per-block to reduce CPU-GPU transfers
  - [x] multi-GPU
  - [x] multi-GPU (async overlapping data transfer)
- [ ] fix visualize_returns_distribution_gpu_reduceBlock: histogram/statistics aren't correct.
- [ ] withdrawal strategies
    - [ ] taking out fixed amount every period
    - [ ] taking out some percentage every period
    - [ ] taking out varying percentage every period





