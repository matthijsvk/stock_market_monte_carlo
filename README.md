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
./build/example_gui_simulated 360 1000000

# for lots of simulations (>10 million), 'v2' stores only the final result, not all intermediate values.
# this speeds things up and mostly saves lots of memory
./build/example_gui_simulated_v2 360 10000000
```

If you have an NVIDIA GPU with CUDA, there's also a GPU version which can do hundreds of millions of simulations in a few seconds. 

```
build/example_gui_simulated_gpu 360 1000000000
```

### Benchmarks: 360 periods, 100 000 000 simulations.

CPU: Ryzen 7 5800X 8-Core, 3.6GHz
GPU: NVIDIA RTX 3070

|       Program       | Time (s) | Memory (GB) |
|:-------------------:|:--------:|:-----------:|
| CPU v1, single core |  479.52  |     ~28     |
|   CPU v1, openMP    |  85.26   |     ~28     |
|   CPU v2, openMP    |  41.81   |    ~0.5     |
|         GPU         |   1.76   |    ~0.5     |
|   GPU, optimized    |   0.26   |    ~0.5     |

You can use [Google Benchmark](https://github.com/google/benchmark) with the 'benchmark_mc_gpu' example program:
```
build/benchmark_mc_gpu --benchmark_out=profiling/bench.json --benchmark_repetitions=10
cat profiling/bench.json
```
If you have 2 runs stored to `bench1.json` and `bench2.json`, compare them with:
```
python benchmark/tools/compare.py benchmarks bench1.json bench2.json
```

### Profiling
see `profiling/howto.sh`
```
# with GPU 2
CUDA_VISIBLE_DEVICES=1 sudo -E /usr/local/cuda/bin/nv-nsight-cu-cli -f --devices 0 --target-processes all --set full  --call-stack --nvtx -o `pwd`/profiling/ncu_1k_1M_bs256_v3 "build/benchmark_mc_gpu_bs256_v3" 1000 5000000
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
  see [implot](https://github.com/epezent/implot)
    - [x] separate threads for simulation and visualization
    - [x] slider for final amount & probability of reaching that
- [x] c++ executable with arguments to avoid need to recompile
- [x] Monte Carlo simulation with openMP
- [x] Monte Carlo simulation on GPU
  - [x] optimize kernel w/ Nsight Compute (fancy local PRNG, shmem, etc)
  - [x] summarize per-block to reduce CPU-GPU transfers
  - [ ] multi-GPU
- [ ] withdrawal strategies
    - [ ] taking out fixed amount every period
    - [ ] taking out some percentage every period
    - [ ] taking out varying percentage every period


### debugging
checking races in CUDA kernel:
`compute-sanitizer --tool racecheck build/benchmark_mc_gpu_reduceBlock 1 360 1000000`


