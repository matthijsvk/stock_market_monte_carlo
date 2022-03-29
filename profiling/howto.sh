#!/bin/bash
sudo /usr/local/cuda/bin/nv-nsight-cu-cli -f --target-processes all --set full  --call-stack --nvtx -o /home/matthijs/Code/finances/stock_market_monte_carlo/profiling/ncu_bs256 "build/example_gui_simulated_gpu" 1000 10000000
