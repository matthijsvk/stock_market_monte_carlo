#!/bin/bash
sudo /usr/local/cuda/bin/nv-nsight-cu-cli -f --target-processes all --set full  --call-stack --nvtx -o `pwd`/profiling/ncu_bs256 "build/benchmark_mc_gpu" 1000 10000000
