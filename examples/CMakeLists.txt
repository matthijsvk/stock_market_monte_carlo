###############################################################################
# Samples
###############################################################################

## 1. Utilities
add_executable(visualize_rng visualize_rng.cpp)
target_link_libraries(visualize_rng PRIVATE imgui implot fmt::fmt-header-only)
target_include_directories(visualize_rng PRIVATE "${implot_SOURCE_DIR}")

add_executable(visualize_dist visualize_distribution.cpp)
target_link_libraries(visualize_dist PRIVATE imgui implot fmt::fmt-header-only)
target_include_directories(visualize_dist PRIVATE "${implot_SOURCE_DIR}")


## 2. get returns from historical or simulated data
add_executable(monte_carlo_simulated monte_carlo_simulated.cpp)
target_link_libraries(monte_carlo_simulated PRIVATE stock_market_monte_carlo)

add_executable(monte_carlo_historical monte_carlo_historical.cpp)
target_link_libraries(monte_carlo_historical PRIVATE stock_market_monte_carlo)


## 3. visualize return graphs with ImGUI (with CPU)
add_executable(example_imgui example_imgui.cpp)
target_link_libraries(example_imgui PRIVATE imgui implot)
target_include_directories(example_imgui PRIVATE "${implot_SOURCE_DIR}")

add_executable(visualize_returns_cpu visualize_returns_cpu.cpp)
target_link_libraries(visualize_returns_cpu PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(visualize_returns_cpu PRIVATE "${implot_SOURCE_DIR}")

add_executable(visualize_returns_cpu_v2 visualize_returns_cpu_v2.cpp)
target_link_libraries(visualize_returns_cpu_v2 PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(visualize_returns_cpu_v2 PRIVATE "${implot_SOURCE_DIR}")
#
##add_executable(benchmark_mc_cpu benchmark_mc_cpu.cpp)
##target_link_libraries(benchmark_mc_cpu PRIVATE stock_market_monte_carlo benchmark::benchmark)

############################ GPU ##################################
## 1. GPU examples  

add_executable(example_gpu example_gpu.cpp)
target_link_libraries(example_gpu PRIVATE stock_market_monte_carlo)

add_executable(reduce_mean benchmark_reduce_mean.cpp)
target_link_libraries(reduce_mean PRIVATE stock_market_monte_carlo)


### 2. Distribution of returns, simulation with GPU

# transfer all data between CPU and GPU
add_executable(visualize_returns_gpu visualize_returns_gpu.cpp)
target_link_libraries(visualize_returns_gpu PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(visualize_returns_gpu PRIVATE "${implot_SOURCE_DIR}")

# more efficient implementation for mean return by doing reduction on GPU (less CPU-GPU traffic), but loses distribution information...
add_executable(visualize_returns_distribution_gpu_reduceBlock visualize_returns_distribution_gpu_reduceBlock.cpp)
target_link_libraries(visualize_returns_distribution_gpu_reduceBlock PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(visualize_returns_distribution_gpu_reduceBlock PRIVATE "${implot_SOURCE_DIR}")

## 3. Benchmarks

add_executable(benchmark_mc_gpu_google benchmark_mc_gpu_google.cpp)
target_link_libraries(benchmark_mc_gpu_google PRIVATE stock_market_monte_carlo benchmark::benchmark)


add_executable(benchmark_mc_cpu benchmark_mc_cpu.cpp)
target_link_libraries(benchmark_mc_cpu PRIVATE stock_market_monte_carlo benchmark::benchmark)

add_executable(benchmark_mc_cpu_v2 benchmark_mc_cpu_v2.cpp)
target_link_libraries(benchmark_mc_cpu_v2 PRIVATE stock_market_monte_carlo benchmark::benchmark)

add_executable(benchmark_mc_gpu benchmark_mc_gpu.cpp)
target_link_libraries(benchmark_mc_gpu PRIVATE stock_market_monte_carlo benchmark::benchmark)

add_executable(benchmark_mc_gpu_reduceBlock benchmark_mc_gpu_reduceBlock.cpp)
target_link_libraries(benchmark_mc_gpu_reduceBlock PRIVATE stock_market_monte_carlo benchmark::benchmark)






