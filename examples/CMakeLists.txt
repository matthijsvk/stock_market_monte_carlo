###############################################################################
# Samples
###############################################################################

add_executable(example_simulated monte_carlo_simulated.cpp)
target_link_libraries(example_simulated PRIVATE stock_market_monte_carlo)

add_executable(example_historical monte_carlo_historical.cpp)
target_link_libraries(example_historical PRIVATE stock_market_monte_carlo)

# With ImGUI
add_executable(example_gui gui.cpp)
target_link_libraries(example_gui PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(example_gui PRIVATE "${implot_SOURCE_DIR}")

add_executable(example_gui_simulated gui_simulated.cpp)
target_link_libraries(example_gui_simulated PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(example_gui_simulated PRIVATE "${implot_SOURCE_DIR}")

add_executable(example_gui_simulated_v2 gui_simulated_v2.cpp)
target_link_libraries(example_gui_simulated_v2 PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(example_gui_simulated_v2 PRIVATE "${implot_SOURCE_DIR}")

# with GPU
add_executable(example_gpu example_gpu.cpp)
target_link_libraries(example_gpu PRIVATE stock_market_monte_carlo)

add_executable(example_gui_simulated_gpu gui_simulated_gpu.cpp)
target_link_libraries(example_gui_simulated_gpu PRIVATE stock_market_monte_carlo imgui implot)
target_include_directories(example_gui_simulated_gpu PRIVATE "${implot_SOURCE_DIR}")