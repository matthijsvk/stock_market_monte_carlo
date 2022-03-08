###############################################################################
# Samples
###############################################################################

add_executable(example_simulated monte_carlo_simulated.cpp)
target_link_libraries(example_simulated PUBLIC stock_market_monte_carlo)

add_executable(example_historical monte_carlo_historical.cpp)
target_link_libraries(example_historical PUBLIC stock_market_monte_carlo)

add_executable(example_gui gui.cpp)
target_link_libraries(example_gui PRIVATE
        stock_market_monte_carlo
        imgui
        implot
        )

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC "${implot_SOURCE_DIR}")