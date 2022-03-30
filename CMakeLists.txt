cmake_minimum_required(VERSION 3.19)
#see https://cliutils.gitlab.io/modern-cmake/chapters/basics/example.html
# and https://gitlab.com/CLIUtils/modern-cmake/tree/master/examples/extended-project

SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

PROJECT(
        stock_market_monte_carlo
        VERSION 0.1
        DESCRIPTION "Trying out things"
        LANGUAGES CXX CUDA
)

option(BUILD_EXAMPLES "Build ${CMAKE_PROJECT_NAME} example applications?" ON)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)


###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No release type specified. Setting to 'Release'.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

################################################################################
## CUDA compiler setup
################################################################################

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED OFF)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
endif ()
## find max architecture for available GPU
## adapted from https://stackoverflow.com/a/69353718
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

## Remove unsupported architectures
list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "86+PTX")

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

message(STATUS "Targeting GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
get_directory_property(TCNN_HAS_PARENT PARENT_DIRECTORY)
if (TCNN_HAS_PARENT)
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
endif ()

#list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
#list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

## for Profiling
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")

###############################################################################
# other libraries
###############################################################################

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 8.1.1)
FetchContent_Declare(csv GIT_REPOSITORY https://github.com/ben-strasser/fast-cpp-csv-parser.git GIT_TAG master)

# Externally provided libraries
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG release-1.11.0)
FetchContent_Declare(googlebenchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG master)

FetchContent_MakeAvailable(
        fmt
        csv
        googletest
        googlebenchmark)

#set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

###############################################################################
# other libraries: GUI stuff: https://alandefreitas.github.io/moderncpp/guis/imgui/
###############################################################################

find_package(OpenGL)
find_package(X11 REQUIRED)

FetchContent_Declare(glad GIT_REPOSITORY https://github.com/Dav1dde/glad.git GIT_TAG v0.1.36)
FetchContent_MakeAvailable(glad)

if (NOT TARGET glfw)
    # Fetch GLFW
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.3.6)
    FetchContent_MakeAvailable(glfw)
    target_compile_options(glfw PRIVATE -Wno-pedantic -Wno-missing-field-initializers -Wno-unused-parameter)
endif ()
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

# Fetch ImGUI
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.86)
FetchContent_GetProperties(imgui)
if (NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
    set(IMGUI_INCLUDE_DIR ${imgui_SOURCE_DIR}/)
    file(GLOB IMGUI_SOURCES ${imgui_SOURCE_DIR}/*.cpp)
    file(GLOB IMGUI_HEADERS ${imgui_SOURCE_DIR}/*.h)
    add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_SOURCES} ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp src/simulations.cu include/stock_market_monte_carlo/gpu.h)
    add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

    # Link
    target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLAD_INCLUDE_DIR})
    target_link_libraries(imgui ${OPENGL_LIBRARIES} glfw glad ${CMAKE_DL_LIBS})
endif ()

# ImPlot
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG v0.13)
FetchContent_GetProperties(implot)
if (NOT implot_POPULATED)
    FetchContent_Populate(implot)
    file(GLOB implot_SOURCES ${implot_SOURCE_DIR}/*.cpp)
    file(GLOB implot_HEADERS ${implot_SOURCE_DIR}/*.h)
    add_library(implot ${implot_HEADERS} ${implot_SOURCES})
    target_link_libraries(implot PUBLIC imgui)
    target_compile_definitions(implot PUBLIC IMPLOT_BACKEND_ENABLE_OPENGL3 IMGUI_IMPL_OPENGL_LOADER_GLAD)
    target_compile_options(implot PRIVATE -Wall -Wextra -pedantic -Werror -mavx2 -O2)
endif ()




###############################################################################
# Include files
###############################################################################
#find_library(
#        #        CUDA_CUBLASLT_LIBRARY cublasLt
#        ${CUDA_HOME}/lib64
#        ${CUDA_HOME}/lib
#)
#message(STATUS "CUDA_HOME: $ENV{CUDA_HOME}")


###############################################################################
# tiny-cuda-nn library, samples, and benchmarks
###############################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

add_subdirectory("src")

if (BUILD_EXAMPLES)
    add_subdirectory("examples")
endif ()
