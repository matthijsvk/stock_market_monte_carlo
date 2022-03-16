###############################################################################
# Source files
###############################################################################

set(SOURCES
        simulations.cpp
        helpers.cpp
        gpu.cpp
        gpu.cu
        simulations.cu
        )

###############################################################################
# Linker / library
###############################################################################

add_library(${CMAKE_PROJECT_NAME} STATIC ${SOURCES})

##### Others
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ../include)

# link with header-only libraries # TODO is this the right way???
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC "${csv_SOURCE_DIR}")

# link built libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
        fmt::fmt-header-only
        Threads::Threads
        )

### CUDA
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} cuda cublas)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)


## TODO is this useful at all? From https://developer.nvidia.com/blog/building-cuda-applications-cmake/
target_compile_features(${CMAKE_PROJECT_NAME} PUBLIC cxx_std_20)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# IDEs should put the headers in a nice place
source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})
