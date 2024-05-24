cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cudadev LANGUAGES C CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA 12.3 REQUIRED)
find_package(GTest REQUIRED)

# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --extended-lambda -Xcudafe --diag_suppress=997")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utest
)

set(SOURCES
    unittest.cu
)

add_executable(unittest ${SOURCES})
target_link_libraries(unittest ${GTEST_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(unittest)


