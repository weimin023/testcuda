cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cudadev LANGUAGES C CXX CUDA)

enable_testing()
set(CMAKE_CUDA_ARCHITECTURES 86;89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#-----------------------------------------------#
#                     ARMPL                     #
#-----------------------------------------------#
# Specify the ARMPL installation path
set(ARMPL_DIR "/opt/arm/armpl_24.10_gcc")

# Include directories
include_directories(${ARMPL_DIR}/include)

# Link directories
link_directories(${ARMPL_DIR}/lib)


find_package(OpenMP REQUIRED)
#-----------------------------------------------#
#                     CUDA                      #
#-----------------------------------------------#
find_package(CUDA 12.3 REQUIRED)

# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --extended-lambda -Xcudafe --diag_suppress=997")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utest
)
#-----------------------------------------------#
#                     GTest                     #
#-----------------------------------------------#
# Find and install Google Test if not found
find_package(GTest)

if (NOT GTest_FOUND)
    message(STATUS "Google Test not found. Installing...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.11.0.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Include GTest directories
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

#-----------------------------------------------#
#                     FFTW                      #
#-----------------------------------------------#
#find_library(FFTW_LIBRARIES NAMES fftw3 PATHS /usr/lib/x86_64-linux-gnu)
#
#if(NOT FFTW_LIBRARIES)
#    message(FATAL_ERROR "FFTW not found")
#endif()

#-----------------------------------------------#
#                     openCV                    #
#-----------------------------------------------#
#find_package(OpenCV REQUIRED)

#message(STATUS "OpenCV library status:" )
#message(STATUS "    config: ${OpenCV_DIR}" )
#message(STATUS "    version: ${OpenCV_VERSION}" )
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )
#include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA compile options
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")

set(SOURCES
    unittest.cu
    ./utest/misc.cu
    ./utest/evd.cu
)

add_executable(unittest ${SOURCES})
target_link_libraries(unittest gtest gtest_main ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES} cublas cusolver)

add_executable(gemm_naive gemm_naive.cu)
target_link_libraries(gemm_naive ${CUDA_LIBRARIES})

add_executable(test_graph test_graph.cu ./utest/evd.cu)
target_link_libraries(test_graph ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES} cublas cusolver)

add_executable(test_graph2 test_graph2.cu)
target_link_libraries(test_graph ${CUDA_LIBRARIES})

add_executable(test_graph3 test_graph3.cu)
target_link_libraries(test_graph3 ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES} cublas cusolver)

#add_executable(test_armpl test_armpl.cu)
#target_link_libraries(test_armpl ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES} cublas cusolver armpl_lp64_mp pthread m)

add_executable(test_cuMul test_cuMul.cu)
target_link_libraries(test_cuMul ${CUDA_LIBRARIES})

add_executable(test_conflict test_conflict.cu)
target_link_libraries(test_conflict ${CUDA_LIBRARIES})

add_executable(test_topk test_topk.cu)
target_link_libraries(test_topk ${CUDA_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(unittest)
