cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cudadev LANGUAGES C CXX CUDA)

enable_testing()
set(CMAKE_CUDA_ARCHITECTURES 86;89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE Release)

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
set(CMAKE_CUDA_FLAGS_DEBUG "-std=c++17 --extended-lambda -Xcudafe --diag_suppress=997 -g -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -std=c++17 --extended-lambda -Xcudafe --diag_suppress=997 -DNDEBUG -lineinfo")
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
#                   CUTLASS                     #
#-----------------------------------------------#
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

#-----------------------------------------------#
#                     openCV                    #
#-----------------------------------------------#
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:" )
message(STATUS "    config: ${OpenCV_DIR}" )
message(STATUS "    version: ${OpenCV_VERSION}" )
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )
include_directories(${OpenCV_INCLUDE_DIRS})

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

add_executable(test_maxFilter test_maxFilter.cu)
target_link_libraries(test_maxFilter ${CUDA_LIBRARIES})

add_executable(test_sort test_sort.cu)
target_link_libraries(test_sort ${CUDA_LIBRARIES})

add_executable(test_tensorCore_leimao test_tensorCore_leimao.cu)
target_link_libraries(test_tensorCore_leimao ${CUDA_LIBRARIES})

add_executable(test_compare_cuda_tensor test_compare_cuda_tensor.cu)
target_link_libraries(test_compare_cuda_tensor ${CUDA_LIBRARIES})

add_executable(test_cpu_gemm test_cpu_gemm.cc)
target_compile_options(test_cpu_gemm PRIVATE -march=native -O3 -ffast-math)
target_link_libraries(test_cpu_gemm PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_simpleTensor test_simpleTensor.cu)
target_link_libraries(test_simpleTensor ${CUDA_LIBRARIES})

add_executable(test_transpose test_transpose.cu)
target_link_libraries(test_transpose ${CUDA_LIBRARIES})

add_executable(test_ldmatrix test_ldmatrix.cu)
target_link_libraries(test_ldmatrix ${CUDA_LIBRARIES})

add_executable(test_ccl1 test_ccl1.cu)
target_link_libraries(test_ccl1 ${CUDA_LIBRARIES})

#add_executable(test_ccl2 test_ccl2.cu)
#target_link_libraries(test_ccl2 ${CUDA_LIBRARIES})

add_executable(test_ballot test_ballot.cu)
target_link_libraries(test_ballot ${CUDA_LIBRARIES})

add_executable(test_atomic test_atomic.cu)
target_link_libraries(test_atomic ${CUDA_LIBRARIES})

add_executable(test_float4 test_float4.cu)
target_link_libraries(test_float4 ${CUDA_LIBRARIES})

add_executable(test_histogram test_histogram.cu)
target_link_libraries(test_histogram ${CUDA_LIBRARIES})

add_executable(opt_label_propagation opt_label_propagation.cu)
target_link_libraries(opt_label_propagation ${CUDA_LIBRARIES})

add_executable(test_clz test_clz.cu)
target_link_libraries(test_clz ${CUDA_LIBRARIES})

add_executable(test_warp_sfl test_warp_sfl.cu)
target_link_libraries(test_warp_sfl ${CUDA_LIBRARIES})

add_executable(test_ccl_opencv test_ccl_opencv.cpp)
target_link_libraries(test_ccl_opencv ${OpenCV_LIBS})

add_executable(test_ccl_opencv2 test_ccl_opencv2.cu)
target_link_libraries(test_ccl_opencv2 ${OpenCV_LIBS} ${CUDA_LIBRARIES})

add_executable(test_cutlass test_cutlass.cu)
target_include_directories(test_cutlass PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_compile_options(test_cutlass PRIVATE --expt-relaxed-constexpr)

include(GoogleTest)
gtest_discover_tests(unittest)
