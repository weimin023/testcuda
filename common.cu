#include <cuda_fp16.h>
#include <random>

inline cudaError_t CHECK_CUDA(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void fill_random_float_values(float* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int8_values(int8_t* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_int_distribution<int8_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int16_values(int16_t* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_int_distribution<int16_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int32_values(int32_t* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_int_distribution<int32_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_half_values(__half* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(0.0f, 127.0f);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = __float2half(uniform_dist(e));
    }
}

void fill_random_double_values(double* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_real_distribution<double> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = uniform_dist(e);
    }
}