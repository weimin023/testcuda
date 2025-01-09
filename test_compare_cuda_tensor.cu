#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

void fill_random_double_values(double* arr, size_t n, std::default_random_engine& e)
{
    std::uniform_real_distribution<double> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i) {
        arr[i] = uniform_dist(e);
    }
}

// Naive GEMM
template<typename T>
__global__ void __launch_bounds__(1024) gemm_naive(const T *__restrict__ dA, const T *__restrict__ dB, T *__restrict__ dC, int M, int K, int N)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    T tmp = 0;

    if (row < M && col < N)
    {
        for (int s = 0; s < K; s++)
        {
            tmp += dA[row * K + s] * dB[s * N + col];
        }
        dC[row * N + col] = tmp;
    }
}

template<typename T>
__global__ void __launch_bounds__(1024) gemm_CUDA(T *__restrict__ c, const T *__restrict__ a, const T *__restrict__ b, int M, int N, int K) {
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int TILE_SIZE = 16;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = bx * TILE_SIZE + tx;
    const int row = by * TILE_SIZE + ty;

    __shared__ T SA[TILE_SIZE][TILE_SIZE];
    __shared__ T SB[TILE_SIZE][TILE_SIZE];

    T sum = 0;
    for (int k = 0; k < (K + TILE_SIZE - 1)/TILE_SIZE; ++k) {
        if (row < M && k * TILE_SIZE + tx < K) {
            SA[ty][tx] = a[row * K + k * TILE_SIZE + tx];
        } else {
            SA[ty][tx] = 0;
        }

        if (col < N && k * TILE_SIZE + ty < K) {
            SB[ty][tx] = b[col + (k * TILE_SIZE + ty) * N];
        } else {
            SB[ty][tx] = 0;
        }

        __syncthreads();

        for (int n_k = 0; n_k < TILE_SIZE; ++n_k) {
            sum += SA[ty][n_k] * SB[n_k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
    

}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;
const int warpSize = 32;
__global__ void row_wmma_ker(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int lda = K; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = N;
    int ldc = N;

    int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpY = (blockIdx.y * blockDim.y + threadIdx.y);
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int aRow = warpX * WMMA_M; //(aRow, aCol),x方向数多少个WMMA_M
    int bCol = warpY * WMMA_N;
    for (int i = 0; i < K; i += WMMA_K)
    {
        int aCol = i;
        int bRow = i;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // 读取A,B矩阵里面子矩阵的元素
            wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);
            // 子矩阵做乘法
            wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        }
    }
    int cRow = warpX * WMMA_M;
    int cCol = warpY * WMMA_N;
    if (cRow < M && cCol < N)
    {
        // Store the output
        wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}

int main() {

    int M = 2048;
    int K = 512;
    int N = 1024;

    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;

    std::default_random_engine random_engine(0);

    thrust::host_vector<float> h_a_vec(M*K);
    thrust::host_vector<float> h_b_vec(K*N);

    fill_random_float_values(h_a_vec.data(), h_a_vec.size(), random_engine);
    fill_random_float_values(h_b_vec.data(), h_b_vec.size(), random_engine);
    
    thrust::device_vector<float> d_a_vec = h_a_vec;
    thrust::device_vector<float> d_b_vec = h_b_vec;
    thrust::device_vector<float> d_c_vec(M*N);

    // CPU
    /*std::vector<int16_t> h_c_cpu(M*N, 0);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int r=0 ; r<M ; ++r) {
        for (int c=0; c<N ; ++c) {
            for (int k=0; k<K; ++k) {
                h_c_cpu[r * N + c] += h_a_vec[r * K + k] * h_b_vec[k * N + c];
            }
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto t_cpu = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();*/

    dim3 threadNum(16, 16);
    dim3 blockNum((M + threadNum.x - 1)/threadNum.x, (N + threadNum.y - 1)/threadNum.y);

    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    const int numIterations = 1;
    float naive_totalTime = 0.0f;

    // 1. CUDA NAIVE
    for (int i = 0; i < numIterations; ++i) {
        cudaEventRecord(cuda_start, 0);

        gemm_naive<float><<<blockNum, threadNum>>>(d_a_vec.data().get(), d_b_vec.data().get(), d_c_vec.data().get(), M, K, N);

        cudaEventRecord(cuda_end, 0);
        cudaEventSynchronize(cuda_end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, cuda_start, cuda_end);

        naive_totalTime += ms;
    }

    thrust::host_vector<float> h_naive_c_vec = d_c_vec;

    // 2. CUDA
    float v2_totalTime = 0.0f;
    for (int i = 0; i < numIterations; ++i) {
        cudaEventRecord(cuda_start, 0);

        gemm_CUDA<float><<<blockNum, threadNum>>>(d_c_vec.data().get(), d_a_vec.data().get(), d_b_vec.data().get(), M, N, K);

        cudaEventRecord(cuda_end, 0);
        cudaEventSynchronize(cuda_end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, cuda_start, cuda_end);

        v2_totalTime += ms;
    }
    
    thrust::host_vector<float> h_c_vec = d_c_vec;

    // 3. 

    int BLOCK_DIM_x = 32; // 必须取32，否则结果报错
    int BLOCK_DIM_y = 4;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim((M + (WMMA_M * BLOCK_DIM_x / warpSize - 1)) / (WMMA_M * BLOCK_DIM_x / warpSize), (N + WMMA_N * BLOCK_DIM_y - 1) / (WMMA_N * BLOCK_DIM_y), 1);

    float v3_totalTime = 0.0f;
    for (int i = 0; i < numIterations; ++i) {
        cudaEventRecord(cuda_start, 0);

        row_wmma_ker<<<grid_dim, block_dim>>>(d_a_vec.data().get(), d_b_vec.data().get(), d_c_vec.data().get(), M, K, N);

        cudaEventRecord(cuda_end, 0);
        cudaEventSynchronize(cuda_end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, cuda_start, cuda_end);

        v3_totalTime += ms;
    }
    
    thrust::host_vector<float> h_c_vec_v3 = d_c_vec;

    // compare
    bool flg = 0;
    for (int r=0 ; r<M ; ++r) {
        for (int c=0; c<N ; ++c) {
            if (h_naive_c_vec[r * N + c] != h_c_vec[r * N + c]) {
                printf("Failed.\n");
                flg = 1;
            }
            if (flg) break;
        }
        if (flg) break;
    }

    if (!flg) {
        std::cout << "NAIVE execution time: " << naive_totalTime/numIterations << " ms" << std::endl;
        std::cout << "V2 execution time: " << v2_totalTime/numIterations << " ms" << std::endl;
        std::cout << "V3 execution time: " << v3_totalTime/numIterations << " ms" << std::endl;
    }

    return 0;
}