#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include "gemm_comm.cu"

// Naive GEMM
__global__ void gemm_naive(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    float tmp = 0;
    if (row < M && col < N)
    {
        for (int s = 0; s < K; s++)
        {
            tmp += dA[row * K + s] * dB[s * N + col];
        }
        dC[row * N + col] = tmp;
    }
}

// shared mem with tile
template<int TILE_SIZE> __global__ void gemm_shared(float *dA, float *dB, float *dC, int M, int K, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int width = (K + TILE_SIZE - 1) / TILE_SIZE;

    __shared__ float SA[TILE_SIZE][TILE_SIZE];
    __shared__ float SB[TILE_SIZE][TILE_SIZE];
    float reg_tmp = 0;

    for (int w=0;w<width;++w) {
        if (row < M && (w * TILE_SIZE + threadIdx.y) < K) {
            SA[threadIdx.x][threadIdx.y] = dA[row * K + w * TILE_SIZE + threadIdx.y];
        } else {
            SA[threadIdx.x][threadIdx.y] = 0;
        }
        if (col < N && (w * TILE_SIZE + threadIdx.x) < K) {
            SB[threadIdx.x][threadIdx.y] = dB[(w * TILE_SIZE + threadIdx.x)*N + col];
        } else {
            SB[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();

        for (int s=0;s<TILE_SIZE;++s) {
            reg_tmp += SA[threadIdx.x][s] * SB[s][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        dC[row * N + col] = reg_tmp;
    }
    
}

int main() {
    
    int m = 1024;
    int n = 1024;
    int k = 1024;

    int trials = 1;

    int matrixSize = m * n;

    thrust::host_vector<float> h_A(matrixSize);
    thrust::host_vector<float> h_B(matrixSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 10);

    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(matrixSize, 0.0);

    dim3 threadNum(32, 32);
    dim3 blockNum((m + threadNum.x - 1)/threadNum.x, (n + threadNum.y - 1)/threadNum.y);

    float ker_time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i=0;i<trials;++i) {
        gemm_naive<<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, k, n);
        gemm_shared<32><<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, k, n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (trials * 1000.), ker_time / trials);
    printf("grid dim: %d, %d, %d\n", blockNum.x, blockNum.y, blockNum.z);
    printf("block dim: %d, %d, %d\n", threadNum.x, threadNum.y, threadNum.z);

    thrust::host_vector<float> h_C = d_C;

    std::vector<float> cpu_v(m*n);
    double st, ela;
    st = get_walltime();
    matrixSerial(h_A.data(), h_B.data(), cpu_v.data(), m, k, n);
    ela = get_walltime() - st;
    printf("CPU time:%.2f second\n", ela);

    compare(h_C.data(), cpu_v.data(), m, n);
    
    return 0;
}