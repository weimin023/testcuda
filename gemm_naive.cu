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
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M && col < N) {
        float tmp = 0;
        for (int k=0;k<K;++k) {
            tmp += dA[row * K + k] * dB[k * N + col];
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

template<int TILE_SIZE> __global__ void gemm_reg(const float *dA, const float *dB, float *dC, int M, int K, int N) {

    int c = threadIdx.x;
    int r = threadIdx.y;
    
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float SA[TILE_SIZE][TILE_SIZE];
    __shared__ float SB[TILE_SIZE][TILE_SIZE];

    float reg_tile = 0;
    for (int t = 0; t < K; t += TILE_SIZE) {

        if (row < M && (t + c) < K) {
            SA[r][c] = dA[row * K + (t + c)];
        } else {
            SA[r][c] = 0;
        }

        if ((t + r) < K && col < N) {
            SB[r][c] = dB[(t + r) * N + col];
        } else {
            SB[r][c] = 0;
        }
        __syncthreads();

        // accumulate sum
        // global idx = i * N + j;
        for (int k = 0; k < TILE_SIZE; ++k) {
            reg_tile += SA[r][k] * SB[k][c];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        dC[row * N + col] = reg_tile;
    }
    
}

template<int TILE_SIZE>
__global__ void gemm_shared_transposed(float *dA, float *dB, float *dC, int M, int K, int N) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Starting indices for this block
    int row = TILE_SIZE * bx + tx;
    int col = TILE_SIZE * by + ty;
    
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile - directly in row-major order
        if (row < M && (t * TILE_SIZE + ty) < K) {
            As[tx][ty] = dA[row * K + t * TILE_SIZE + ty];
        } else {
            As[tx][ty] = 0.0f;
        }
        
        // Load B tile - with transposition
        if ((t * TILE_SIZE + tx) < K && col < N) {
            // Original B is in row-major: B[k][n]
            // Load it transposed into shared memory
            Bs[ty][tx] = dB[(t * TILE_SIZE + tx) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute on the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Now both matrices are accessed in a coalesced manner
            sum += As[tx][k] * Bs[ty][k];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        dC[row * N + col] = sum;
    }
}

template<int TILE_SIZE>
__global__ void gemm_three(const float *A, const float *B, const float *C, float *D, int M, int N, int K, int L) {
    // M*K, K*N, N*L
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int r = threadIdx.y;
    int c = threadIdx.x;
    
    __shared__ float SA[TILE_SIZE][TILE_SIZE];
    __shared__ float SB[TILE_SIZE][TILE_SIZE];
    __shared__ float SC[TILE_SIZE][TILE_SIZE];
    __shared__ float AB[TILE_SIZE][TILE_SIZE];

    float reg = 0;
    for (int n = 0; n < N; n += TILE_SIZE) {
        AB[r][c] = 0;

        for (int k = 0; k < K; k += TILE_SIZE) {
            if (row < M && (k + c) < K) {
                SA[r][c] = A[row * K + (k + c)];
            } else {
                SA[r][c] = 0;
            }
    
            if ((k + r) < K && (n + c) < N) {
                SB[r][c] = B[(k + r) * N + (n + c)];
            } else {
                SB[r][c] = 0.0f;
            }
            __syncthreads();
    
            for (int i = 0; i < TILE_SIZE; ++i) {
                AB[r][c] += SA[r][i] * SB[i][c];
            }
            __syncthreads();
        }

        if ((n + r) < N && col < L) {
            SC[r][c] = C[(n + r) * L + col];
        } else {
            SC[r][c] = 0;
        }
        __syncthreads();

        // 计算(A*B)*C的部分结果
        for (int i = 0; i < TILE_SIZE; ++i) {
            if ((n + i) < N) {
                reg += AB[r][i] * SC[i][c];
            }
        }
        __syncthreads();
    }

    if (row < M && col < L) {
        D[row * L + col] = reg;
    }
    
}

int main() {
    
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int l = 1024;

    int trials = 100;

    int matrixSize = m * n;

    thrust::host_vector<float> h_A(matrixSize);
    thrust::host_vector<float> h_B(matrixSize);
    thrust::host_vector<float> h_C(matrixSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 10);

    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
        h_C[i] = dis(gen);
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;
    thrust::device_vector<float> d_D1(matrixSize, 0.0);

    dim3 threadNum(8, 8);
    dim3 blockNum((m + threadNum.x - 1)/threadNum.x, (n + threadNum.y - 1)/threadNum.y);

    float ker_time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<float> d_tmp(matrixSize, 0.0);
    for (int i=0;i<trials;++i) {
        //gemm_naive<<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, k, n);
        //gemm_shared<32><<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, k, n);
        //gemm_shared_transposed<32><<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, k, n);
        gemm_reg<8><<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_tmp.data().get(), m, k, n);
        gemm_reg<8><<<blockNum, threadNum>>>(d_tmp.data().get(), d_C.data().get(), d_D1.data().get(), m, k, n);
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

    //---------------CPU

    std::vector<float> tmp_cpu_v(m*n);
    std::vector<float> cpu_v(m*n);
    double st, ela;
    st = get_walltime();
    matrixSerial(h_A.data(), h_B.data(), tmp_cpu_v.data(), m, k, n);
    matrixSerial(tmp_cpu_v.data(), h_C.data(), cpu_v.data(), m, k, n);
    ela = get_walltime() - st;
    printf("CPU time:%.2f second\n", ela);
    //---------------

    thrust::device_vector<float> d_D2(matrixSize, 0.0);
    gemm_three<8><<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), d_D2.data().get(), m, n, k, l);

    thrust::host_vector<float> h_D1 = d_D1;
    thrust::host_vector<float> h_D2 = d_D2;
    compare(h_D2.data(), h_D1.data(), m, n);
    
    return 0;
}