#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>

// Naive GEMM

__global__ void NaiveGEMM(int *in_A, int *in_B, int *out_C, int M, int N, int K) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= M || col >= N) return;

    int s = 0.0;
    for (int k=0;k<K;++k) {
        s += in_A[row * K + k] * in_B[k * N + col];
    }
    out_C[row * N + col] = s;

}

int main() {
    
    int m = 8;
    int n = 8;
    int k = 8;

    int matrixSize = m * n;

    thrust::host_vector<int> h_A(matrixSize);
    thrust::host_vector<int> h_B(matrixSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 10);

    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    thrust::device_vector<int> d_A = h_A;
    thrust::device_vector<int> d_B = h_B;
    thrust::device_vector<int> d_C(matrixSize, 0.0);

    dim3 threadNum(16, 16);
    dim3 blockNum((m + threadNum.x - 1)/threadNum.x, (n + threadNum.y - 1)/threadNum.y);
    NaiveGEMM<<<blockNum, threadNum>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, 8);

    thrust::host_vector<int> h_C = d_C;

    std::cout << "A_in: " << std::endl;
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            int idx = i * n + j;
            std::cout << h_A[idx] << ", ";
            if ((idx+1)%m == 0) std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "B_in: " << std::endl;
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            int idx = i * n + j;
            std::cout << h_B[idx] << ", ";
            if ((idx+1)%m == 0) std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "C_out: " << std::endl;
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            int idx = i * n + j;
            std::cout << h_C[idx] << ", ";
            if ((idx+1)%m == 0) std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::vector<int> cpu_c(m*n, 0);
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            int idx = i * n + j;
            for (int c=0;c<k;++c) {
                cpu_c[idx] += h_A[i * k + c] * h_B[j + c * n];
            }
        }
    }

    bool flg = 1;
    for (int i=0;i<m*n;++i) {
        if (cpu_c[i] != h_C[i]) {
            std::cout << "CPU and GPU results NOT match!" << std::endl;
            flg = 0;
            break;
        }
    }
    if (flg) std::cout << "SUCCESS: CPU and GPU results match!" << std::endl;
    
    return 0;
}