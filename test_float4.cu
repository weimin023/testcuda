#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define N 1024
#define THREADS_PER_BLOCK 256

#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])

__global__ void sigmoid(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void sigmoid_float4(float* x, float* y, int n) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < n) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = 1.0f / (1.0f + expf(-tmp_x.x));
        tmp_y.y = 1.0f / (1.0f + expf(-tmp_x.y));
        tmp_y.z = 1.0f / (1.0f + expf(-tmp_x.z));
        tmp_y.w = 1.0f / (1.0f + expf(-tmp_x.w));
        FLOAT4(y[idx]) = tmp_y;
    }
}

void check_results(float* cpu_result, float* gpu_result, int n) {
    bool pass = true;
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i] << std::endl;
            pass = false;
            break;
        }
    }
    if (pass) {
        std::cout << "Results match!\n";
    }
}

int main() {
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_y_float4 = new float[N];

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i) / N - 0.5f;  // 初始化輸入數據
    }

    float *d_x, *d_y, *d_y_float4;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_y_float4, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sigmoid<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, N);
    sigmoid_float4<<<blocks / 4, THREADS_PER_BLOCK>>>(d_x, d_y_float4, N);

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_float4, d_y_float4, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 計算基準
    float *cpu_y = new float[N];
    for (int i = 0; i < N; i++) {
        cpu_y[i] = 1.0f / (1.0f + expf(-h_x[i]));
    }

    std::cout << "Checking standard sigmoid kernel..." << std::endl;
    check_results(cpu_y, h_y, N);

    std::cout << "Checking vectorized sigmoid kernel..." << std::endl;
    check_results(cpu_y, h_y_float4, N);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_float4);
    delete[] h_x;
    delete[] h_y;
    delete[] h_y_float4;
    delete[] cpu_y;

    return 0;
}