#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <chrono>
#include "cuda_fp16.h"

#define N 500000

__device__ static __inline__ cuFloatComplex cuCmulfOpt(const cuFloatComplex a, const cuFloatComplex b) {
    return make_cuFloatComplex(__hsub(__hmul(a.x, b.x), __hmul(a.y, b.y)), __hfma(a.x, b.y, __hmul(a.y, b.x)));
    // __hfma(a.x, b.x, __hneg(__hmul(a.y, b.y)));
}

__global__ void kernel_cuCmulf(cuFloatComplex *out, const cuFloatComplex *a, const cuFloatComplex *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = cuCmulf(a[idx], b[idx]);
    }
}

__global__ void kernel_cuCmulfOpt(cuFloatComplex *out, const cuFloatComplex *a, const cuFloatComplex *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = cuCmulfOpt(a[idx], b[idx]);  // using out[idx] as a dummy third parameter
    }
}

void profileFunction(void (*kernel)(cuFloatComplex*, const cuFloatComplex*, const cuFloatComplex*), 
                     cuFloatComplex *d_out, const cuFloatComplex *d_a, const cuFloatComplex *d_b) {
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch the kernel multiple times to accumulate enough time for profiling
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1; i++) {
        kernel<<<gridSize, blockSize>>>(d_out, d_a, d_b);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken: " << duration.count() / 100 << " ms per kernel call" << std::endl;
}

int main() {
    cuFloatComplex *h_a = new cuFloatComplex[N];
    cuFloatComplex *h_b = new cuFloatComplex[N];
    cuFloatComplex *h_out = new cuFloatComplex[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = make_cuFloatComplex(1.0f, 2.0f);
        h_b[i] = make_cuFloatComplex(3.0f, 4.0f);
    }

    cuFloatComplex *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(cuFloatComplex));
    cudaMalloc(&d_b, N * sizeof(cuFloatComplex));
    cudaMalloc(&d_out, N * sizeof(cuFloatComplex));

    cudaMemcpy(d_a, h_a, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    std::cout << "Profiling cuCmulf:" << std::endl;
    profileFunction(kernel_cuCmulf, d_out, d_a, d_b);

    std::cout << "Profiling cuCmulfOpt:" << std::endl;
    profileFunction(kernel_cuCmulfOpt, d_out, d_a, d_b);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;

    return 0;
}