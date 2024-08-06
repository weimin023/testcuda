#include <vector>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include "CUDA1052.cuh"

__global__ void scan(const int *customers, const int *grumpy, int n, int *result) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid < n) {
        result[tid] = (grumpy[tid] == 0) ? (customers[tid]):0;
    }
}

__global__ void scan2(const int *customers, const int *grumpy, int n, const int window, int *result) {
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid < (n-window+1)) {
        int sum = 0;
        for (int idx = tid; idx < tid + window; ++idx) {
            sum += (grumpy[idx] == 1) ? (customers[idx]):0;
        }
        result[tid+2] = sum;
    }
}

int maxSatisfied(std::vector<int>& customers, std::vector<int>& grumpy, int minutes) {
    //std::vector<int> customers{1,0,1,2,1,1,7,5};
    //std::vector<int> grumpy{0,1,0,1,0,1,0,1};
    int N_ = customers.size();

    int *d_scan_result, *d_scan2_result, *d_scan_local, *d_customers, *d_grumpy;
    cudaMalloc((void**)&d_scan_result, sizeof(int)*N_);
    cudaMalloc((void**)&d_scan_local, sizeof(int)*N_);
    cudaMalloc((void**)&d_customers, sizeof(int)*N_);
    cudaMalloc((void**)&d_grumpy, sizeof(int)*N_);
    cudaMalloc((void**)&d_scan2_result, sizeof(int)*N_);

    cudaMemcpy(d_customers, customers.data(), sizeof(int)*N_, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grumpy, grumpy.data(), sizeof(int)*N_, cudaMemcpyHostToDevice);

    int threadNum = 32;
    int blockNum = (N_ + threadNum-1)/threadNum;
    scan<<<blockNum, threadNum>>>(d_customers, d_grumpy, N_, d_scan_result);
    scan2<<<blockNum, threadNum>>>(d_customers, d_grumpy, N_, minutes, d_scan2_result);

    thrust::device_ptr<int> d_scan_result_ptr(d_scan_result);
    int sum = thrust::reduce(d_scan_result_ptr, d_scan_result_ptr + N_);

    thrust::device_ptr<int> d_scan2_result_ptr(d_scan2_result);
    int max_value = *thrust::max_element(d_scan2_result_ptr, d_scan2_result_ptr + N_);

    cudaFree(d_scan_result);
    cudaFree(d_scan2_result);
    cudaFree(d_scan_local);
    cudaFree(d_customers);
    cudaFree(d_grumpy);

    return sum + max_value;
}