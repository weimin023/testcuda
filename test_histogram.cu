#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

template<typename T> std::vector<T> genRandom(int N, int K, unsigned int seed = 1) {
    std::vector<T> res(N);

    std::mt19937 eng(seed);
    std::uniform_int_distribution<> dist(0, K);

    for (int i=0;i<N;++i) {
        res[i] = dist(eng);
    }

    return res;
}

template<typename T> __global__ void histogram_naive(const T *src, int N, int *bins, int n_bin) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gidx < N) {
        int bin = src[gidx] % n_bin;
        atomicAdd(&bins[bin], 1);
    }
}

template<typename T, int B>  __global__ void histogram_share(const T *src, int N, int *bins) {
    __shared__ int s_bins[B];

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int t = threadIdx.x; t < B; t += blockDim.x) {
        s_bins[t] = 0;
    }
    __syncthreads();

    if (gidx < N) {
        int bin_idx = src[gidx] % B;
        atomicAdd(&s_bins[bin_idx], 1);
    }

    for (int t = threadIdx.x; t < B; t += blockDim.x) {
        atomicAdd(&bins[t], s_bins[t]);
    }

}

template<typename T, int B>  __global__ void histogram_private(const T *src, int N, int *bins, int *private_bins) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gidx < N) {
        int bin_idx = src[gidx] % B;
        atomicAdd(&private_bins[blockIdx.x * B + bin_idx], 1);
    }
    __syncthreads();

    // write back shared memory hist to global mem
    for (int i = threadIdx.x; i < B; i += blockDim.x) {
        atomicAdd(&bins[i], private_bins[blockIdx.x * B + i]);
    }

}

int main() {
    const int N = 1024*1024;
    const int K = 9000;
    const int buckets = 1000;

    auto h_data = genRandom<int>(N, K);

    int *d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int *d_bins;
    cudaMalloc((void**)&d_bins, buckets * sizeof(int));

    int threadNum = 32;
    int blockNum = (N + threadNum - 1) / threadNum;

    printf("ThreadNum: %d\n", threadNum);
    printf("BlockNum: %d\n", blockNum);

    // 1. 
    histogram_naive<int><<<blockNum, threadNum>>>(d_data, N, d_bins, buckets);

    std::vector<int> h_bins(buckets);
    cudaMemcpy(h_bins.data(), d_bins, buckets * sizeof(int), cudaMemcpyDeviceToHost);

    for (int &i:h_bins) {
        std::cout<<i<<", ";
    }
    std::cout<<"========================"<<std::endl;

    // 2.
    int *d_bins2;
    cudaMalloc((void**)&d_bins2, buckets * sizeof(int));

    histogram_share<int, buckets><<<blockNum, threadNum>>>(d_data, N, d_bins2);

    cudaMemcpy(h_bins.data(), d_bins2, buckets * sizeof(int), cudaMemcpyDeviceToHost);
    for (int &i:h_bins) {
        std::cout<<i<<", ";
    }
    std::cout<<"========================"<<std::endl;

    // 3.
    int *d_bins3;
    cudaMalloc((void**)&d_bins3, buckets * sizeof(int));

    int *d_private_bins;
    cudaMalloc((void**)&d_private_bins, threadNum * blockNum * buckets * sizeof(int));

    histogram_private<int, buckets><<<blockNum, threadNum>>>(d_data, N, d_bins3, d_private_bins);

    cudaMemcpy(h_bins.data(), d_bins3, buckets * sizeof(int), cudaMemcpyDeviceToHost);
    for (int &i:h_bins) {
        std::cout<<i<<", ";
    }
    std::cout<<"========================"<<std::endl;
    return 0;
}