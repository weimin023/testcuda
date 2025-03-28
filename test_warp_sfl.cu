#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void warp_reduce(int *dest, const int *src, int N) {
    
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    if (tid >= N) return;

    int n_iter = (N + WARP_SIZE - 1) / WARP_SIZE;

    int reg_sum = 0;
    for (int iter = 0; iter < n_iter; ++iter) {

        int idx = iter * WARP_SIZE + tid;
        int v = src[idx];
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }

        if (laneId == 0) reg_sum += v;
    }

    if (tid == 0) *dest = reg_sum;
            
}

__global__ void normal_reduce(int *dest, const int *src, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid > N) return;

    extern __shared__ int shared_block_sum[];

    if (tid < N) {
        shared_block_sum[threadIdx.x] = src[tid];
    } else {
        shared_block_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int strip = blockDim.x / 2; strip > 0; strip /= 2) {
        if (threadIdx.x < strip) {
            shared_block_sum[threadIdx.x] += shared_block_sum[threadIdx.x + strip];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(dest, shared_block_sum[0]);
    
}

__global__ void normal_reduce_opt(int *dest, const int *src, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;

    extern __shared__ int shared_block_sum[];

    if (tid < N) {
        shared_block_sum[threadIdx.x] = src[tid];
    } else {
        shared_block_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    int val = shared_block_sum[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (laneId == 0) shared_block_sum[0] = val;

    if (threadIdx.x == 0) atomicAdd(dest, shared_block_sum[0]);
    
}

__global__ void normal_reduce_reg_opt(int *dest, const int *src, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;

    int reg_block_sum[WARP_SIZE];

    if (tid < N) {
        reg_block_sum[threadIdx.x] = src[tid];
    } else {
        reg_block_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    int val = reg_block_sum[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (laneId == 0) reg_block_sum[0] = val;

    if (threadIdx.x == 0) atomicAdd(dest, reg_block_sum[0]);
    
}

void randomGen(int *dist, int n) {
    std::random_device rd;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(0, 1000);

    for (int i=0;i<n;++i) {
        dist[i] = dis(gen);
    }
}

int main() {
    const int N = 1024;

    int h_data[N];
    randomGen(h_data, N);

    // CPU
    int res_cpu = 0;
    for (int i=0;i<N;++i) res_cpu += h_data[i];
    std::cout << "CPU Result: " << res_cpu << std::endl;

    // GPU warp sfl
    int *d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    int h_res_warp = 0;
    int *d_res_warp;
    cudaMalloc((void**)&d_res_warp, 1 * sizeof(int));
    cudaMemcpy(d_res_warp, &h_res_warp, 1 * sizeof(int), cudaMemcpyHostToDevice);
    warp_reduce<<<1, 32>>>(d_res_warp, d_data, N);
    cudaMemcpy(&h_res_warp, d_res_warp, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU Warp Result: " << h_res_warp << std::endl;

    // GPU normal reduce
    int threadNum = 32;
    int blockNum = (threadNum + N - 1) / threadNum; // 32

    int h_res_normal = 0;
    int *d_res_normal;
    cudaMalloc((void**)&d_res_normal, 1 * sizeof(int));
    cudaMemcpy(d_res_normal, &h_res_normal, 1 * sizeof(int), cudaMemcpyHostToDevice);
    normal_reduce<<<blockNum, threadNum, blockNum * sizeof(int)>>>(d_res_normal, d_data, N);
    cudaMemcpy(&h_res_normal, d_res_normal, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU Normal Result: " << h_res_normal << std::endl;

    // GPU reg opt
    int h_res_reg = 0;
    int *d_res_reg;
    cudaMalloc((void**)&d_res_reg, sizeof(int));
    cudaMemcpy(d_res_reg, &h_res_reg, sizeof(int), cudaMemcpyHostToDevice);
    normal_reduce_reg_opt<<<blockNum, threadNum>>>(d_res_reg, d_data, N);
    cudaMemcpy(&h_res_reg, d_res_reg, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "GPU Reg Result: " << h_res_reg << std::endl;
}