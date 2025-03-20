#include <iostream>
#include <cuda/atomic>
#include <cuda_runtime.h>

__global__ void batch_naive_atomic_countOnesKernel(const int* input, int* result, int size) {
    int n_warp = blockDim.x/32;
    int n_iter = size/32/n_warp;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int reg_cnt = 0;

    for (int itr = 0; itr < n_iter; ++itr) {
        int curr_tid = blockDim.x * itr + tid;

        if (curr_tid < size && input[curr_tid] == 1) {
            reg_cnt++;
        }
    }

    if (reg_cnt > 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_res(*result);
        atomic_res.fetch_add(reg_cnt, cuda::memory_order_relaxed);
    }
}

__global__ void blocks_naive_atomic_countOnesKernel(const int* input, int* result, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        if (input[tid] == 1) {
            atomicAdd(result, 1);
        }
    }
}

int main() {
    const int size = 2048;  // 32*32
    int input[size];
    for (int i = 0; i < size; i++) {
        input[i] = (i % 2 == 0) ? 1 : 0;  // Fill the array with 1's and 0's
    }

    /*for (int i=0;i<32;++i) {
        for (int j=0;j<32;++j) {
            std::cout<<input[i*32+j]<<", ";
        }
        std::cout<<std::endl;
    }*/

    int* d_input, *d_result;
    int result = 0;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    int threadNum = 256;
    int blockNum = (size + threadNum - 1)/threadNum;
    std::cout<<"threads: "<<threadNum<<std::endl;
    std::cout<<"blocks: "<<blockNum<<std::endl;

    // 1. single block with naive atomic
    batch_naive_atomic_countOnesKernel<<<1, threadNum>>>(d_input, d_result, size);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Number of 1's in the array: " << result << std::endl;

    // 2. multi-block with naive atomic
    blocks_naive_atomic_countOnesKernel<<<blockNum, threadNum>>>(d_input, d_result, size);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Number of 1's in the array: " << result << std::endl;

    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
