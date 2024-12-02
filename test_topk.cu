#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <float.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cmath>
#include <cuda_runtime.h>

__device__ inline void replace_smaller(float* array, int* idx_buffer, int k, float data, int data_idx) {
    if(data < array[k-1]) return;
        
    for(int j=k-2; j>=0; j--) {
        if(data > array[j]) {
            array[j+1] = array[j];
            idx_buffer[j+1] = idx_buffer[j];
        } else {
            array[j+1] = data;
            idx_buffer[j+1] = data_idx;
            return;
        }
    }

    array[0] = data;
    idx_buffer[0] = data_idx;
}

__device__ inline void mergeTwoK(float* left, int* left_idx, float* right, int* right_idx, int k) {
    int i;
    for(i=0; i<k; i++) {
        replace_smaller(left, left_idx, k, right[i], right_idx[i]);
    }
}

__device__ inline void replace_smaller_warp(volatile float* array, volatile int* idx_buffer, int k, float data, int data_idx) {
    if(data < array[k-1]) return;
        
    for(int j=k-2; j>=0; j--) {
        if(data > array[j]) {
            array[j+1] = array[j];
            idx_buffer[j+1] = idx_buffer[j];
        } else {
            array[j+1] = data;
            idx_buffer[j+1] = data_idx;
            return;
        }
    }

    array[0] = data;
    idx_buffer[0] = data_idx;
}

__device__ inline void mergeTwoK_warp(volatile float* left, volatile int* left_idx, volatile float* right, volatile int* right_idx, int k) {
    int i;
    for(i=0; i<k; i++) {
        replace_smaller_warp(left, left_idx, k, right[i], right_idx[i]);
    }
}

template<unsigned int numThreads>
__global__ void top_k_gpu_opt(float* input, int length, int k, float* output, int* out_idx) {
    extern __shared__ float shared_buffer[];

    int gtid = threadIdx.x + blockIdx.x * numThreads;
    int tid = threadIdx.x;
    float *buffer = shared_buffer + tid * k;
    int *buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + tid * k;

    int threadNum = gridDim.x * numThreads;

    for (int i = 0;i < k; ++i) {
        buffer[i] = -FLT_MAX;
        buffer_idx[i] = -1;
    }

    for (int i = gtid; i < length; i += threadNum) {
        replace_smaller(buffer, buffer_idx, k, input[i], i);
    }
    __syncthreads();

    /*for (int stride = numThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float* next_buffer = shared_buffer + (tid + stride) * k;
            int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + stride) * k;
            mergeTwoK(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        __syncthreads();
    }*/

    if (numThreads >= 1024) {
        if (tid < 512) {
            float* next_buffer = shared_buffer + (tid + 512) * k;
            int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 512) * k;
            mergeTwoK(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        __syncthreads();
    }
    if (numThreads >= 512) {
        if (tid < 256) {
            float* next_buffer = shared_buffer + (tid + 256) * k;
            int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 256) * k;
            mergeTwoK(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        __syncthreads();
    }
    if (numThreads >= 256) {
        if (tid < 128) {
            float* next_buffer = shared_buffer + (tid + 128) * k;
            int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 128) * k;
            mergeTwoK(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        __syncthreads();
    }
    if (numThreads >= 128) {
        if (tid < 64) {
            float* next_buffer = shared_buffer + (tid + 64) * k;
            int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 64) * k;
            mergeTwoK(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (numThreads >= 64) {
            volatile float* next_buffer = shared_buffer + (tid + 32) * k;
            volatile int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 32) * k;
            mergeTwoK_warp(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        if (numThreads >= 32) {
            volatile float* next_buffer = shared_buffer + (tid + 16) * k;
            volatile int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 16) * k;
            mergeTwoK_warp(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        if (numThreads >= 16) {
            volatile float* next_buffer = shared_buffer + (tid + 8) * k;
            volatile int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 8) * k;
            mergeTwoK_warp(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        if (numThreads >= 8) {
            volatile float* next_buffer = shared_buffer + (tid + 4) * k;
            volatile int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 4) * k;
            mergeTwoK_warp(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        if (numThreads >= 4) {
            volatile float* next_buffer = shared_buffer + (tid + 2) * k;
            volatile int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 2) * k;
            mergeTwoK_warp(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        if (numThreads >= 2) {
            volatile float* next_buffer = shared_buffer + (tid + 1) * k;
            volatile int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + numThreads * k) + (tid + 1) * k;
            mergeTwoK_warp(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }

        if (tid == 0) {
            for (int i = 0; i < k; ++i) {
                output[blockIdx.x * k + i] = buffer[i];
                out_idx[blockIdx.x * k + i] = buffer_idx[i];
            }
        }
    }
}

__global__ void top_k_gpu(float* input, int length, int k, float* output, int* out_idx) {
    extern __shared__ float shared_buffer[];

    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    float *buffer = shared_buffer + tid * k;
    int *buffer_idx = reinterpret_cast<int*>(shared_buffer + blockDim.x * k) + tid * k;

    int threadNum = gridDim.x * blockDim.x;

    for (int i = 0;i < k; ++i) {
        buffer[i] = -FLT_MAX;
        buffer_idx[i] = -1;
    }

    for (int i = gtid; i < length; i += threadNum) {
        replace_smaller(buffer, buffer_idx, k, input[i], i);
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float* next_buffer = shared_buffer + (tid + stride) * k;
            int* next_buffer_idx = reinterpret_cast<int*>(shared_buffer + blockDim.x * k) + (tid + stride) * k;
            mergeTwoK(buffer, buffer_idx, next_buffer, next_buffer_idx, k);
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            output[blockIdx.x * k + i] = buffer[i];
            out_idx[blockIdx.x * k + i] = buffer_idx[i];
        }
    }
}


__global__ void top_k_gpu_final(float* input, int* in_idx, int length, int k, float* output, int* out_idx) {
    extern __shared__ float shared_buffer[];

    int tid = threadIdx.x;
    int buckets = length / k;

    int *buffer_idx = reinterpret_cast<int*>(shared_buffer + length * sizeof(float));

    for (int i = tid; i < length; i += blockDim.x) {
        shared_buffer[i] = input[i];
        buffer_idx[i] = in_idx[i];
    }
    __syncthreads();

    for (int stride = buckets >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float *bucketPonter = shared_buffer + tid * k;
            float *bucketPonterNext = shared_buffer + (tid + stride) * k;

            int *bucketPonter_idx = buffer_idx + tid * k;
            int *bucketPonter_idxNext = buffer_idx + (tid + stride) * k;
            mergeTwoK(bucketPonter, bucketPonter_idx, bucketPonterNext, bucketPonter_idxNext, k);
        }
        __syncthreads();
    }

    if (tid == 0) {
        if ((buckets & 1) == 1 && buckets > 1) {
            float *bucketPonter = shared_buffer + tid * k;
            float *bucketPonterNext = shared_buffer + (tid + buckets - 1) * k;

            int *bucketPonter_idx = buffer_idx + tid * k;
            int *bucketPonter_idxNext = buffer_idx + (tid + buckets - 1) * k;
            mergeTwoK(bucketPonter, bucketPonter_idx, bucketPonterNext, bucketPonter_idxNext, k);
            __syncthreads();
        }

        for (int i = 0; i < k; ++i) {
            output[i] = shared_buffer[i];
            out_idx[i] = buffer_idx[i];
        }
    }
    
}

int main() {

    int n = 100;
    thrust::host_vector<float> h_vec(n);
    for (int i = 0; i < n; ++i) {
        h_vec[i] = static_cast<float>(rand() % 100);  // Random values between 0 and 99
    }

    std::cout << "DATA:" << std::endl;
    for (int i=0;i<n;++i) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << std::endl;

    thrust::device_vector<float> d_vec = h_vec;
    
    int k = 4;

    const int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * k * sizeof(int) + threadsPerBlock * k * sizeof(float);

    thrust::device_vector<float> d_res_first(k * blocksPerGrid);
    thrust::device_vector<int> d_idx_first(k * blocksPerGrid);

    thrust::device_vector<float> d_res_final(k);
    thrust::device_vector<int> d_idx_final(k);

    printf("N: %d, threads: %d, blocks: %d\n", n, threadsPerBlock, blocksPerGrid);

    top_k_gpu_opt<128><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_vec.data().get(), n, k, d_res_first.data().get(), d_idx_first.data().get());
    top_k_gpu_final<<<1, threadsPerBlock, sharedMemSize>>>(d_res_first.data().get(), d_idx_first.data().get(), blocksPerGrid*k, k, d_res_final.data().get(), d_idx_final.data().get());
    
    thrust::host_vector<float> h_res_first = d_res_first;
    thrust::host_vector<int> h_idx_first = d_idx_first;
    std::cout << "========FIRST" << std::endl;
    for (int i=0;i<blocksPerGrid;++i) {
        for (int j=0;j<k;++j) {
            int idx = i*k + j;
            std::cout<<"[" << h_idx_first[idx] << "]" << h_res_first[idx]<<", ";
        }
        std::cout<<std::endl;
    }

    thrust::host_vector<float> h_res_final = d_res_final;
    thrust::host_vector<int> h_idx_final = d_idx_final;
    std::cout << "========FINAL" << std::endl;
    for (int i=0;i<k;++i) {
        std::cout << "[" << h_idx_final[i] << "]" << h_res_final[i] << ", ";
    }

    std::cout << std::endl;

    thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<float>());
    thrust::host_vector<float> h_top_k(k);
    thrust::copy(d_vec.begin(), d_vec.begin() + k, h_top_k.begin());

    // Print the Top-K values
    std::cout << "========GT" << std::endl;
    for (auto val : h_top_k) {
        std::cout << val << ", ";
    }
    std::cout << std::endl;

}
