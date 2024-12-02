#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

#define TILE_DIM 30

__global__ void coalescedMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     transposedTile[TILE_DIM][TILE_DIM+3];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}

int main() {
    const int N = 100; // Define the matrix size (N x N)

    // Define input matrix (N x N)
    thrust::host_vector<float> h_a(N * N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i * N + j] = static_cast<float>(i + j + 1); // Populate the matrix with a pattern
        }
    }

    // Allocate device memory
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_c(N * N, 0);

    // Configure the block and grid dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    coalescedMultiply<<<gridDim, blockDim>>>(d_a.data().get(), 
                                             d_c.data().get(), 
                                             N);
    cudaDeviceSynchronize();

    // Copy the result back to host
    thrust::host_vector<float> h_c = d_c;

    // Print the result
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}