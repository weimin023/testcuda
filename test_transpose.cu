#include "common.cu"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

const int TILE_DIM = 32;
const int TILE_SIZE = 16;
const int BLOCK_ROWS = 8;

void transpose_cpu(float *out, const float *in, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out[i + j * N] = in[i * N + j];
        }
    }
}

__global__ void transpose_naive_tile(float *out, const float *in, int M, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    for (int j=0;j<TILE_SIZE;j+=BLOCK_ROWS) {
        if ((row + j) < M && col < N)
            out[(row + j) + col * N] = in[(row + j)*N + col];
    }
}

__global__ void transpose_naive(float *out, const float *in, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        out[row + col * N] = in[row * N + col];
    }
}

__global__ void transpose_coalesced(float *out, const float *in) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = TILE_DIM * gridDim.x;

    for (int j=0;j<TILE_DIM;j+=BLOCK_ROWS) {
        tile[threadIdx.y+j][threadIdx.x] = in[x + (j + y) * width];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
    std::default_random_engine random_engine(0);

    const int M = 1024;
    const int N = 1024;

    // input
    thrust::host_vector<float> h_data(M*N);
    fill_random_float_values(h_data.data(), M*N, random_engine);

    // CPU
    std::vector<float> h_out_cpu(M*N);
    transpose_cpu(h_out_cpu.data(), h_data.data(), M, N);

    // naive tile
    thrust::device_vector<float> d_data = h_data;
    thrust::device_vector<float> d_out_naive_tile(M*N);

    dim3 blockSize(16, 16);
    dim3 gridSize(M/TILE_SIZE, N/TILE_SIZE);
    transpose_naive_tile<<<gridSize, blockSize>>>(d_out_naive_tile.data().get(), d_data.data().get(), M, N);

    // naive
    thrust::device_vector<float> d_out_native(M*N);
    dim3 blockSize2(16, 16);
    dim3 gridSize2((N + blockSize2.x - 1)/blockSize2.x, (M + blockSize2.y - 1)/blockSize2.y);
    transpose_naive<<<gridSize2, blockSize2>>>(d_out_native.data().get(), d_data.data().get(), M, N);

    // naive coalesced
    thrust::device_vector<float> d_out_native_coalesced(M*N);
    dim3 blockSize3(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize3(M/TILE_DIM, N/TILE_DIM);
    transpose_coalesced<<<gridSize3, blockSize3>>>(d_out_native_coalesced.data().get(), d_data.data().get());

    // compare
    bool flg = 1;
    thrust::host_vector<float> h_out_naive_tile = d_out_naive_tile;
    thrust::host_vector<float> h_out_naive = d_out_native;
    thrust::host_vector<float> h_out_naive_coalesced = d_out_native_coalesced;

    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            if (h_out_naive_tile[i*N+j] != h_out_cpu[i*N+j] || h_out_naive[i*N+j] != h_out_cpu[i*N+j] || h_out_naive_coalesced[i*N+j] != h_out_cpu[i*N+j]) {
                flg = false;
                break;
            }
        }
        if (!flg) break;
    }

    if (!flg) std::cout << "Failed.\n";

    return 0;
}