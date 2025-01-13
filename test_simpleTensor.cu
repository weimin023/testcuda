#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 64
#define N_TILES 64
#define K_TILES 64

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define WARP_SIZE 32

__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld, int n_ld, int k_ld, float alpha, float beta) {
    int warp_row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_col = (blockIdx.y * blockDim.y + threadIdx.y);

    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k_ld; i += WMMA_K) {
      int aCol = i;
      int aRow = warp_row * WMMA_M;
      int bCol = warp_col * WMMA_N;
      int bRow = i;

      if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
        wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
        wmma::load_matrix_sync(b_frag, b + bCol * ldb + bRow, ldb);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
    }

    int cCol = warp_col * WMMA_N;
    int cRow = warp_row * WMMA_M;

    if (cCol < n_ld && cRow < m_ld) {
      wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

      for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

__host__ void init_host_matrices(half *a, half *b, float *c) {
    for (int i = 0; i < M_GLOBAL; i++) {
      for (int j = 0; j < K_GLOBAL; j++) {
        a[i * K_GLOBAL + j] = (half)(rand() % 3);
      }
    }
  
    for (int i = 0; i < N_GLOBAL; i++) {
      for (int j = 0; j < K_GLOBAL; j++) {
        b[i * K_GLOBAL + j] = (half)(rand() % 3);
      }
    }
  
    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
      c[t] = static_cast<float>(rand() % 3);
    }
}

__host__ void matMultiplyOnHost(half *A, half *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

int main() {

    printf("Initializing...\n");
    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

#if CPU_DEBUG
    thrust::host_vector<float> result_hD(M_GLOBAL * N_GLOBAL);
    thrust::host_vector<float> result_host(M_GLOBAL * N_GLOBAL);
#endif

    thrust::host_vector<__half> h_a(M_GLOBAL * K_GLOBAL);
    thrust::host_vector<__half> h_b(K_GLOBAL * N_GLOBAL);
    thrust::host_vector<float> h_c(M_GLOBAL * N_GLOBAL);
    init_host_matrices(h_a.data(), h_b.data(), h_c.data());

    printf("Preparing data for GPU...\n");
    thrust::device_vector<__half> d_a = h_a;
    thrust::device_vector<__half> d_b = h_b;
    thrust::device_vector<float> d_c = h_c;
    thrust::device_vector<float> d_d(M_GLOBAL * N_GLOBAL, 0);

    const float alpha = 1.1f;
    const float beta = 1.2f;

    cudaEvent_t start, stop;

    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));
    (cudaEventRecord(start));

    dim3 blockV4(128, 4);
    dim3 gridV4;

    gridV4.x = (M_GLOBAL + (WMMA_M * blockV4.x / 32 - 1)) / (WMMA_M * blockV4.x / 32);
    gridV4.y = (N_GLOBAL + WMMA_N * blockV4.y - 1) / (WMMA_N * blockV4.y);

    simple_wmma_gemm<<<gridV4, blockV4>>>(d_a.data().get(), d_b.data().get(), d_c.data().get(), d_d.data().get(), M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);

    (cudaEventRecord(stop));
    (cudaEventSynchronize(stop));

#if CPU_DEBUG
    result_hD = d_d;
#endif

#if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");

    result_host = h_c;
    matMultiplyOnHost(h_a.data(), h_b.data(), result_host.data(), alpha, beta, M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
        if (fabs(result_hD[i] - result_host[i]) > 0.1f)
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], result_host[i]);
    }

#endif

    float milliseconds = 0;

    (cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);
    printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);

}