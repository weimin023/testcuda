#include "evd.cuh"
#include "../sharedmem.h"

namespace TronBLAS {
    template <typename T> 
    __global__ void transpose_kernal(T *in, T *out, int m, int n) {

        SharedMemory<T> smem;
        T *shm = smem.getPointer();
        
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < m && y < n) {
            shm[x + y*m] = in[x + y*m];
            
            __syncthreads();

            out[x*n + y] = shm[x + y*m];
        }
    }
}

namespace TronCUDA {
    template <> void cuSyEVD<float, float>::preprocess() {
        (cusolverDnSsyevd_bufferSize(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N,
            m_eigen_vec.data().get(), N, m_eigen_value.data().get(), &lwork));
    
        (cudaMalloc(&d_work, lwork * sizeof(float)));
    }
    
    template <> void cuSyEVD<cuComplex, float>::preprocess() {
        (cusolverDnCheevd_bufferSize(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N,
            m_eigen_vec.data().get(), N, m_eigen_value.data().get(), &lwork));
    
        cudaMalloc(&d_work, lwork * sizeof(cuComplex));
    }
    
    template <> void cuSyEVD<cuDoubleComplex, double>::preprocess() {
        (cusolverDnZheevd_bufferSize(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N,
            m_eigen_vec.data().get(), N, m_eigen_value.data().get(), &lwork));
    
        (cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
    }
    
    template <> void cuSyEVD<cuFloatComplex, float>::exec(const thrust::device_vector<cuFloatComplex> &d_data) {
        m_eigen_vec = d_data;
        (cusolverDnCheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, m_eigen_vec.data().get(), N, m_eigen_value.data().get(), d_work, lwork, devInfo));
    
        int info;
        (cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) { std::cerr << "Error: cuSOLVER operation failed with info = " << info << std::endl; }
    
        toRow();
    }
    
    template <> void cuSyEVD<cuDoubleComplex, double>::exec(const thrust::device_vector<cuDoubleComplex> &d_data) {
        m_eigen_vec = d_data;
        (cusolverDnZheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, m_eigen_vec.data().get(), N, m_eigen_value.data().get(), d_work, lwork, devInfo));
    
        int info;
        (cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) { std::cerr << "Error: cuSOLVER operation failed with info = " << info << std::endl; }
    
        toRow();
    }
    
    template <typename T, typename D>
    void cuSyEVD<T, D>::toRow() {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
        size_t shm_size = blockSize.x * blockSize.y * sizeof(T);
        TronBLAS::transpose_kernal<<<gridSize, blockSize, shm_size, m_stream>>>(m_eigen_vec.data().get(), m_eigen_vec_row_majored.data().get(), N, N);
    }
}
