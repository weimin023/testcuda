#include "misc.cuh"
#include <cublas_v2.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>



double cuComplexDistance(const cuDoubleComplex& z1, const cuDoubleComplex& z2) {
    double x1 = cuCreal(z1);
    double y1 = cuCimag(z1);
    double x2 = cuCreal(z2);
    double y2 = cuCimag(z2);

    double real_diff = x2 - x1;
    double imag_diff = y2 - y1;

    return std::sqrt(real_diff * real_diff + imag_diff * imag_diff);
}

std::vector<std::vector<cuDoubleComplex>> matrixMultiply(
    const std::vector<std::vector<cuDoubleComplex>>& A,
    const std::vector<std::vector<cuDoubleComplex>>& B) {

    // Get dimensions of the matrices
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    // Check if matrices can be multiplied
    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // Initialize the result matrix with zeros
    std::vector<std::vector<cuDoubleComplex>> C(rowsA, std::vector<cuDoubleComplex>(colsB, make_cuDoubleComplex(0.0, 0.0)));

    // Perform matrix multiplication
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
            for (size_t k = 0; k < colsA; ++k) {
                sum = cuCadd(sum, cuCmul(A[i][k], B[k][j]));
            }
            C[i][j] = sum;
        }
    }

    return C;
}

cublasStatus_t gemm_cuDoubleComplex(cuDataZ &out, cublasHandle_t &handle, const cuDataZ &in1, cublasOperation_t trans1, int row1, int col1, const cuDataZ &in2, cublasOperation_t trans2, int row2, int col2, float alpha, float beta) {
    cuDoubleComplex alpha_ = make_cuDoubleComplex(alpha, 0);
    cuDoubleComplex beta_  = make_cuDoubleComplex(beta, 0);

    if (trans1 == CUBLAS_OP_N && trans2 != CUBLAS_OP_N) {
        return cublasZgemm(handle, trans2, trans1,
                            col2, row1, col1,
                            &alpha_,
                            in2.data(), row2,
                            in1.data(), col1,
                            &beta_,
                            out.data(), col2);
    } else if (trans1 == CUBLAS_OP_N && trans2 == CUBLAS_OP_N) {
        return cublasZgemm(handle, trans2, trans1,
                            col2, row1, col1,
                            &alpha_,
                            in2.data(), col2,
                            in1.data(), col1,
                            &beta_,
                            out.data(), col2);
    } else if (trans1 != CUBLAS_OP_N && trans2 == CUBLAS_OP_N) {
        return cublasZgemm(handle, trans2, trans1,
                            col2, row1, col1,
                            &alpha_,
                            in2.data(), col2,
                            in1.data(), row1,
                            &beta_,
                            out.data(), col2);
    } else {
        throw std::invalid_argument("Unsupported transposition combination in gemm_cuDoubleComplex.");
    }
}

template <typename T> __global__ void set_value_kernel(T *data, T value, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) { 
        data[idx] = value;
    }
}

template <typename T> void set_value_x(T value, T *mem, size_t size, size_t offset, size_t target_size, cudaStream_t stream = 0) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    if (target_size == 0) target_size = (size - offset);
    set_value_kernel<<<numBlocks, blockSize, 0, stream>>>(mem + offset, value, size);
    cudaDeviceSynchronize();
}

template <> void cuData_t<float>::set_value(float value, size_t offset, size_t target_size) { set_value_x(value, m_internal_mem_, size(), offset, target_size); }
template <> void cuData_t<double>::set_value(double value, size_t offset, size_t target_size) { set_value_x(value, m_internal_mem_, size(), offset, target_size); }
template <> void cuData_t<cuComplex>::set_value(cuComplex value, size_t offset, size_t target_size) { set_value_x(value, m_internal_mem_, size(), offset, target_size); }
template <> void cuData_t<cuDoubleComplex>::set_value(cuDoubleComplex value, size_t offset, size_t target_size) { set_value_x(value, m_internal_mem_, size(), offset, target_size); }
