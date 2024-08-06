#include <gtest/gtest.h>
#include "evd.cuh"
#include "misc.cuh"
#include <random>

TEST(CUDAfunction, test_EVD) {

    // initialize EVD operator
    cuSyEVD<cuDoubleComplex, double> evd2(3);

    // There is one received signal "m_signals", the a cov matrix "m_cov"
    cuData_t<cuDoubleComplex> m_signals, m_cov;
    std::vector<cuDoubleComplex> h_B{
            {1.0, 0.0}, {1.0, 0.0}, {5.0, 2.5},
            {2.0, 1.0}, {2.0, 1.0}, {6.0, 3.0},
            {3.0, 1.5}, {3.0, 1.5}, {7.0, 3.5}
    };
    m_signals.assign_by_vec(h_B);
    m_signals.resize(3, 3);

    // allocate mamory for m_cov
    m_cov.allocate(3, 3);

    cublasHandle_t m_handle;
    if (cublasCreate_v2(&m_handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    // Get the covariance matrix of m_signals
    cublasStatus_t stat;
    stat = gemm_cuDoubleComplex(m_cov, m_handle, m_signals, CUBLAS_OP_N, 3, 3, m_signals, CUBLAS_OP_C, 3, 3, 1, 0);
            
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to compute covariance matrix");
    }

    // AV = VD (covariance mat * eigen vector = eigen vector * eigen values)
    auto left = matrixMultiply(cuda_cov2d, cuda_eigen_vec);
    auto right = matrixMultiply(cuda_eigen_vec, cuda_eigen_val);
    
    double err = 1e-5;
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            auto dis = cuComplexDistance(left[i][j], right[i][j]);
            EXPECT_NEAR(dis, 0, err);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

