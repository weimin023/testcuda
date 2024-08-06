#include "evd.cuh"

template <> void cuSyEVD<cuDoubleComplex, double>::exec(cuDataZ &A) {
    m_eigen_vec.assign_by_ref(A);
    cusolverDnZheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, m_eigen_vec.data(), N, m_eigen_value.data(), d_work, lwork, devInfo);

    int info;
    (cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    // if (info != 0) { std::cerr << "Error: cuSOLVER operation failed with info = " << info << std::endl; }
}

template <> void cuSyEVD<cuDoubleComplex, double>::preprocess() {
    (cusolverDnCreate(&cusolverH));

    m_eigen_vec.allocate(N, N);
    m_eigen_value.allocate(N);

    (cusolverDnZheevd_bufferSize(
        cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N,
        m_eigen_vec.data(), N, m_eigen_value.data(), &lwork));

    (cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
    (cudaMalloc(&devInfo, sizeof(int)));
}