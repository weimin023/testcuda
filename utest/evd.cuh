#pragma once
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <thrust/device_vector.h>

namespace TronCUDA {
    template <typename T, typename D>
    class cuSyEVD {
    public:
        cuSyEVD() {
            cusolverDnCreate(&cusolverH);
        }
        ~cuSyEVD() {
            cusolverDnDestroy(cusolverH);
            cudaFree(d_work);
            cudaFree(devInfo);
        }

        void setup(int n, cudaStream_t stream = 0) {
            N = n;
            m_eigen_vec.resize(N*N);
            m_eigen_vec_row_majored.resize(N*N);
            m_eigen_value.resize(N);
            (cudaMalloc(&devInfo, sizeof(int)));
            m_stream = stream;
            (cusolverDnSetStream(cusolverH, m_stream));
            preprocess();
        }
        const auto &GetEigenVec() { return m_eigen_vec_row_majored; }
        const auto &GetEigenVal() { return m_eigen_value; }

        void exec(const thrust::device_vector<T> &d_data);
    private:
        void preprocess();
        void toRow();

        cudaStream_t m_stream;
        cusolverDnHandle_t cusolverH;
        thrust::device_vector<T> m_eigen_vec;
        thrust::device_vector<T> m_eigen_vec_row_majored;
        thrust::device_vector<D> m_eigen_value;

        T *d_work;
        int *devInfo;
        int lwork = 0;
        int N;
    };
}
