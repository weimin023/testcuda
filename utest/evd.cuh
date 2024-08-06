#pragma once
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <stdexcept>
#include <cublas_v2.h>
#include "misc.cuh"

template <typename T, typename D>
class cuSyEVD {
public:
    cuSyEVD(uint n) : N(n) {
        preprocess();
    }
    ~cuSyEVD() {
        cusolverDnDestroy(cusolverH);
        m_eigen_vec.free();
        m_eigen_value.free();
        cudaFree(d_work);
        cudaFree(devInfo);
    }

    const auto &V() { return m_eigen_vec; }
    const auto &W() { return m_eigen_value; }

    void exec(cuData_t<T> &A);

private:
    void preprocess();

    cusolverDnHandle_t cusolverH;
    cuData_t<T> m_eigen_vec;
    cuData_t<D> m_eigen_value;
    T *d_work;
    int *devInfo;
    int lwork = 0;
    int N;
};