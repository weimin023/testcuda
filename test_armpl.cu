#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <chrono>
#include <thrust/host_vector.h>
#include "armpl.h"
#include <complex.h> 

#define N 8

#define CHECK_CUDA(call)                                                                                                                                                                               \
    do {                                                                                                                                                                                               \
        cudaError_t err = call;                                                                                                                                                                        \
        if (err != cudaSuccess) {                                                                                                                                                                      \
            fprintf(stderr, "Cuda failed with error code %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__);                                                                    \
            exit(EXIT_FAILURE);                                                                                                                                                                        \
        }                                                                                                                                                                                              \
    } while (0)

#define CHECK_CUSOLVER(call)                                                                                                                                                                           \
    do {                                                                                                                                                                                               \
        cusolverStatus_t stat = call;                                                                                                                                                                  \
        if (stat != CUSOLVER_STATUS_SUCCESS) {                                                                                                                                                         \
            fprintf(stderr, "CuSolver failed with error code %s at line %d in file %s\n", cusolverGetErrorString(stat), __LINE__, __FILE__);                                                           \
            exit(EXIT_FAILURE);                                                                                                                                                                        \
        }                                                                                                                                                                                              \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                                                                                                               \
    do {                                                                                                                                                                                               \
        cublasStatus_t err = call;                                                                                                                                                                        \
        if (err != CUBLAS_STATUS_SUCCESS) {                                                                                                                                                                      \
            fprintf(stderr, "cuBLAS failed with error code %s at line %d in file %s\n", cublasGetStatusName(err), __LINE__, __FILE__);                                                                    \
            exit(EXIT_FAILURE);                                                                                                                                                                        \
        }                                                                                                                                                                                              \
    } while (0)

#define CHECK_LAPACK(call)       \
    do {                         \
        armpl_int_t stat = call; \
        if (stat != 0) {         \
            fprintf(stderr, "LAPACK Function Failed at line %d in file %s\n", __LINE__, __FILE__); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)                                                                                    \


const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:
            return "Unknown cuSOLVER error";
    }
}

cuFloatComplex h_cov[64] = {
        make_cuFloatComplex(4140.65f, -2.24197e-06f), make_cuFloatComplex(137.897f, 3938.01f),
        make_cuFloatComplex(-158.647f, -1241.86f), make_cuFloatComplex(-1936.52f, 2616.06f),
        make_cuFloatComplex(-743.871f, 2664.88f), make_cuFloatComplex(623.323f, -4050.98f),
        make_cuFloatComplex(-1115.17f, 1268.56f), make_cuFloatComplex(-1067.13f, 133.398f),
        make_cuFloatComplex(137.897f, -3938.01f), make_cuFloatComplex(4154.02f, 3.32665e-06f),
        make_cuFloatComplex(-1563.14f, -967.872f), make_cuFloatComplex(2961.04f, 2489.75f),
        make_cuFloatComplex(3325.31f, 1229.58f), make_cuFloatComplex(-3941.03f, -737.05f),
        make_cuFloatComplex(1463.98f, 2163.99f), make_cuFloatComplex(-995.137f, 1257.28f),
        make_cuFloatComplex(-158.647f, 1241.86f), make_cuFloatComplex(-1563.14f, 967.872f),
        make_cuFloatComplex(3843.31f, -2.39818e-07f), make_cuFloatComplex(-2763.36f, 255.503f),
        make_cuFloatComplex(-2805.64f, 1415.83f), make_cuFloatComplex(1326.26f, 228.303f),
        make_cuFloatComplex(-3621.11f, -722.38f), make_cuFloatComplex(584.441f, -3733.83f),
        make_cuFloatComplex(-1936.52f, -2616.06f), make_cuFloatComplex(2961.04f, -2489.75f),
        make_cuFloatComplex(-2763.36f, -255.503f), make_cuFloatComplex(4067.95f, -3.1887e-06f),
        make_cuFloatComplex(3742.5f, -1311.93f), make_cuFloatComplex(-2992.45f, 1599.3f),
        make_cuFloatComplex(3214.86f, 1165.23f), make_cuFloatComplex(-611.432f, 2481.71f),
        make_cuFloatComplex(-743.871f, -2664.88f), make_cuFloatComplex(3325.31f, -1229.58f),
        make_cuFloatComplex(-2805.64f, -1415.83f), make_cuFloatComplex(3742.5f, 1311.94f),
        make_cuFloatComplex(4021.9f, -3.66673e-06f), make_cuFloatComplex(-2973.91f, 342.678f),
        make_cuFloatComplex(2854.54f, 2360.54f), make_cuFloatComplex(-1725.97f, 2476.39f),
        make_cuFloatComplex(623.323f, 4050.98f), make_cuFloatComplex(-3941.03f, 737.05f),
        make_cuFloatComplex(1326.26f, -228.303f), make_cuFloatComplex(-2992.45f, -1599.3f),
        make_cuFloatComplex(-2973.91f, -342.678f), make_cuFloatComplex(4204.25f, 3.96076e-06f),
        make_cuFloatComplex(-1613.83f, -1059.15f), make_cuFloatComplex(-198.559f, -1266.46f),
        make_cuFloatComplex(-1115.17f, -1268.56f), make_cuFloatComplex(1463.98f, -2163.99f),
        make_cuFloatComplex(-3621.11f, 722.38f), make_cuFloatComplex(3214.86f, -1165.23f),
        make_cuFloatComplex(2854.54f, -2360.54f), make_cuFloatComplex(-1613.83f, 1059.15f),
        make_cuFloatComplex(3900.07f, -1.04732e-06f), make_cuFloatComplex(171.296f, 3648.69f),
        make_cuFloatComplex(-1067.13f, -133.398f), make_cuFloatComplex(-995.137f, -1257.28f),
        make_cuFloatComplex(584.441f, 3733.84f), make_cuFloatComplex(-611.432f, -2481.71f),
        make_cuFloatComplex(-1725.97f, -2476.39f), make_cuFloatComplex(-198.559f, 1266.46f),
        make_cuFloatComplex(171.296f, -3648.69f), make_cuFloatComplex(3863.89f, -2.28193e-06f)
    };

lapack_complex_float h_cov_arm[N * N] = {
    {4140.65f, -2.24197e-06f}, {137.897f, 3938.01f}, {-158.647f, -1241.86f}, {-1936.52f, 2616.06f},
    {-743.871f, 2664.88f}, {623.323f, -4050.98f}, {-1115.17f, 1268.56f}, {-1067.13f, 133.398f},
    {137.897f, -3938.01f}, {4154.02f, 3.32665e-06f}, {-1563.14f, -967.872f}, {2961.04f, 2489.75f},
    {3325.31f, 1229.58f}, {-3941.03f, -737.05f}, {1463.98f, 2163.99f}, {-995.137f, 1257.28f},
    {-158.647f, 1241.86f}, {-1563.14f, 967.872f}, {3843.31f, -2.39818e-07f}, {-2763.36f, 255.503f},
    {-2805.64f, 1415.83f}, {1326.26f, 228.303f}, {-3621.11f, -722.38f}, {584.441f, -3733.83f},
    {-1936.52f, -2616.06f}, {2961.04f, -2489.75f}, {-2763.36f, -255.503f}, {4067.95f, -3.1887e-06f},
    {3742.5f, -1311.93f}, {-2992.45f, 1599.3f}, {3214.86f, 1165.23f}, {-611.432f, 2481.71f},
    {-743.871f, -2664.88f}, {3325.31f, -1229.58f}, {-2805.64f, -1415.83f}, {3742.5f, 1311.94f},
    {4021.9f, -3.66673e-06f}, {-2973.91f, 342.678f}, {2854.54f, 2360.54f}, {-1725.97f, 2476.39f},
    {623.323f, 4050.98f}, {-3941.03f, 737.05f}, {1326.26f, -228.303f}, {-2992.45f, -1599.3f},
    {-2973.91f, -342.678f}, {4204.25f, 3.96076e-06f}, {-1613.83f, -1059.15f}, {-198.559f, -1266.46f},
    {-1115.17f, -1268.56f}, {1463.98f, -2163.99f}, {-3621.11f, 722.38f}, {3214.86f, -1165.23f},
    {2854.54f, -2360.54f}, {-1613.83f, 1059.15f}, {3900.07f, -1.04732e-06f}, {171.296f, 3648.69f},
    {-1067.13f, -133.398f}, {-995.137f, -1257.28f}, {584.441f, 3733.84f}, {-611.432f, -2481.71f},
    {-1725.97f, -2476.39f}, {-198.559f, 1266.46f}, {171.296f, -3648.69f}, {3863.89f, -2.28193e-06f}
};

class evaluate_evd {
public:
    evaluate_evd() {
	cusolverDnCreate(&m_cusolverH);

	cudaMalloc((void**)&d_eigen_vec, N * N * sizeof(cuFloatComplex));
        cudaMalloc((void**)&d_eigen_val, N * sizeof(float));

        CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
        CHECK_CUSOLVER(cusolverDnCheevd_bufferSize(m_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, d_eigen_vec, N, d_eigen_val, &lwork));
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(cuComplex)));
    
	t_cuda  = 0;
        t_armpl = 0;
    }
    void run_cuda(const cuFloatComplex *d_cov) {
        CHECK_CUDA(cudaMemcpy(d_eigen_vec, d_cov, N * N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    
        auto start = std::chrono::high_resolution_clock::now();
        CHECK_CUSOLVER(cusolverDnCheevd(m_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, d_eigen_vec, N, d_eigen_val, d_work, lwork, devInfo));
        auto end = std::chrono::high_resolution_clock::now();

	t_cuda += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        int info;
        CHECK_CUDA(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            std::cerr << "Error: cuSOLVER operation failed with info = " << info << std::endl;
        }

	thrust::host_vector<cuFloatComplex> h_eigen_vec(N*N);
        thrust::host_vector<float> h_eigen_val(N);
        cudaMemcpy(h_eigen_vec.data(), d_eigen_vec, N * N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eigen_val.data(), d_eigen_val, N * sizeof(float), cudaMemcpyDeviceToHost);

        //std:: cout << "=========CHECK EIGENVALS=========" << std::endl;
        //for (int i=0;i<N;++i) {
        //    std::cout << h_eigen_val[i] << std::endl;
        //}

    }
    void run_armpl(const lapack_complex_float *cov_arm) {
        lapack_complex_float cov_in[N*N];
	std::copy(cov_arm, cov_arm + (N * N), cov_in);

	auto start = std::chrono::high_resolution_clock::now();
	CHECK_LAPACK(LAPACKE_cheev(LAPACK_ROW_MAJOR, 'V', 'U', N, cov_in, N, w));
	auto end = std::chrono::high_resolution_clock::now();

	t_armpl += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        //std::cout<<"==========LAPACK:"<<std::endl;
        //for (int i=0;i<N;++i) {
        //    std::cout<<w[i]<<std::endl;
        //}
    }

    float getCudaTime() { return t_cuda; }
    float getArmplTime() { return t_armpl; }
private:
    float w[N];

    cuFloatComplex *d_work;
    int *devInfo;
    int lwork = 0;
    cusolverDnHandle_t m_cusolverH;

    cuFloatComplex *d_eigen_vec;
    float *d_eigen_val;

    float t_cuda  = 0;
    float t_armpl = 0;
};

void ver_armpl(lapack_complex_float *cov_arm) {
    float w[N];
    CHECK_LAPACK(LAPACKE_cheev(LAPACK_ROW_MAJOR, 'V', 'U', N, cov_arm, N, w));

    std::cout<<"LAPACK:"<<std::endl;
    for (int i=0;i<N;++i) {
        std::cout<<w[i]<<std::endl;
    }
}

void ver_cuda(cuFloatComplex *m_eigen_vec, float *m_eigen_val, const cuFloatComplex *d_cov, cusolverDnHandle_t &m_cusolverH) {
    cuFloatComplex *d_work;
    int *devInfo;
    int lwork = 0;

    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnCheevd_bufferSize(m_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, m_eigen_vec, N, m_eigen_val, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(cuComplex)));

    CHECK_CUDA(cudaMemcpy(m_eigen_vec, d_cov, N * N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUSOLVER(cusolverDnCheevd(m_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, m_eigen_vec, N, m_eigen_val, d_work, lwork, devInfo));
    auto end = std::chrono::high_resolution_clock::now();

    int info;
    CHECK_CUDA(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "Error: cuSOLVER operation failed with info = " << info << std::endl;
    }
}

int main() {

    evaluate_evd EV;
    int trials = 100;

    float t_cuda = 0, t_arm = 0;
    for (int i=0;i<trials;++i) {
        EV.run_cuda(h_cov);
        EV.run_armpl(h_cov_arm);
    }

    t_cuda = EV.getCudaTime();
    t_arm = EV.getArmplTime();

    std::cout<<"CUDA AVG. TIME:"<<std::endl;
    std::cout<<t_cuda/trials<<std::endl;

    std::cout<<"ARM AVG. TIME:"<<std::endl;
    std::cout<<t_arm/trials<<std::endl;
    return 0;
}
