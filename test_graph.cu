#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>

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

const char *cusolverGetErrorString(cusolverStatus_t error) {
    switch (error) {
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
    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    default:
        return "UNKNOWN_ERROR";
    }
}
// Custom CUDA kernel (simple operation)
__global__ void myKernel(cuComplex *data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        data[idx].x += 1.0f;
    }
}

int main() {
    const int m = 3;
    cudaStream_t stream;
    cusolverDnHandle_t cusolverH;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));

    

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cuFloatComplex *d_work;
    int *devInfo;
    int lwork = 0;
    cuFloatComplex *m_eigen_vec, *m_eigen_vec_row_majored;
    float *m_eigen_value;

    CHECK_CUDA(cudaMallocAsync(&m_eigen_vec, m*m*sizeof(cuFloatComplex), stream));
    CHECK_CUDA(cudaMallocAsync(&m_eigen_vec_row_majored, m*m*sizeof(cuFloatComplex), stream));
    CHECK_CUDA(cudaMallocAsync(&m_eigen_value, m*sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&devInfo, sizeof(int), stream));

    CHECK_CUSOLVER(cusolverDnCheevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, m_eigen_vec, m, m_eigen_value, &lwork));

    
    CHECK_CUDA(cudaMallocAsync(&d_work, lwork * sizeof(cuFloatComplex), stream));
    

    cuFloatComplex h_A[m * m] = {
        {1, 0}, {2, 0}, {3, 0},
        {2, 0}, {5, 0}, {6, 0},
        {3, 0}, {6, 0}, {9, 0}
    };
    cuFloatComplex *d_A;
    CHECK_CUDA(cudaMallocAsync(&d_A, m * m * sizeof(cuFloatComplex), stream));
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, m * m * sizeof(cuFloatComplex), cudaMemcpyHostToDevice, stream));
    

    // NODE A: EVD
    CHECK_CUDA(cudaMemcpyAsync(m_eigen_vec, d_A, m * m * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice, stream));

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK_CUSOLVER(cusolverDnCheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, d_A, m, m_eigen_value, d_work, lwork, devInfo));

    // NODE B: My Kernel
    
    int blockSize = 32;
    int gridSize = (m * m + blockSize - 1) / blockSize;
    myKernel<<<gridSize, blockSize, 0, stream>>>(d_A, m * m);

    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return 0;
}
