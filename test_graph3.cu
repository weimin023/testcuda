#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void transpose_kernal(cuFloatComplex *in, cuFloatComplex *out, int m, int n) {

    extern __shared__ cuFloatComplex shm_trans[];
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < m && y < n) {
        shm_trans[x + y*m] = in[x + y*m];
        
        __syncthreads();

        out[x*n + y] = shm_trans[x + y*m];
    }
}

__global__ void mdl(int *n_out, const float *eigen_vals, int n) {
    __shared__ float shm[8];

    int tid = threadIdx.x;
    float threshold;

    if (tid < n) {
        shm[tid] = eigen_vals[tid];
        __syncthreads();

        if (tid == (n-1)) {
            threshold = shm[tid] * 0.007;
            int count = 0;
            for (int i = 0;i < n;++i) {
                count += (shm[i] >= threshold) ? 1 : 0;
            }
            *n_out = count;
        }
    }
}

cuFloatComplex randomComplexFloat() {
    float real = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f; // Random real part
    float imag = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f; // Random imaginary part
    return make_cuFloatComplex(real, imag);
}

class test {
public:
    test() {
        cudaMalloc((void**)&d_out, sizeof(int));
        cudaMalloc((void**)&d_val, 8 * sizeof(float));
        cudaMalloc((void**)&m_eigen_vec, 8 * 8 * sizeof(cuFloatComplex));
        cudaMalloc((void**)&m_eigen_vec_row_majored, 8 * 8 * sizeof(cuFloatComplex));
    }
    void build() {
        std:: cout << "=========GRAPH BUILDING=========" << std::endl;
        cudaStreamCreate(&m_stream);
        cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal);

        // to row-majored
        dim3 blockSize(16, 16);
        dim3 gridSize((8 + blockSize.x - 1) / blockSize.x, (8 + blockSize.y - 1) / blockSize.y);
        size_t shm_size = blockSize.x * blockSize.y * sizeof(cuFloatComplex);
        transpose_kernal<<<gridSize, blockSize, shm_size, m_stream>>>(m_eigen_vec, m_eigen_vec_row_majored, 8, 8);

        

        mdl<<<1, 8, 0, m_stream>>>(d_out, d_val, 8);

        cudaStreamEndCapture(m_stream, &m_graph);
        cudaGraphInstantiate(&m_graphExec, m_graph);
    }
    void evd() {
        cusolverDnCreate(&cusolverH);

        srand(static_cast<unsigned>(time(0)) + rand());
        thrust::host_vector<cuFloatComplex> h_matrix(64);

        // Fill the host vector with random cuFloatComplex values
        for (int i = 0; i < 64; ++i) {
            h_matrix[i] = randomComplexFloat();
        }

        thrust::device_vector<cuFloatComplex> cov = h_matrix;

        cudaMalloc(&devInfo, sizeof(int));

        cusolverDnCheevd_bufferSize(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 8,
            m_eigen_vec, 8, m_eigen_value, &lwork);
    
        cudaMalloc(&d_work, lwork * sizeof(cuComplex));

        cudaMemcpy(m_eigen_vec, cov.data().get(), 8 * 8 * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
        cusolverDnCheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 8, m_eigen_vec, 8, m_eigen_value, d_work, lwork, devInfo);
    
        int info;
        cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, m_stream);

        cudaMemcpy(d_val, m_eigen_value, 8 * sizeof(float), cudaMemcpyDeviceToDevice);

        thrust::host_vector<cuFloatComplex> h_vec(8*8);
        thrust::host_vector<float> h_val(8);
        cudaMemcpy(h_vec.data(), m_eigen_vec, 8 * 8 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val.data(), m_eigen_value, 8 * sizeof(float), cudaMemcpyDeviceToHost);

        std:: cout << "=========CHECK EIGENVALUES=========" << std::endl;
        for (int i=0;i<h_val.size();++i) {
            std::cout << h_val[i] << std::endl;
        }

        std:: cout << "=========CHECK EIGENVECS=========" << std::endl;
        for (int i=0;i<8;++i) {
            for (int j=0;j<8;++j) {
                int idx = j + i*8;
                std::cout << "(" << h_vec[idx].x << "," << h_vec[idx].y << "), ";
            }
            std::cout << std::endl;
        }
    }
    void run() {

        //-----------Prepare Input Data-----------
        evd();

        

        

        std:: cout << "=========GRAPH LAUNCHING=========" << std::endl;
        cudaGraphLaunch(m_graphExec, m_stream);
        cudaStreamSynchronize(m_stream);
        

        int h_out;
        cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Ans: " << h_out <<std::endl;

        std::cout << "=========" <<"Transposed Eigenvectors: \n";
        thrust::host_vector<cuFloatComplex> h_eigen_vec(8*8);
        cudaMemcpy(h_eigen_vec.data(), m_eigen_vec_row_majored, 8 * 8 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        for (int i=0;i<8;++i) {
            for (int j=0;j<8;++j) {
                int idx = j + i*8;
                std::cout << "(" << h_eigen_vec[idx].x << "," << h_eigen_vec[idx].y << "), ";
            }
            std::cout << std::endl;
        }
    }
private:
    cudaGraph_t m_graph;
    cudaGraphExec_t m_graphExec;
    cudaStream_t m_stream;

    cusolverDnHandle_t cusolverH;
    cuFloatComplex *m_eigen_vec;
    cuFloatComplex *m_eigen_vec_row_majored;
    float *m_eigen_value;

    cuFloatComplex *d_work;
    int *devInfo;
    int lwork = 0;

    float *d_val;
    int *d_out;
};

int main() {
    
    test tmp;
    tmp.build();
    tmp.run();
    tmp.run();
    tmp.run();
    tmp.run();
    
    return 0;
}