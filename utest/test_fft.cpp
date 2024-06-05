#include <fftw3.h>

#include <cmath>
#include <complex>
#include <vector>
#include <omp.h>
#include <chrono>

#include <cufft.h>
#include <cuda_runtime.h>

#define N 1024

const double PI = 3.141592653589793238460;

class testFFT {
  public:
    // testFFT() {}
    int run(const std::vector<std::complex<double>> &in) {
        /*auto start_time_a = std::chrono::high_resolution_clock::now();
        auto a = fft1d(in);
        auto end_time_a = std::chrono::high_resolution_clock::now();

        auto duration_a = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_a - start_time_a);
        std::cout << "fft1d execution time: " << duration_a.count() << " milliseconds" << std::endl;*/
        
        auto start_time_b = std::chrono::high_resolution_clock::now();
        auto b = fft1d_cu_stream(in);
        auto end_time_b = std::chrono::high_resolution_clock::now();
        
        auto duration_b = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_b - start_time_b);
        std::cout << "fft1d_cu_stream time: " << duration_b.count() << " milliseconds" << std::endl;
        

        /*auto start_time_c = std::chrono::high_resolution_clock::now();
        auto c = fft1d_cu_stream(in);
        auto end_time_c = std::chrono::high_resolution_clock::now();
        
        auto duration_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_c - start_time_c);
        std::cout << "fft1d_cu_stream time: " << duration_c.count() << " milliseconds" << std::endl;*/
        

        return duration_b.count();
    }

  private:
    std::vector<std::complex<double>> fft1d(const std::vector<std::complex<double>> &in) {
        
        std::vector<std::complex<double>> data = in;

        omp_set_num_threads(4);
        fft1d_worker(data);

        return data;
    }
    void fft1d_worker(std::vector<std::complex<double>> &in) {
        int len = in.size();
        if (len <= 1) return;

        std::vector<std::complex<double>> a1(len / 2), a2(len / 2);

        #pragma omp parallel for
        for (int i = 0; i < len / 2; ++i) {
            a1[i] = in[i * 2];
            a2[i] = in[i * 2 + 1];
        }

        fft1d_worker(a1);
        fft1d_worker(a2);

        #pragma omp parallel for
        for (int k = 0; k < len / 2; ++k) {
            std::complex<double> factor = std::polar(1.0, -2 * PI * k / len) * a2[k];
            in[k] = a1[k] + factor;
            in[k + len / 2] = a1[k] - factor;
        }
    }
    std::vector<std::complex<double>> fft1d_gold(const std::vector<std::complex<double>> &in) {
        int len = in.size();

        fftw_complex *input = fftw_alloc_complex(len);
        fftw_complex *output = fftw_alloc_complex(len);

        for (int i=0;i<len;++i) {
            input[i][0] = in[i].real();
            input[i][1] = in[i].imag();
        }

        auto plan = fftw_plan_dft_1d(len, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        std::vector<std::complex<double>> out(len);
        for (int i = 0; i < len; ++i) {
            out[i] = std::complex<double>(output[i][0], output[i][1]);
        }

        fftw_destroy_plan(plan);
        fftw_free(input);
        fftw_free(output);

        return out;
    }
    std::vector<std::complex<double>> fft1d_cu(const std::vector<std::complex<double>> &in) {
        int size_ = in.size();
        cufftComplex *hostData = new cufftComplex[size_];
        
        for (int i=0;i<size_;++i) {
            hostData[i].x = in[i].real();
            hostData[i].y = in[i].imag();
        }

        cufftComplex *deviceData;
        cudaMalloc((void**)&deviceData, sizeof(cufftComplex)*size_);
        
        cudaMemcpy(deviceData, hostData, sizeof(cufftComplex)*size_, cudaMemcpyHostToDevice);

        cufftHandle plan;
        cufftPlan1d(&plan, size_, CUFFT_C2C, 1);
        if (cufftExecC2C(plan, deviceData, deviceData, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT EXEC ERROR.");
            return std::vector<std::complex<double>>{};
        }

        cudaMemcpy(hostData, deviceData, sizeof(cufftComplex)*size_, cudaMemcpyDeviceToHost);

        std::vector<std::complex<double>> output;
        for (int i=0;i<size_;++i) {
            output.push_back({hostData[i].x, hostData[i].y});
        }

        // Cleanup
        cufftDestroy(plan);
        cudaFree(deviceData);
        delete[] hostData;

        return output;
    }
    std::vector<std::complex<double>> fft1d_cu_stream(const std::vector<std::complex<double>> &in) {
        int size_ = in.size();
        cufftComplex *hostData = new cufftComplex[size_];
        
        for (int i=0;i<size_;++i) {
            hostData[i].x = in[i].real();
            hostData[i].y = in[i].imag();
        }

        cufftComplex *deviceData;
        cudaMalloc((void**)&deviceData, sizeof(cufftComplex)*size_);

        cudaStream_t stream_;
        cudaStreamCreate(&stream_);
        
        cudaMemcpyAsync(deviceData, hostData, sizeof(cufftComplex)*size_, cudaMemcpyHostToDevice, stream_);

        cufftHandle plan;
        cufftPlan1d(&plan, size_, CUFFT_C2C, 1);
        cufftSetStream(plan, stream_);
        if (cufftExecC2C(plan, deviceData, deviceData, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT EXEC ERROR.");
            return std::vector<std::complex<double>>{};
        }

        cudaMemcpyAsync(hostData, deviceData, sizeof(cufftComplex)*size_, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        std::vector<std::complex<double>> output;
        for (int i=0;i<size_;++i) {
            output.push_back({hostData[i].x, hostData[i].y});
        }

        // Cleanup
        cufftDestroy(plan);
        cudaFree(deviceData);
        delete[] hostData;
        cudaStreamDestroy(stream_);

        return output;
    }
    
};
