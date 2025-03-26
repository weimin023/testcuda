#include <iostream>
#include <cuda_runtime.h>

__global__ void bitOperationsKernel(int *input, int *leadingZeros, int *firstSetBit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = input[idx];
        leadingZeros[idx] = __clz(val);
        firstSetBit[idx] = __ffs(val);
    }
}

int main() {
    const int N = 5;
    int h_input[N] = {static_cast<int>(0x80000000), 0x00000001, 0x0000000F, 0x00010000, 0}; 
    int h_leadingZeros[N], h_firstSetBit[N];

    int *d_input, *d_leadingZeros, *d_firstSetBit;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_leadingZeros, N * sizeof(int));
    cudaMalloc(&d_firstSetBit, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    bitOperationsKernel<<<1, N>>>(d_input, d_leadingZeros, d_firstSetBit, N);
    cudaMemcpy(h_leadingZeros, d_leadingZeros, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_firstSetBit, d_firstSetBit, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_leadingZeros);
    cudaFree(d_firstSetBit);

    std::cout << "Value\tLeading Zeros\tFirst Set Bit\n";
    for (int i = 0; i < N; i++) {
        std::cout << std::hex << "0x" << h_input[i] << "\t" 
                  << std::dec << h_leadingZeros[i] << "\t\t"
                  << h_firstSetBit[i] << "\n";
    }

    return 0;
}
