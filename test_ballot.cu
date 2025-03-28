#include <stdio.h>

__global__ void vote_ballot(int *a, unsigned int *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int temp = a[tid];
    
    // 在 warp 內對 temp > 50 進行投票
    unsigned int ballot_result = __ballot_sync(0xFFFFFFFF, temp > 0);
    
    if (tid % 32 == 0) { // 讓 warp 的第一個線程存儲投票結果
        b[tid / 32] = ballot_result;
    }
}

int main() {
    const int N = 32;
    int h_a[N] = {0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0};
    unsigned int h_b[1] = {0};  // 一個 warp 只需要存一個 32-bit 結果

    int *d_a;
    unsigned int *d_b;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, sizeof(unsigned int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    vote_ballot<<<1, 32>>>(d_a, d_b, N);

    cudaMemcpy(h_b, d_b, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

    printf("Ballot result (binary): ");
    for (int i = 31; i >= 0; i--) {
        printf("%d", (h_b[0] >> i) & 1);
    }
    printf("\n");

    return 0;
}