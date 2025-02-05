const int M = 1024;
const int N = 1024;
const int K = 1024;

const int MI = 128;
const int NI = 128;
const int KI = 32;

// warp
const int MII = 64;
const int NII = 64;
const int KII = 16;

const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K) {
    extern __shared__ uint8_t shared_mem[];
    half *SA = reinterpret_cast<half*>(shared_mem);
    half *SB = reinterpret_cast<half*>(shared_mem + MI * KI * sizeof(half));
    float *SC = reinterpret_cast<float*>(shared_mem);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> fragA[MII/wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> fragB[NII/wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> accum[MII/wmmaM * NII/wmmaN];

    for (int mii = 0; mii < MII/wmmaM; ++mii) {
        for (int nii = 0; nii < NII/wmmaN; ++nii) {
            nvcuda::wmma::fill_fragment(accum[mii * (NII/wmmaN) + nii], 0.0);
        }
    }

    for (int ko = 0; ko < K/KI; ++ko) {

        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, N, K, ko);
        __syncthreads();

        for (int ki = 0; ki < KI/KII; ++ki) {
            
            loadFragA(fragA, SA, ki);
            loadFragB(fragB, SB, ki);

            for (int mii = 0; mii < MII/wmmaM; ++mii) {
                for (int nii = 0; nii < NII/wmmaN; ++nii) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(accum[mii * (NII/wmmaN) + nii], fragA[mii], fragB[nii], accum[mii * (NII/wmmaN) + nii]);
                }
            }
        }
    }
    storeAccm(SC, accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}


__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = A[(by * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki) {
    // load 64 * 16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB() {

}

__device__ void storeAccm() {

}

__device__ void storeSmemC() {

}

int main() {
    dim3 dimBlock(32, 2, 2);
    dim3 dimGrid(N / 128, M / 128);
}