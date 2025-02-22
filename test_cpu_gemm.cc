#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <omp.h>
#include <immintrin.h>  // AVX2 intrinsics

void get_rand_vec(std::vector<std::vector<int>> &A, std::vector<std::vector<int>> &B, int M, int N, int K) {
    A.resize(M, std::vector<int>(K));
    B.resize(N, std::vector<int>(K));

    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(10); // Mersenne Twister random number generator
    std::uniform_int_distribution<> dis(0, 100); // Range of random numbers

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i][j] = dis(gen); // Populate with random numbers
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            B[i][j] = dis(gen); // Populate with random numbers
        }
    }
}

void show(const std::vector<std::vector<int>> &vec) {
    int M = vec.size();
    int N = vec[0].size();

    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            std::cout<<vec[i][j]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void gemm_naive(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    int M = A.size();
    int N = B.size();
    int K = A[0].size();

    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            for (int k=0;k<K;++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void gemm_naive_reorder(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    int M = A.size();
    int N = B.size();
    int K = A[0].size();

    for (int i=0;i<M;++i) {
        for (int k=0;k<K;++k) {
            for (int j=0;j<N;++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

template <int M, int N, int K, int tile_size>
void gemm_tile_two_dim(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    // Choose tile sizes based on typical L1 cache size
    // These values should be tuned based on your specific hardware
    
    // Iterate over tiles
    for (int i0 = 0; i0 < M; i0 += tile_size) {
        for (int k0 = 0; k0 < K; k0 += tile_size) {
            for (int j0 = 0; j0 < N; j0 += tile_size) {
                // Define tile boundaries
                const int i_end = std::min(i0 + tile_size, M);
                const int j_end = std::min(j0 + tile_size, N);
                const int k_end = std::min(k0 + tile_size, K);
                
                // Compute on the current tile
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        for (int j = j0; j < j_end; j++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

template <int rows, int cols, int ks, int tile_size>
void gemm_tile_k_dim(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    for (int innerTile = 0; innerTile < ks; innerTile += tile_size) {
        for (int row = 0; row < rows; ++row) {
            int innerTileEnd = std::min(ks, innerTile + tile_size);

            for (int k = innerTile; k < innerTileEnd; ++k) {
                for (int col = 0; col < cols; ++col) {
                    C[row][col] += A[row][k] * B[k][col];
                }
            }
        }
    }
}

template<int M, int N, int K, int TILE_SIZE>
void gemm_tile_opt(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    std::vector<int> A_local(TILE_SIZE*TILE_SIZE);
    std::vector<int> B_local(TILE_SIZE*TILE_SIZE);
    std::vector<int> C_local(TILE_SIZE*TILE_SIZE);

    // Iterate over tiles
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {
        const int i_end = std::min(i0 + TILE_SIZE, M);
        
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            const int j_end = std::min(j0 + TILE_SIZE, N);
            
            // Clear C_local
            std::fill(C_local.begin(), C_local.end(), 0);
            
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                const int k_end = std::min(k0 + TILE_SIZE, K);
                
                // Copy tiles to local buffers
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        A_local[(i - i0) * TILE_SIZE + (k - k0)] = A[i][k];
                    }
                }
                
                for (int k = k0; k < k_end; k++) {
                    for (int j = j0; j < j_end; j++) {
                        B_local[(k - k0) * TILE_SIZE + (j - j0)] = B[k][j];
                    }
                }
                
                // Compute on local tiles
                for (int i = 0; i < i_end - i0; i++) {
                    for (int k = 0; k < k_end - k0; k++) {
                        const int a_val = A_local[i * TILE_SIZE + k];
                        for (int j = 0; j < j_end - j0; j++) {
                            C_local[i * TILE_SIZE + j] += a_val * B_local[k * TILE_SIZE + j];
                        }
                    }
                }
            }
            
            // Write back C_local to C
            for (int i = i0; i < i_end; i++) {
                for (int j = j0; j < j_end; j++) {
                    C[i][j] = C_local[(i - i0) * TILE_SIZE + (j - j0)];
                }
            }
        }
    }
}

void gemm_naive_AVX2(const int *A, const int *B, int *C) {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    for (int i=0;i<M;++i) {
        for (int k=0;k<K;++k) {
            for (int j=0;j<N;j+=8) {
                // Load 8 values from C (initially, this should be zeroed out)
                __m256i c_vec = _mm256_loadu_si256((__m256i*)&C[i * N + j]);

                // Load a single value from A[i][k], broadcast it to all elements of a vector
                __m256i a_vec = _mm256_set1_epi32(A[i * K + k]);

                // Load 8 values from B[k][j]
                __m256i b_vec = _mm256_loadu_si256((__m256i*)&B[k * N + j]);

                // Multiply A[i][k] by B[k][j] for 8 elements and add the result to C[i][j]
                __m256i result = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a_vec, b_vec));

                // Store the updated result back into C
                _mm256_storeu_si256((__m256i*)&C[i * N + j], result);
            }
        }
    }
}

template <int rows, int cols, int ks, int tile_size>
void gemm_tile_k_dim_openMP(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    //#pragma omp parallel for collapse(2) 
    for (int innerTile = 0; innerTile < ks; innerTile += tile_size) {
        for (int row = 0; row < rows; ++row) {
            int innerTileEnd = std::min(ks, innerTile + tile_size);

            for (int k = innerTile; k < innerTileEnd; ++k) {
                #pragma omp simd
                for (int col = 0; col < cols; ++col) {
                    C[row][col] += A[row][k] * B[k][col];
                }
            }
        }
    }
}

void gemm_regMN(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
    int M = A.size();
    int N = B.size();
    int K = A[0].size();

    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            int k_sum = 0;
            for (int k=0;k<K;++k) {
                k_sum += A[i][k] * B[k][j];
            }
            C[i][j] = k_sum;
        }
    }
}

#define TILE 10
void worker(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B,
            std::vector<std::vector<int>> &C, int rowStart, int colStart, int K) {
    for (int i = 0; i < TILE; ++i) {
        for (int j = 0; j < TILE; ++j) {
            int row = rowStart + i;
            int col = colStart + j;

            // 检查边界
            if (row < C.size() && col < C[0].size()) {
                int reg = 0;
                for (int k = 0; k < K; ++k) {
                    reg += A[row][k] * B[k][col];
                }
                C[row][col] = reg;
            }
        }
    }
}

// 多线程矩阵乘法
void gemm_multithreads(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B,
                       std::vector<std::vector<int>> &C) {
    int M = A.size();
    int N = B[0].size();
    int K = A[0].size();

    // 创建线程
    std::vector<std::thread> thread_vec;

    // 按块分配任务
    for (int i = 0; i < M; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            thread_vec.emplace_back(worker, std::cref(A), std::cref(B), std::ref(C), i, j, K);
        }
    }

    // 等待所有线程完成
    for (auto &t : thread_vec) {
        t.join();
    }
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::vector<std::vector<int>> A_mat, B_mat;
    get_rand_vec(A_mat, B_mat, M, N, K);

    std::vector<int> A_mat_flatten(M*K), B_mat_flatten(N*K);
    for (int i=0;i<M;++i) {
        for (int j=0;j<K;++j) {
            A_mat_flatten[i*K + j] = A_mat[i][j];
        }
    }

    for (int i=0;i<K;++i) {
        for (int j=0;j<N;++j) {
            B_mat_flatten[i*N + j] = B_mat[i][j];
        }
    }

    std::vector<std::vector<int>> c_naive(M, std::vector<int>(N)), c_naive_reorder(M, std::vector<int>(N)), c_naive_reg(M, std::vector<int>(N)), c_multithread(M, std::vector<int>(N)), c_tile(M, std::vector<int>(N)), c_tile_openMP(M, std::vector<int>(N)), c_tile2(M, std::vector<int>(N)), c_tile3(M, std::vector<int>(N));
    std::vector<int> c_tile_AVX2(M*N);

    auto start = std::chrono::high_resolution_clock::now();
    gemm_naive(A_mat, B_mat, c_naive);
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_naive_reorder(A_mat, B_mat, c_naive_reorder);
    end = std::chrono::high_resolution_clock::now();
    auto naive_reordered_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_regMN(A_mat, B_mat, c_naive_reg);
    end = std::chrono::high_resolution_clock::now();
    auto naive_reg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_tile_k_dim<M, N, K, 32>(A_mat, B_mat, c_tile);
    end = std::chrono::high_resolution_clock::now();
    auto naive_tile_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_tile_k_dim_openMP<M, N, K, 32>(A_mat, B_mat, c_tile_openMP);
    end = std::chrono::high_resolution_clock::now();
    auto naive_tile_openMP_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_naive_AVX2(A_mat_flatten.data(), B_mat_flatten.data(), c_tile_AVX2.data());
    end = std::chrono::high_resolution_clock::now();
    auto naive_tile_AVX2_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_multithreads(A_mat, B_mat, c_multithread);
    end = std::chrono::high_resolution_clock::now();
    auto mt_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_tile_two_dim<M, N, K, 32>(A_mat, B_mat, c_tile2);
    end = std::chrono::high_resolution_clock::now();
    auto naive_tile2_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_tile_opt<M, N, K, 32>(A_mat, B_mat, c_tile3);
    end = std::chrono::high_resolution_clock::now();
    auto naive_tile3_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    

    bool flg = 0;
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            if ((c_naive[i][j] != c_naive_reg[i][j]) || (c_naive[i][j] != c_multithread[i][j]) || (c_naive[i][j] != c_naive_reorder[i][j]) || (c_naive[i][j] != c_tile[i][j]) || (c_naive[i][j] != c_tile_openMP[i][j]) || (c_naive[i][j] != c_tile_AVX2[i * N + j]) || (c_naive[i][j] != c_tile2[i][j]) || (c_naive[i][j] != c_tile3[i][j])) {
                std::cout << "FAILED." << std::endl;
                return 0;
            }
        }
    }

    /*show(c_naive);
    show(c_naive_reg);
    show(c_multithread);*/
    
    std::cout << "SUCCESS." << std::endl;
    std::cout << "Naive Elapsed Time: " << naive_time.count() << "ms" << std::endl;
    std::cout << "Naive reorder Elapsed Time: " << naive_reordered_time.count() << "ms" << std::endl;
    std::cout << "Naive reorder + AVX2 Elapsed Time: " << naive_tile_AVX2_time.count() << "ms" << std::endl;
    std::cout << "Naive + Reg Elapsed Time: " << naive_reg_time.count() << "ms" << std::endl;
    std::cout << "Naive + Tile Elapsed Time: " << naive_tile_time.count() << "ms" << std::endl;
    std::cout << "Naive + Tile2 Elapsed Time: " << naive_tile2_time.count() << "ms" << std::endl;
    std::cout << "Naive + Tile3 Elapsed Time: " << naive_tile3_time.count() << "ms" << std::endl;
    std::cout << "Naive + Tile + openMP Elapsed Time: " << naive_tile_openMP_time.count() << "ms" << std::endl;
    std::cout << "Multithreads Elapsed Time: " << mt_time.count() << "ms" << std::endl;

    

    return 0;

}