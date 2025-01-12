#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

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

    std::vector<std::vector<int>> c_naive(M, std::vector<int>(N)), c_naive_reg(M, std::vector<int>(N)), c_multithread(M, std::vector<int>(N));


    auto start = std::chrono::high_resolution_clock::now();
    gemm_naive(A_mat, B_mat, c_naive);
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_regMN(A_mat, B_mat, c_naive_reg);
    end = std::chrono::high_resolution_clock::now();
    auto naive_reg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    gemm_multithreads(A_mat, B_mat, c_multithread);
    end = std::chrono::high_resolution_clock::now();
    auto mt_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    

    bool flg = 0;
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            if ((c_naive[i][j] != c_naive_reg[i][j]) || (c_naive[i][j] != c_multithread[i][j])) {
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
    std::cout << "Naive + Reg Elapsed Time: " << naive_reg_time.count() << "ms" << std::endl;
    std::cout << "Multithreads Elapsed Time: " << mt_time.count() << "ms" << std::endl;

    

    return 0;

}