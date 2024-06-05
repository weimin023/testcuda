#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include "test_cuda.cu"
#include "test_fft.cpp"

#include <random>


/*TEST(CUDAfunction, test_cuMath_vec) {
    cuMath::cuVec v(7, 2);
    auto a = v[0];

    thrust::device_vector<int> vec(10, 77);

    EXPECT_EQ(a, 2);
    EXPECT_EQ(vec[0], 77);
}*/


TEST(nsight, fft) {
    int N_ = 5000000;
    testFFT fft;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> unif(0, 200);

    std::vector<int> elapse_t;

    for (int k=0;k<20;++k) {
        std::vector<std::complex<double>> data;
        for (int i=0;i<N_;++i) {
            data.push_back({unif(generator), unif(generator)});
        }

        elapse_t.push_back(fft.run(data));
    }
    
}

TEST(opencv, open) {
    
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

