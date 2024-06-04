#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include "test_cuda.cu"
#include "test_fft.cpp"

/*TEST(CUDAfunction, test_cuMath_vec) {
    cuMath::cuVec v(7, 2);
    auto a = v[0];

    thrust::device_vector<int> vec(10, 77);

    EXPECT_EQ(a, 2);
    EXPECT_EQ(vec[0], 77);
}*/

TEST(nsight, fft) {
    testFFT fft;
    std::vector<std::complex<double>> data = {
        {1, 0}, {7, 0}, {8, 0}, {2, 0},
        {1, 0}, {2, 0}, {8, 0}, {2, 0},
        {3, 0}, {4, 0}, {2, 0}, {8, 0},
        {9, 0}, {8, 0}, {1, 0}, {2, 0}
    };
    fft.run(data);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

