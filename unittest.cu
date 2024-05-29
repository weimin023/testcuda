#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include "test_cuda.cu"

TEST(CUDAfunction, test_cuMath_vec) {
    cuMath::cuVec v(7, 2);
    auto a = v[0];

    thrust::device_vector<int> vec(10, 77);

    EXPECT_EQ(a, 2);
    EXPECT_EQ(vec[0], 77);

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

