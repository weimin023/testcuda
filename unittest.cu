#include <gtest/gtest.h>
#include "test_cuda.cu"

TEST(CUDAfunction, test_cuMath_vec) {
    cuMath::cuVec v(7, 2);
    auto a = v[0];
    EXPECT_EQ(a, 2);

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

