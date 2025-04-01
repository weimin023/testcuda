#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <unordered_set>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../test_ccl2.cu"

bool show(const cv::Mat &src) {

    if (src.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return false;
    }
    
    std::cout << "Image loaded successfully. Size: " 
              << src.rows << "x" << src.cols << std::endl;
    
    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::imshow("Display window", src);
    cv::waitKey(0);

    return true;
}

cv::Mat ccl_opencv(const cv::Mat &src) {

    cv::Mat labels;
    int numComponents = cv::connectedComponents(src, labels, 4, CV_16U);
    std::cout << "  Connected components (4-connectivity): " << numComponents - 1 << std::endl;

    return labels;
}

cv::Mat ccl_cuda(const cv::Mat &src) {

    int c = src.cols;
    int r = src.rows;

    thrust::device_vector<bool> d_input(c*r);
    thrust::copy(src.data, src.data + c * r, d_input.begin());

    thrust::device_vector<unsigned> d_labels(c*r, 0);
    thrust::device_vector<unsigned> d_output(c*r);

    // Launch CUDA kernel
    dim3 blocks((c + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (r + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);

    std::cout<<"blocks size: "<<blocks.x<<", "<<blocks.y<<std::endl;
    std::cout<<"threads size: "<<threads.x<<", "<<threads.y<<std::endl;

    
    
    stripLabelingMerging<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                       thrust::raw_pointer_cast(d_labels.data()),
                                       c, r, c);

    stripRelabeling<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                              thrust::raw_pointer_cast(d_labels.data()),
                                              c, r, c);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    thrust::host_vector<unsigned> h_labels = d_labels;
    std::unordered_map<int, cv::Vec3b> color_lut;

    for (int i=0;i<r;++i) {
        for (int j=0;j<c;++j) {
            int label = h_labels[i*c + j];
            if (label == 0) continue;
            if (color_lut.find(label) != color_lut.end()) continue;

            color_lut[label] = cv::Vec3b(rand() % 200, rand() % 200, rand() % 200);
            std::cout << cv::Vec3b(rand() % 200, rand() % 200, rand() % 200) << std::endl;
        }
    }

    // Create a colored visualization of the labels
    cv::Mat coloredLabels(r, c, CV_8UC3, cv::Scalar(0, 0, 0));
    int numComponents = color_lut.size();

    std::cout << "  CUDA Connected components (4-connectivity): " << numComponents << std::endl;

    // Create a random color for each label (except background which is black)
    std::vector<cv::Vec3b> colors(numComponents);
    colors[0] = cv::Vec3b(0, 0, 0); // Background is black
    
    // Apply the colors to the labeled image
    for (int y = 0; y < coloredLabels.rows; y++) {
        for (int x = 0; x < coloredLabels.cols; x++) {
            int label = h_labels[y * c + x];
            if (label > 0) {
                coloredLabels.at<cv::Vec3b>(y, x) = color_lut[label];  // Assign color for the label
            }
        }
    }

    return coloredLabels;
}

int main() {
    cv::Mat img1 = cv::imread("../ccl_testcases/ccl_test3_wide.png", cv::IMREAD_GRAYSCALE);
    
    cv::Mat trans;
    cv::transpose(img1, trans);

    show(trans);
    
    ccl_opencv(trans);
    auto label_res = ccl_cuda(trans);
    show(label_res);
    
    return 0;
}