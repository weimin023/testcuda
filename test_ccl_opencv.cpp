#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>

// Function to save a binary matrix as an image file
void saveBinaryMatrix(const cv::Mat& matrix, const std::string& filename) {
    // Save the original binary image
    cv::imwrite(filename, matrix * 255); // Scale 0-1 to 0-255
    std::cout << "Saved " << filename << " (" << matrix.cols << "x" << matrix.rows << ")" << std::endl;
    
    // Run connected components
    cv::Mat labels;
    int numComponents = cv::connectedComponents(matrix, labels, 4, CV_16U);
    std::cout << "  Connected components (4-connectivity): " << numComponents - 1 << std::endl; // Subtract 1 for background
    
    // Create a colored visualization of the labels
    cv::Mat coloredLabels = cv::Mat::zeros(labels.size(), CV_8UC3);
    
    // Create a random color for each label (except background which is black)
    std::vector<cv::Vec3b> colors(numComponents);
    colors[0] = cv::Vec3b(0, 0, 0); // Background is black
    
    for (int i = 1; i < numComponents; i++) {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }
    
    // Apply the colors to the labeled image
    for (int y = 0; y < coloredLabels.rows; y++) {
        for (int x = 0; x < coloredLabels.cols; x++) {
            int label = labels.at<uint16_t>(y, x);
            coloredLabels.at<cv::Vec3b>(y, x) = colors[label];
        }
    }
    
    // Save the colored labels image
    std::string labelFilename = filename.substr(0, filename.find_last_of('.')) + "_labels.png";
    cv::imwrite(labelFilename, coloredLabels);
    std::cout << "  Saved label visualization: " << labelFilename << std::endl;
    
    // Display the original image and the labeled image
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Labels", cv::WINDOW_NORMAL);
    cv::imshow("Original", matrix * 255);
    cv::imshow("Labels", coloredLabels);
    cv::waitKey(100); // Brief pause to show the image, adjust as needed
}

// Test matrix 1: Contains various basic shapes repeated across a wide canvas
cv::Mat createTestMatrix1(int width) {
    int rows = 32;
    int cols = width;
    cv::Mat matrix = cv::Mat::zeros(rows, cols, CV_8UC1);
    
    // Create pattern units and repeat them across the width
    for (int x = 0; x < cols; x += 128) {
        // Rectangle
        cv::rectangle(matrix, cv::Point(x + 5, 5), cv::Point(x + 25, 12), cv::Scalar(1), -1);
        
        // Circle
        cv::circle(matrix, cv::Point(x + 40, 10), 6, cv::Scalar(1), -1);
        
        // Horizontal line
        cv::line(matrix, cv::Point(x + 60, 10), cv::Point(x + 90, 10), cv::Scalar(1), 2);
        
        // Vertical line
        cv::line(matrix, cv::Point(x + 100, 5), cv::Point(x + 100, 25), cv::Scalar(1), 2);
        
        // L shape
        if (x + 120 < cols) {
            cv::line(matrix, cv::Point(x + 110, 18), cv::Point(x + 110, 28), cv::Scalar(1), 2);
            cv::line(matrix, cv::Point(x + 110, 28), cv::Point(x + 120, 28), cv::Scalar(1), 2);
        }
    }
    
    return matrix;
}

// Test matrix 2: Contains text patterns and small components across a wide canvas
cv::Mat createTestMatrix2(int width) {
    int rows = 32;
    int cols = width;
    cv::Mat matrix = cv::Mat::zeros(rows, cols, CV_8UC1);
    
    // Add repeated text-like patterns
    for (int x = 0; x < cols; x += 150) {
        // "C"
        if (x + 30 < cols)
            cv::ellipse(matrix, cv::Point(x + 20, 16), cv::Size(8, 12), 0, -30, 210, cv::Scalar(1), 2);
        
        // "C"
        if (x + 60 < cols)
            cv::ellipse(matrix, cv::Point(x + 50, 16), cv::Size(8, 12), 0, -30, 210, cv::Scalar(1), 2);
        
        // "L"
        if (x + 90 < cols) {
            cv::line(matrix, cv::Point(x + 80, 6), cv::Point(x + 80, 24), cv::Scalar(1), 2);
            cv::line(matrix, cv::Point(x + 80, 24), cv::Point(x + 90, 24), cv::Scalar(1), 2);
        }
        
    }
    
    return matrix;
}

// Test matrix 3: Random blobs and complex patterns across a wide canvas
cv::Mat createTestMatrix3(int width) {
    int rows = 32;
    int cols = width;
    cv::Mat matrix = cv::Mat::zeros(rows, cols, CV_8UC1);
    
    std::default_random_engine generator;
    std::normal_distribution<double> size_distribution(5.0, 2.0);
    
    // Create blobs at regular intervals
    for (int x = 10; x < cols; x += 60) {
        // Vary the y position between top and bottom half
        int y = (x / 60 % 2 == 0) ? 8 : 24;
        
        // Create a random blob
        int a = std::max(3, (int)std::abs(size_distribution(generator)));
        int b = std::max(3, (int)std::abs(size_distribution(generator)));
        double angle = (rand() % 180) * CV_PI / 180.0;
        
        cv::ellipse(matrix, cv::Point(x, y), cv::Size(a, b), angle, 0, 360, cv::Scalar(1), -1);
        
        // Randomly connect some blobs with lines
        if (rand() % 3 == 0 && x + 60 < cols) {
            int nextY = (x / 60 % 2 == 0) ? 24 : 8; // Opposite y for next blob
            cv::line(matrix, cv::Point(x + a, y), cv::Point(x + 60 - a, nextY), cv::Scalar(1), 1);
        }
    }
    
    // Add some horizontal structures at different positions
    for (int x = 0; x < cols; x += 200) {
        if (x + 150 < cols) {
            // Draw chain-like structures
            for (int i = 0; i < 20; i++) {
                cv::circle(matrix, cv::Point(x + 100 + i*5, 16 + (i%2)*2), 2, cv::Scalar(1), -1);
                if (i < 19) {
                    cv::line(matrix, cv::Point(x + 100 + i*5 + 2, 16 + (i%2)*2), 
                            cv::Point(x + 100 + (i+1)*5 - 2, 16 + ((i+1)%2)*2), 
                            cv::Scalar(1), 1);
                }
            }
        }
    }
    
    return matrix;
}

int main() {
    // Set width to at least 512
    int width = 512;
    
    // Create and save the test matrices
    cv::Mat test1 = createTestMatrix1(width);
    cv::Mat test2 = createTestMatrix2(width);
    cv::Mat test3 = createTestMatrix3(width);
    
    saveBinaryMatrix(test1, "ccl_test1_wide.png");
    saveBinaryMatrix(test2, "ccl_test2_wide.png");
    saveBinaryMatrix(test3, "ccl_test3_wide.png");
    
    // Display the matrices (resized for viewing)
    cv::Mat display1, display2, display3;
    cv::resize(test1 * 255, display1, cv::Size(640, 32), 0, 0, cv::INTER_NEAREST);
    cv::resize(test2 * 255, display2, cv::Size(640, 32), 0, 0, cv::INTER_NEAREST);
    cv::resize(test3 * 255, display3, cv::Size(640, 32), 0, 0, cv::INTER_NEAREST);
    
    cv::namedWindow("Test Matrix 1", cv::WINDOW_NORMAL);
    cv::imshow("Test Matrix 1", display1);
    
    cv::namedWindow("Test Matrix 2", cv::WINDOW_NORMAL);
    cv::imshow("Test Matrix 2", display2);
    
    cv::namedWindow("Test Matrix 3", cv::WINDOW_NORMAL);
    cv::imshow("Test Matrix 3", display3);
    
    cv::waitKey(0);
    
    // Create binary file versions for direct loading
    std::ofstream file1("ccl_test1.bin", std::ios::binary);
    file1.write(reinterpret_cast<const char*>(test1.data), test1.total() * test1.elemSize());
    file1.close();
    
    std::ofstream file2("ccl_test2.bin", std::ios::binary);
    file2.write(reinterpret_cast<const char*>(test2.data), test2.total() * test2.elemSize());
    file2.close();
    
    std::ofstream file3("ccl_test3.bin", std::ios::binary);
    file3.write(reinterpret_cast<const char*>(test3.data), test3.total() * test3.elemSize());
    file3.close();
    
    std::cout << "Binary files saved: ccl_test1.bin, ccl_test2.bin, ccl_test3.bin" << std::endl;
    
    return 0;
}