#include <iostream>
#include <vector>
#include <fstream>

#include <vector>
#include <algorithm>

// Create a 64x64 test pattern with multiple connected components
void createTestPattern(std::vector<unsigned char>& image, int width, int height) {
    // Ensure the size matches the expected dimensions
    image.resize(width * height, 0);
    
    // Shape 1: Small rectangle in the top-left corner
    for (int row = 10; row < 20; row++) {
        for (int col = 10; col < 25; col++) {
            image[row * width + col] = 255;
        }
    }
    
    // Shape 2: L-shape in the top-right corner
    for (int row = 10; row < 30; row++) {
        for (int col = 45; col < 55; col++) {
            image[row * width + col] = 255;
        }
    }
    for (int row = 20; row < 30; row++) {
        for (int col = 55; col < 60; col++) {
            image[row * width + col] = 255;
        }
    }
    
    // Shape 3: Circle in the bottom-left corner
    int centerX = 15;
    int centerY = 50;
    int radius = 8;
    for (int row = centerY - radius; row <= centerY + radius; row++) {
        for (int col = centerX - radius; col <= centerX + radius; col++) {
            if ((row - centerY) * (row - centerY) + (col - centerX) * (col - centerX) <= radius * radius) {
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    image[row * width + col] = 255;
                }
            }
        }
    }
    
    // Shape 4: Cross in the bottom-right corner
    for (int row = 40; row < 60; row++) {
        for (int col = 30; col < 34; col++) {
            image[row * width + col] = 255;
        }
    }
    for (int row = 50; row < 54; row++) {
        for (int col = 25; col < 40; col++) {
            image[row * width + col] = 255;
        }
    }
}


// Save the test image to a file (simple PGM format)
void saveImage(const std::vector<unsigned char>& image, int width, int height, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write PGM header
    file << "P5\n" << width << " " << height << "\n255\n";
    
    // Write image data
    file.write(reinterpret_cast<const char*>(image.data()), width * height);
    
    file.close();
}

// Function to save as a simple binary file for direct loading
void saveBinary(const std::vector<unsigned char>& image, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file.write(reinterpret_cast<const char*>(image.data()), image.size());
    file.close();
}

// Print a preview of the image in ASCII to console
void previewImage(const std::vector<unsigned char>& image, int width, int height) {
    std::cout << "Image Preview:\n";
    
    int step_y = std::max(1, height / 64);
    int step_x = std::max(1, width / 64);

    for (int row = 0; row < height; row += step_y) {
        for (int col = 0; col < width; col += step_x) {
            char pixel = (image[row * width + col] > 0) ? '#' : ' ';
            std::cout << pixel;
        }
        std::cout << '\n';
    }
}


int main() {
    // Define image dimensions
    int width = 64;
    int height = 64;
    
    // Create the test image
    std::vector<unsigned char> image(width * height);
    createTestPattern(image, width, height);
    
    // Save the image as PGM
    saveImage(image, width, height, "test_input.pgm");
    
    // Save as binary for direct loading
    saveBinary(image, "test_input.bin");
    
    // Preview the image
    previewImage(image, width, height);
    
    std::cout << "\nTest image created and saved as 'test_input.pgm' and 'test_input.bin'\n";
    std::cout << "The image contains 4 distinct shapes that should be labeled as separate components.\n";
    
    return 0;
}